from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score

import logging
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class HybridTrainer(BaseTrainer):
    """Trainer for hybrid DSVDD + tag prediction model.

    Combines:
    1. Deep SVDD loss: minimizes distance from hypersphere center
    2. Tag prediction loss: multi-label BCE for CelebA attributes
    """

    def __init__(
        self,
        objective,
        R,
        c,
        nu: float,
        tag_loss_weight: float = 1.0,
        optimizer_name: str = "adam",
        lr: float = 0.001,
        n_epochs: int = 150,
        lr_milestones: tuple = (),
        batch_size: int = 128,
        weight_decay: float = 1e-6,
        device: str = "cuda",
        n_jobs_dataloader: int = 0,
    ):
        super().__init__(
            optimizer_name,
            lr,
            n_epochs,
            lr_milestones,
            batch_size,
            weight_decay,
            device,
            n_jobs_dataloader,
        )

        assert objective in ("one-class", "soft-boundary"), (
            "Objective must be either 'one-class' or 'soft-boundary'."
        )
        self.objective = objective

        # Deep SVDD parameters
        self.R = torch.tensor(
            R, device=self.device
        )  # radius R initialized with 0 by default.
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.nu = nu

        # Tag prediction parameters
        self.tag_loss_weight = tag_loss_weight
        self.num_tags = 40  # CelebA has 40 attributes

        # Optimization parameters
        self.warm_up_n_epochs = 10  # number of training epochs for soft-boundary Deep SVDD before radius R gets updated

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None

        # Tag prediction results
        self.test_tag_accuracy = None

    def train(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get train data loader
        train_loader, _ = dataset.loaders(
            batch_size=self.batch_size, num_workers=self.n_jobs_dataloader
        )

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(
            net.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            amsgrad=self.optimizer_name == "amsgrad",
        )

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.lr_milestones, gamma=0.1
        )

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            logger.info("Initializing center c...")
            self.c = self.init_center_c(train_loader, net)
            logger.info("Center c initialized.")

        # Training
        logger.info("Starting hybrid training...")
        start_time = time.time()
        net.train()

        for epoch in range(self.n_epochs):
            scheduler.step()
            if epoch in self.lr_milestones:
                logger.info(
                    "  LR scheduler: new learning rate is %g"
                    % float(scheduler.get_lr()[0])
                )

            dsvdd_loss_epoch = 0.0
            tag_loss_epoch = 0.0
            total_loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()

            for data in train_loader:
                # Unpack 4-tuple: (image, outlier_label, all_attributes, index)
                inputs, _, tag_labels, _ = data
                inputs = inputs.to(self.device)
                tag_labels = tag_labels.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Forward pass: get both features and tag predictions
                features, tag_logits = net(inputs)

                # Compute DSVDD loss (distance from hypersphere center)
                dist = torch.sum((features - self.c) ** 2, dim=1)
                if self.objective == "soft-boundary":
                    scores = dist - self.R**2
                    dsvdd_loss = self.R**2 + (1 / self.nu) * torch.mean(
                        torch.max(torch.zeros_like(scores), scores)
                    )
                else:
                    dsvdd_loss = torch.mean(dist)

                # Compute tag prediction loss (multi-label BCE)
                tag_loss = F.binary_cross_entropy_with_logits(tag_logits, tag_labels)

                # Combined loss
                loss = dsvdd_loss + self.tag_loss_weight * tag_loss

                # Backpropagation
                loss.backward()
                optimizer.step()

                # Update hypersphere radius R on mini-batch distances
                if (self.objective == "soft-boundary") and (
                    epoch >= self.warm_up_n_epochs
                ):
                    self.R.data = torch.tensor(
                        get_radius(dist, self.nu), device=self.device
                    )

                dsvdd_loss_epoch += dsvdd_loss.item()
                tag_loss_epoch += tag_loss.item()
                total_loss_epoch += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info(
                "  Epoch {}/{}\t Time: {:.3f}\t DSVDD Loss: {:.8f}\t Tag Loss: {:.8f}\t Total Loss: {:.8f}".format(
                    epoch + 1,
                    self.n_epochs,
                    epoch_train_time,
                    dsvdd_loss_epoch / n_batches,
                    tag_loss_epoch / n_batches,
                    total_loss_epoch / n_batches,
                )
            )

        self.train_time = time.time() - start_time
        logger.info("Training time: %.3f" % self.train_time)

        logger.info("Finished training.")

        return net

    def test(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get test data loader
        _, test_loader = dataset.loaders(
            batch_size=self.batch_size, num_workers=self.n_jobs_dataloader
        )

        # Testing
        logger.info("Starting testing...")
        start_time = time.time()
        idx_label_score = []

        # Tag prediction tracking
        tag_preds_list = []
        tag_labels_list = []

        net.eval()
        with torch.no_grad():
            for data in test_loader:
                # Unpack 4-tuple: (image, outlier_label, all_attributes, index)
                inputs, labels, tag_labels, idx = data
                inputs = inputs.to(self.device)
                tag_labels = tag_labels.to(self.device)

                # Forward pass
                features, tag_logits = net(inputs)

                # DSVDD anomaly score
                dist = torch.sum((features - self.c) ** 2, dim=1)
                if self.objective == "soft-boundary":
                    scores = dist - self.R**2
                else:
                    scores = dist

                # Save triples of (idx, label, score) in a list
                idx_label_score += list(
                    zip(
                        idx.cpu().data.numpy().tolist(),
                        labels.cpu().data.numpy().tolist(),
                        scores.cpu().data.numpy().tolist(),
                    )
                )

                # Track tag predictions
                tag_preds = torch.sigmoid(tag_logits) > 0.5
                tag_preds_list.append(tag_preds.cpu())
                tag_labels_list.append(tag_labels.cpu())

        self.test_time = time.time() - start_time
        logger.info("Testing time: %.3f" % self.test_time)

        self.test_scores = idx_label_score

        # Compute AUC for anomaly detection
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)

        self.test_auc = roc_auc_score(labels, scores)
        logger.info("Test set AUC: {:.2f}%".format(100.0 * self.test_auc))

        # Compute tag prediction accuracy
        tag_preds_all = torch.cat(tag_preds_list, dim=0).numpy()
        tag_labels_all = torch.cat(tag_labels_list, dim=0).numpy()

        # Multi-label accuracy: mean of per-sample accuracies
        per_sample_accuracy = (tag_preds_all == tag_labels_all).mean(axis=1)
        self.test_tag_accuracy = per_sample_accuracy.mean()

        # Per-tag accuracy
        per_tag_accuracy = (tag_preds_all == tag_labels_all).mean(axis=0)
        logger.info(
            "Test tag prediction accuracy: {:.2f}%".format(
                100.0 * self.test_tag_accuracy
            )
        )
        logger.info(
            "Per-tag accuracy - Mean: {:.2f}%, Min: {:.2f}%, Max: {:.2f}%".format(
                100.0 * per_tag_accuracy.mean(),
                100.0 * per_tag_accuracy.min(),
                100.0 * per_tag_accuracy.max(),
            )
        )

        logger.info("Finished testing.")

    def init_center_c(self, train_loader: DataLoader, net: BaseNet, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, _, _, _ = data
                inputs = inputs.to(self.device)
                features, _ = net(inputs)
                n_samples += features.shape[0]
                c += torch.sum(features, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
