"""
Hybrid DSVDD + Tag Prediction for CelebA
A fully deep method for anomaly detection with multi-label tag prediction.

This script trains a hybrid model that simultaneously:
1. Performs Deep SVDD anomaly detection (learns hypersphere around normal samples)
2. Predicts all 40 CelebA attributes as a multi-label classification task

Usage:
    python hybrid_train.py

Or as a Jupyter notebook cell (define settings dict first):
    settings = {
        'dataset_name': 'celeba',
        'net_name': 'celeba_hybrid_Net',
        'xp_path': './results',
        'data_path': './data',
        'normal_class': 31,  # Smiling
        'objective': 'one-class',
        'nu': 0.1,
        'seed': -1,
        'device': 'cuda',
        'n_jobs_dataloader': 0,
        'load_config': None,
        'load_model': None,
        'pretrain': True,
        'train': True,
        'tag_loss_weight': 1.0,
        # ... other parameters
    }
    %run hybrid_train.py
"""

import os
import sys
import math
import copy
import torch
import logging
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from utils.config import Config
from utils.visualization.plot_images_grid import plot_images_grid
from deepSVDD import DeepSVDD
from datasets.main import load_dataset


def run_hybrid_training(settings):
    """Run hybrid DSVDD + tag prediction training.

    Args:
        settings: Dictionary with all configuration parameters

    Returns:
        deep_SVDD: Trained DeepSVDD model object
    """

    # Get configuration
    cfg = Config(settings.copy())

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    log_file = cfg.settings["xp_path"] + "/log.txt"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # If specified, load experiment config from JSON-file
    if cfg.settings["load_config"]:
        cfg.load_config(import_json=cfg.settings["load_config"])
        logger.info("Loaded configuration from %s." % cfg.settings["load_config"])

    # Print arguments
    logger.info("Log file is %s." % log_file)
    logger.info("Data path is %s." % cfg.settings["data_path"])
    logger.info("Export path is %s." % cfg.settings["xp_path"])

    logger.info("Dataset: %s" % cfg.settings["dataset_name"])
    logger.info("Normal class: %d" % cfg.settings["normal_class"])
    logger.info("Network: %s" % cfg.settings["net_name"])

    # Print configuration
    logger.info("Deep SVDD objective: %s" % cfg.settings["objective"])
    logger.info("Nu-parameter: %.2f" % cfg.settings["nu"])

    # Hybrid mode specific logging
    logger.info("Hybrid mode: ENABLED")
    logger.info("Tag loss weight: %.2f" % cfg.settings.get("tag_loss_weight", 1.0))

    # Set seed
    if cfg.settings["seed"] != -1:
        random.seed(cfg.settings["seed"])
        np.random.seed(cfg.settings["seed"])
        torch.manual_seed(cfg.settings["seed"])
        logger.info("Set seed to %d." % cfg.settings["seed"])

    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        cfg.settings["device"] = "cpu"
    logger.info("Computation device: %s" % cfg.settings["device"])
    logger.info("Number of dataloader workers: %d" % cfg.settings["n_jobs_dataloader"])

    # Load data with hybrid mode enabled
    dataset = load_dataset(
        cfg.settings["dataset_name"],
        cfg.settings["data_path"],
        cfg.settings["normal_class"],
        hybrid_mode=True,
    )
    dataset_full = load_dataset(
        cfg.settings["dataset_name"], cfg.settings["data_path"], -1, hybrid_mode=True
    )

    # Initialize DeepSVDD model in hybrid mode and set neural network \phi
    deep_SVDD = DeepSVDD(
        cfg.settings["objective"], cfg.settings["nu"], hybrid_mode=True
    )
    deep_SVDD.set_network(cfg.settings["net_name"])

    # If specified, load Deep SVDD model
    if cfg.settings["load_model"]:
        deep_SVDD.load_model(model_path=cfg.settings["load_model"], load_ae=True)
        logger.info("Loading model from %s." % cfg.settings["load_model"])

    logger.info("Pretraining: %s" % cfg.settings["pretrain"])
    if cfg.settings["pretrain"] and cfg.settings["train"]:
        # Log pretraining details
        logger.info("Pretraining optimizer: %s" % cfg.settings["ae_optimizer_name"])
        logger.info("Pretraining learning rate: %g" % cfg.settings["ae_lr"])
        logger.info("Pretraining epochs: %d" % cfg.settings["ae_n_epochs"])
        logger.info(
            "Pretraining learning rate scheduler milestones: %s"
            % (cfg.settings["ae_lr_milestone"],)
        )
        logger.info("Pretraining batch size: %d" % cfg.settings["ae_batch_size"])
        logger.info("Pretraining weight decay: %g" % cfg.settings["ae_weight_decay"])

        # Pretrain model on dataset (via autoencoder)
        # Note: Pretraining uses the hybrid autoencoder which also has tag prediction head
        deep_SVDD.pretrain(
            dataset_full,
            optimizer_name=cfg.settings["ae_optimizer_name"],
            lr=cfg.settings["ae_lr"],
            n_epochs=cfg.settings["ae_n_epochs"],
            lr_milestones=cfg.settings["ae_lr_milestone"],
            batch_size=cfg.settings["ae_batch_size"],
            weight_decay=cfg.settings["ae_weight_decay"],
            device=cfg.settings["device"],
            n_jobs_dataloader=cfg.settings["n_jobs_dataloader"],
        )

    # Train model on dataset
    if cfg.settings["train"]:
        # Log training details
        logger.info("Training optimizer: %s" % cfg.settings["optimizer_name"])
        logger.info("Training learning rate: %g" % cfg.settings["lr"])
        logger.info("Training epochs: %d" % cfg.settings["n_epochs"])
        logger.info(
            "Training learning rate scheduler milestones: %s"
            % (cfg.settings["lr_milestone"],)
        )
        logger.info("Training batch size: %d" % cfg.settings["batch_size"])
        logger.info("Training weight decay: %g" % cfg.settings["weight_decay"])

        # Train with hybrid mode - includes tag prediction loss
        deep_SVDD.train(
            dataset,
            optimizer_name=cfg.settings["optimizer_name"],
            lr=cfg.settings["lr"],
            n_epochs=cfg.settings["n_epochs"],
            lr_milestones=cfg.settings["lr_milestone"],
            batch_size=cfg.settings["batch_size"],
            weight_decay=cfg.settings["weight_decay"],
            device=cfg.settings["device"],
            n_jobs_dataloader=cfg.settings["n_jobs_dataloader"],
            tag_loss_weight=cfg.settings.get("tag_loss_weight", 1.0),
        )

    # Test model
    # Hybrid test returns both anomaly scores and tag predictions
    deep_SVDD.test(
        dataset,
        device=cfg.settings["device"],
        n_jobs_dataloader=cfg.settings["n_jobs_dataloader"],
    )

    # Log hybrid-specific results
    if (
        "test_tag_accuracy" in deep_SVDD.results
        and deep_SVDD.results["test_tag_accuracy"] is not None
    ):
        logger.info(
            "Test tag prediction accuracy: %.2f%%"
            % (100.0 * deep_SVDD.results["test_tag_accuracy"])
        )

    # Plot most anomalous and most normal (within-class) test samples
    indices, labels, scores = zip(*deep_SVDD.results["test_scores"])
    indices, labels, scores = np.array(indices), np.array(labels), np.array(scores)
    idx_sorted = indices[labels == 0][
        np.argsort(scores[labels == 0])
    ]  # sorted from lowest to highest anomaly score

    # Get attributes list for CelebA
    attributes = [
        "5_o_Clock_Shadow",
        "Arched_Eyebrows",
        "Attractive",
        "Bags_Under_Eyes",
        "Bald",
        "Bangs",
        "Big_Lips",
        "Big_Nose",
        "Black_Hair",
        "Blond_Hair",
        "Blurry",
        "Brown_Hair",
        "Bushy_Eyebrows",
        "Chubby",
        "Double_Chin",
        "Eyeglasses",
        "Goatee",
        "Gray_Hair",
        "Heavy_Makeup",
        "High_Cheekbones",
        "Male",
        "Mouth_Slightly_Open",
        "Mustache",
        "Narrow_Eyes",
        "No_Beard",
        "Oval_Face",
        "Pale_Skin",
        "Pointy_Nose",
        "Receding_Hairline",
        "Rosy_Cheeks",
        "Sideburns",
        "Smiling",
        "Straight_Hair",
        "Wavy_Hair",
        "Wearing_Earrings",
        "Wearing_Hat",
        "Wearing_Lipstick",
        "Wearing_Necklace",
        "Wearing_Necktie",
        "Young",
    ]

    # Return all attributes per image when test set is indexed
    dataset.test_set.apply_target_transform = False

    if cfg.settings["dataset_name"] in ("mnist", "cifar10", "celeba"):
        if cfg.settings["dataset_name"] == "mnist":
            X_normals = dataset.test_set.data[idx_sorted[:32], ...].unsqueeze(1)
            X_outliers = dataset.test_set.data[idx_sorted[-32:], ...].unsqueeze(1)

        if cfg.settings["dataset_name"] == "cifar10":
            X_normals = torch.tensor(
                np.transpose(dataset.test_set.data[idx_sorted[:32], ...], (0, 3, 1, 2))
            )
            X_outliers = torch.tensor(
                np.transpose(dataset.test_set.data[idx_sorted[-32:], ...], (0, 3, 1, 2))
            )

        if cfg.settings["dataset_name"] == "celeba":
            # Load images for visualization
            X_normals = torch.from_numpy(
                np.transpose(
                    np.array(
                        [
                            np.array(
                                Image.open(
                                    os.path.join(
                                        dataset.test_set.root,
                                        dataset.test_set.base_folder,
                                        "img_align_celeba",
                                        dataset.test_set.filename[idx_sorted[i]],
                                    )
                                )
                            )
                            for i in range(100)
                        ]
                    ),
                    (0, 3, 1, 2),
                )
            )
            X_outliers = torch.from_numpy(
                np.transpose(
                    np.array(
                        [
                            np.array(
                                Image.open(
                                    os.path.join(
                                        dataset.test_set.root,
                                        dataset.test_set.base_folder,
                                        "img_align_celeba",
                                        dataset.test_set.filename[idx_sorted[-i]],
                                    )
                                )
                            )
                            for i in range(1, 101)
                        ]
                    ),
                    (0, 3, 1, 2),
                )
            )

            # Find frequency of tags in top 5% anomalous images
            label_frequency_outliers = [
                [attributes[i], 0] for i in range(40)
            ]  # Only 40 attributes for hybrid mode

            ilp_tags_outliers = []

            n_top_5p = math.floor(0.05 * len(idx_sorted))
            for i in idx_sorted[-1 : -(n_top_5p + 1) : -1].tolist():
                # Note: In hybrid mode, __getitem__ returns (image, outlier_label, all_attrs, index)
                _, labels_attrs, _ = dataset.test_set[i][:3]  # Get first 3 elements
                if isinstance(labels_attrs, torch.Tensor):
                    attrs = labels_attrs.numpy()
                else:
                    attrs = labels_attrs
                ilp_tags_outliers.append(attrs)
                for j in range(40):
                    if attrs[j] == 1:
                        label_frequency_outliers[j][1] += 1

            frequency_sorted = copy.deepcopy(label_frequency_outliers)
            frequency_sorted.sort(key=lambda x: x[1])
            labels = [frequency_sorted[i][0] for i in range(40)]
            frequencies = [frequency_sorted[i][1] for i in range(40)]

            fig, ax = plt.subplots()
            ax.barh(labels, frequencies)
            ax.invert_yaxis()
            ax.set_xlabel("Frequency")
            ax.set_title("Frequency of attributes in top 5% most anomalous examples")
            fig.set_size_inches(13, 10)
            plt.savefig(cfg.settings["xp_path"] + "/frequency_outliers", dpi=300)
            plt.clf()

            # Find frequency of tags in top 95% normal images
            label_frequency_normals = [[attributes[i], 0] for i in range(40)]

            ilp_tags_normals = []

            n_bottom_95p = math.floor(0.95 * len(idx_sorted))
            for i in idx_sorted[:n_bottom_95p].tolist():
                _, labels_attrs, _ = dataset.test_set[i][:3]
                if isinstance(labels_attrs, torch.Tensor):
                    attrs = labels_attrs.numpy()
                else:
                    attrs = labels_attrs
                ilp_tags_normals.append(attrs)
                for j in range(40):
                    if attrs[j] == 1:
                        label_frequency_normals[j][1] += 1

            label_frequency = copy.deepcopy(label_frequency_normals)
            label_frequency.sort(key=lambda x: x[1])
            labels = [label_frequency[i][0] for i in range(40)]
            frequencies = [label_frequency[i][1] for i in range(40)]

            fig, ax = plt.subplots()
            ax.barh(labels, frequencies)
            ax.invert_yaxis()
            ax.set_xlabel("Frequency")
            ax.set_title("Frequency of attributes in top 95% most normal examples")
            fig.set_size_inches(13, 10)
            plt.savefig(cfg.settings["xp_path"] + "/frequency_normals", dpi=300)
            plt.clf()

            # Additional: Compare predicted vs actual tags for top anomalous images
            logger.info(
                "Generating tag prediction comparison for top anomalous images..."
            )

            # Get model predictions for top outliers
            deep_SVDD.net.eval()
            with torch.no_grad():
                n_samples = min(10, len(idx_sorted))
                for i in range(n_samples):
                    idx = idx_sorted[-(i + 1)]
                    image, actual_attrs, _ = dataset.test_set[idx][:3]
                    image = image.unsqueeze(0).to(cfg.settings["device"])

                    features, tag_logits = deep_SVDD.net(image)
                    predicted_attrs = (torch.sigmoid(tag_logits) > 0.5).cpu().numpy()[0]

                    if isinstance(actual_attrs, torch.Tensor):
                        actual_attrs = actual_attrs.numpy()

                    # Log comparison
                    logger.info(f"Top outlier {i + 1} (idx={idx}):")
                    logger.info(
                        f"  Actual attributes: {[attributes[j] for j in range(40) if actual_attrs[j] == 1]}"
                    )
                    logger.info(
                        f"  Predicted attributes: {[attributes[j] for j in range(40) if predicted_attrs[j] == 1]}"
                    )

                    # Calculate accuracy for this sample
                    sample_acc = (predicted_attrs == actual_attrs).mean()
                    logger.info(f"  Sample tag accuracy: {sample_acc:.2%}")

        plot_images_grid(
            X_normals,
            export_img=cfg.settings["xp_path"] + "/normals",
            title="Most normal examples",
            padding=2,
        )
        plot_images_grid(
            X_outliers,
            export_img=cfg.settings["xp_path"] + "/outliers",
            title="Most anomalous examples",
            padding=2,
        )

    # Save results, model, and configuration
    if cfg.settings["train"]:
        deep_SVDD.save_results(export_json=cfg.settings["xp_path"] + "/results.json")
        deep_SVDD.save_model(export_model=cfg.settings["xp_path"] + "/model.tar")
        cfg.save_config(export_json=cfg.settings["xp_path"] + "/config.json")

        logger.info("Hybrid model training complete!")
        logger.info("Results saved to: %s" % cfg.settings["xp_path"])

    return deep_SVDD


# For Jupyter notebook usage - check if settings is defined
try:
    settings
except NameError:
    # settings not defined, running as standalone script
    if __name__ == "__main__":
        # Default settings for standalone execution
        default_settings = {
            "dataset_name": "celeba",
            "net_name": "celeba_hybrid_Net",
            "xp_path": "./results",
            "data_path": "./data",
            "normal_class": 31,  # Smiling
            "objective": "one-class",
            "nu": 0.1,
            "seed": -1,
            "device": "cuda",
            "n_jobs_dataloader": 0,
            "load_config": None,
            "load_model": None,
            "pretrain": True,
            "train": True,
            "tag_loss_weight": 1.0,
            "optimizer_name": "adam",
            "lr": 0.001,
            "n_epochs": 50,
            "lr_milestone": (),
            "batch_size": 128,
            "weight_decay": 1e-6,
            "ae_optimizer_name": "adam",
            "ae_lr": 0.001,
            "ae_n_epochs": 100,
            "ae_lr_milestone": (),
            "ae_batch_size": 128,
            "ae_weight_decay": 1e-6,
        }

        print("Running with default settings...")
        print("To customize, define a 'settings' dict before importing this module")
        run_hybrid_training(default_settings)
else:
    # settings is defined (running in Jupyter notebook)
    run_hybrid_training(settings)
