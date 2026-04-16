import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


class CelebA_Hybrid_Net(BaseNet):
    """Hybrid network for Deep SVDD with tag prediction.

    This network outputs both:
    1. A 64-dimensional representation for DSVDD (hypersphere distance)
    2. 40-dimensional logits for multi-label tag prediction (CelebA attributes)
    """

    def __init__(self, num_tags=40):
        super().__init__()

        self.rep_dim = 64
        self.num_tags = num_tags
        self.mult = 48
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder (same architecture as CelebA_Net)
        self.conv1 = nn.Conv2d(3, self.mult, 5, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(self.mult, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(self.mult, self.mult * 2, 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(self.mult * 2, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(self.mult * 2, self.mult * 4, 5, bias=False, padding=2)
        self.bn2d3 = nn.BatchNorm2d(self.mult * 4, eps=1e-04, affine=False)
        self.conv4 = nn.Conv2d(self.mult * 4, self.mult * 8, 5, bias=False, padding=2)
        self.bn2d4 = nn.BatchNorm2d(self.mult * 8, eps=1e-04, affine=False)
        self.conv5 = nn.Conv2d(self.mult * 8, self.mult * 16, 5, bias=False, padding=2)
        self.bn2d5 = nn.BatchNorm2d(self.mult * 16, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(self.mult * 16 * 5 * 5, self.rep_dim, bias=False)

        # Tag prediction head (simple linear classifier on 64-dim representation)
        self.tag_classifier = nn.Linear(self.rep_dim, self.num_tags, bias=True)

    def forward(self, x):
        # Encoder path
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = self.conv4(x)
        x = self.pool(F.leaky_relu(self.bn2d4(x)))
        x = self.conv5(x)
        x = self.pool(F.leaky_relu(self.bn2d5(x)))
        x = x.view(x.size(0), -1)

        # DSVDD representation (64-dim)
        features = self.fc1(x)

        # Tag prediction (40-dim logits)
        tag_logits = self.tag_classifier(features)

        return features, tag_logits


class CelebA_Hybrid_Net_Autoencoder(BaseNet):
    """Autoencoder for pretraining the hybrid network.

    This autoencoder has the same encoder architecture as CelebA_Hybrid_Net
    plus a decoder for reconstruction. Used for weight initialization.
    """

    def __init__(self, num_tags=40):
        super().__init__()

        self.rep_dim = 64
        self.num_tags = num_tags
        self.mult = 48
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder (must match CelebA_Hybrid_Net)
        self.conv1 = nn.Conv2d(3, self.mult, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(
            self.conv1.weight, gain=nn.init.calculate_gain("leaky_relu")
        )
        self.bn2d1 = nn.BatchNorm2d(self.mult, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(self.mult, self.mult * 2, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(
            self.conv2.weight, gain=nn.init.calculate_gain("leaky_relu")
        )
        self.bn2d2 = nn.BatchNorm2d(self.mult * 2, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(self.mult * 2, self.mult * 4, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(
            self.conv3.weight, gain=nn.init.calculate_gain("leaky_relu")
        )
        self.bn2d3 = nn.BatchNorm2d(self.mult * 4, eps=1e-04, affine=False)
        self.conv4 = nn.Conv2d(self.mult * 4, self.mult * 8, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(
            self.conv4.weight, gain=nn.init.calculate_gain("leaky_relu")
        )
        self.bn2d4 = nn.BatchNorm2d(self.mult * 8, eps=1e-04, affine=False)
        self.conv5 = nn.Conv2d(self.mult * 8, self.mult * 16, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(
            self.conv5.weight, gain=nn.init.calculate_gain("leaky_relu")
        )
        self.bn2d5 = nn.BatchNorm2d(self.mult * 16, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(self.mult * 16 * 5 * 5, self.rep_dim, bias=False)

        # Decoder
        self.fc2 = nn.Linear(self.rep_dim, self.mult * 16 * 5 * 5, bias=False)
        self.bn2d6 = nn.BatchNorm2d(self.mult * 16, eps=1e-04, affine=False)
        self.deconv1 = nn.ConvTranspose2d(
            self.mult * 16, self.mult * 8, 5, bias=False, padding=2
        )
        nn.init.xavier_uniform_(
            self.deconv1.weight, gain=nn.init.calculate_gain("leaky_relu")
        )
        self.bn2d7 = nn.BatchNorm2d(self.mult * 8, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(
            self.mult * 8, self.mult * 4, 5, bias=False, padding=2
        )
        nn.init.xavier_uniform_(
            self.deconv2.weight, gain=nn.init.calculate_gain("leaky_relu")
        )
        self.bn2d8 = nn.BatchNorm2d(self.mult * 4, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(
            self.mult * 4, self.mult * 2, 5, bias=False, padding=2
        )
        nn.init.xavier_uniform_(
            self.deconv3.weight, gain=nn.init.calculate_gain("leaky_relu")
        )
        self.bn2d9 = nn.BatchNorm2d(self.mult * 2, eps=1e-04, affine=False)
        self.deconv4 = nn.ConvTranspose2d(
            self.mult * 2, self.mult, 5, bias=False, padding=2
        )
        nn.init.xavier_uniform_(
            self.deconv4.weight, gain=nn.init.calculate_gain("leaky_relu")
        )
        self.bn2d10 = nn.BatchNorm2d(self.mult, eps=1e-04, affine=False)
        self.deconv5 = nn.ConvTranspose2d(self.mult, 3, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(
            self.deconv5.weight, gain=nn.init.calculate_gain("leaky_relu")
        )

        # Tag classifier for pretraining (will be transferred to main network)
        self.tag_classifier = nn.Linear(self.rep_dim, self.num_tags, bias=True)

    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = self.conv4(x)
        x = self.pool(F.leaky_relu(self.bn2d4(x)))
        x = self.conv5(x)
        x = self.pool(F.leaky_relu(self.bn2d5(x)))
        x = x.view(x.size(0), -1)

        features = self.fc1(x)

        # Tag prediction (for pretraining)
        tag_logits = self.tag_classifier(features)

        # Decoder
        x = self.fc2(features)
        x = F.leaky_relu(x)
        x = x.view(x.size(0), self.mult * 16, 5, 5)
        x = F.interpolate(F.leaky_relu(self.bn2d6(x)), size=[13, 11])
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn2d7(x)), size=[27, 22])
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn2d8(x)), size=[54, 44])
        x = self.deconv3(x)
        x = F.interpolate(F.leaky_relu(self.bn2d9(x)), size=[109, 89])
        x = self.deconv4(x)
        x = F.interpolate(F.leaky_relu(self.bn2d10(x)), size=[218, 178])
        x = self.deconv5(x)
        x = torch.sigmoid(x)

        return x, tag_logits
