import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet

# Network copied from:
# https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
# https://arxiv.org/pdf/1902.04601.pdf
class CelebA_VAE_Net(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 64

        modules = []
        hidden_dims = [48, 80, 140, 300, 768]

        in_channels = 3
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*5*5, self.rep_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*5*5, self.rep_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)

        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu


class CelebA_VAE_Net_Autoencoder(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 64

        modules = []
        hidden_dims = [48, 80, 140, 300, 768]

        in_channels = 3
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*5*5, self.rep_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*5*5, self.rep_dim)

        modules = []
        self.decoder_input = nn.Linear(self.rep_dim, hidden_dims[-1]*5*5)
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i+1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.LeakyReLU()
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)

        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        x = eps * std + mu

        x = self.decoder_input(x)
        x = x.view(-1, 768, 5, 5)
        x = self.decoder(x)
        x = self.final_layer(x)

        return x, mu, log_var
