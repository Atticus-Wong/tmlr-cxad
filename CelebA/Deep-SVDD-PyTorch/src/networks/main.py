from .mnist_LeNet import MNIST_LeNet, MNIST_LeNet_Autoencoder
from .cifar10_LeNet import CIFAR10_LeNet, CIFAR10_LeNet_Autoencoder
from .cifar10_LeNet_elu import CIFAR10_LeNet_ELU, CIFAR10_LeNet_ELU_Autoencoder
from .celeba_Net import CelebA_Net, CelebA_Net_Autoencoder
from .celeba_vae_Net import CelebA_VAE_Net, CelebA_VAE_Net_Autoencoder
from .celeba_hybrid_net import CelebA_Hybrid_Net, CelebA_Hybrid_Net_Autoencoder


def build_network(net_name):
    """Builds the neural network."""

    implemented_networks = (
        "mnist_LeNet",
        "cifar10_LeNet",
        "cifar10_LeNet_ELU",
        "celeba_Net",
        "celeba_vae_Net",
        "celeba_hybrid_Net",
    )
    assert net_name in implemented_networks

    net = None

    if net_name == "mnist_LeNet":
        net = MNIST_LeNet()

    if net_name == "cifar10_LeNet":
        net = CIFAR10_LeNet()

    if net_name == "cifar10_LeNet_ELU":
        net = CIFAR10_LeNet_ELU()

    if net_name == "celeba_Net":
        net = CelebA_Net()

    if net_name == "celeba_vae_Net":
        net = CelebA_VAE_Net()

    if net_name == "celeba_hybrid_Net":
        net = CelebA_Hybrid_Net()

    return net


def build_autoencoder(net_name):
    """Builds the corresponding autoencoder network."""

    implemented_networks = (
        "mnist_LeNet",
        "cifar10_LeNet",
        "cifar10_LeNet_ELU",
        "celeba_Net",
        "celeba_vae_Net",
        "celeba_hybrid_Net",
    )
    assert net_name in implemented_networks

    ae_net = None

    if net_name == "mnist_LeNet":
        ae_net = MNIST_LeNet_Autoencoder()

    if net_name == "cifar10_LeNet":
        ae_net = CIFAR10_LeNet_Autoencoder()

    if net_name == "cifar10_LeNet_ELU":
        ae_net = CIFAR10_LeNet_ELU_Autoencoder()

    if net_name == "celeba_Net":
        ae_net = CelebA_Net_Autoencoder()

    if net_name == "celeba_vae_Net":
        ae_net = CelebA_VAE_Net_Autoencoder()

    if net_name == "celeba_hybrid_Net":
        ae_net = CelebA_Hybrid_Net_Autoencoder()

    return ae_net
