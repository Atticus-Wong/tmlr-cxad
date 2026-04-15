from .mnist import MNIST_Dataset
from .cifar10 import CIFAR10_Dataset
from .celeba import CelebA_Dataset


def load_dataset(dataset_name, data_path, normal_class, hybrid_mode=False):
    """Loads the dataset."""

    implemented_datasets = ("mnist", "cifar10", "celeba")
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == "mnist":
        dataset = MNIST_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name == "cifar10":
        dataset = CIFAR10_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name == "celeba":
        dataset = CelebA_Dataset(
            root=data_path, normal_class=normal_class, hybrid_mode=hybrid_mode
        )

    return dataset
