"""CIFAR-10 loading utilities, shared by training and the experiment script."""
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def get_train_loader(data_dir: str, batch_size: int = 128, num_workers: int = 2) -> DataLoader:
    transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    ds = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def get_test_loader(data_dir: str, batch_size: int = 256, num_workers: int = 2) -> DataLoader:
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    ds = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
