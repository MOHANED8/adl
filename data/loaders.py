import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from config import DatasetName, NoiseType
from data.noise import inject_label_noise
from data.subset import get_stratified_subset

def get_loaders(dataset_name: DatasetName, fraction: float, noise_type: NoiseType, noise_level: float, batch_size: int):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if dataset_name == DatasetName.CIFAR10:
        train_set = torchvision.datasets.CIFAR10(root='./data_root', train=True, download=True, transform=transform)
        val_set = torchvision.datasets.CIFAR10(root='./data_root', train=False, download=True, transform=transform)
        num_classes = 10
    elif dataset_name == DatasetName.CIFAR100:
        train_set = torchvision.datasets.CIFAR100(root='./data_root', train=True, download=True, transform=transform)
        val_set = torchvision.datasets.CIFAR100(root='./data_root', train=False, download=True, transform=transform)
        num_classes = 100
    else:
        raise ValueError(f"Dataset {dataset_name} not supported yet in this loader.")

    # Apply label noise
    train_set = inject_label_noise(train_set, noise_type, noise_level, num_classes)

    # Apply stratification/subsampling
    train_set = get_stratified_subset(train_set, fraction, num_classes)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, num_classes
