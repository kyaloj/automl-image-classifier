import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_cifar10_loaders(batch_size=32, num_workers=2):
    """Load CIFAR-10 with standard augmentation"""
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010))
                           # These specific numbers are the actual mean and standard deviation of CIFAR-10 images. 
                           # They are standard values used across the ML community.
    ])
    
    # Test transforms (no augmentation). We test on the actual images.
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010))
    ])
    
    # Download and load datasets
    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, 
        transform=train_transform
    )
    
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True,
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=True, num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers
    )
    
    return train_loader, test_loader

# Class names for CIFAR-10
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]