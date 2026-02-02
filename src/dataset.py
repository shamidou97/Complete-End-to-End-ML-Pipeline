import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def get_dataloaders(data_path='/app/data', batch_size=16, val_split=0.2):
    """
    Downloads CIFAR-10 and returns Train (Augmented) and Validation (Clean) DataLoaders.
    
    Args:
        val_split (float): The decimal percentage of data to use for validation. 
                           Default 0.2 means 20% validation, 80% training.
    """
    # 1. Define Transforms
    # Augmentation for Training: Helps the model generalize
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),       # Randomly crop with padding
        transforms.RandomHorizontalFlip(),          # Randomly flip left-right
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Clean for Validation: No random changes, just normalization
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 2. Download Data Source (we load it just to get the length first)
    base_dataset = torchvision.datasets.CIFAR10(
        root=data_path, train=True, download=True, transform=None
    )

    # 3. Create Split Indices
    dataset_size = len(base_dataset) # 50,000 for CIFAR-10 training set
    indices = list(range(dataset_size))
    split_index = int(np.floor(val_split * dataset_size)) # e.g., 0.2 * 50000 = 10000
    
    # Log the split sizes for verification
    print(f"Total Dataset Size: {dataset_size}")
    print(f"Training Split (80%): {dataset_size - split_index} images")
    print(f"Validation Split (20%): {split_index} images")
    
    # Shuffle indices with a fixed seed for reproducibility
    np.random.seed(42)
    np.random.shuffle(indices)
    
    # Slice the indices: first 'split_index' are val, rest are train
    val_indices, train_indices = indices[:split_index], indices[split_index:]

    # 4. Create Two Dataset Objects with Different Transforms
    # We map the SAME indices to two different dataset objects
    # to ensure train_subset gets augmentation and val_subset does not.
    train_set = torchvision.datasets.CIFAR10(
        root=data_path, train=True, download=False, transform=train_transform
    )
    val_set = torchvision.datasets.CIFAR10(
        root=data_path, train=True, download=False, transform=val_transform
    )

    # Create Subsets based on the indices
    train_subset = Subset(train_set, train_indices)
    val_subset = Subset(val_set, val_indices)

    # 5. Create DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader

def get_classes():
    return ('plane', 'car', 'bird', 'cat', 'deer', 
            'dog', 'frog', 'horse', 'ship', 'truck')
