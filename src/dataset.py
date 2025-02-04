import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=64, download=True):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
        transforms.Lambda(lambda x: x.view(-1))  # Flatten to (784,)
    ])

    train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=download)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=download)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader