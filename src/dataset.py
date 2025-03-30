import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=64, download=True):

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=download)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader