from pathlib import Path
from typing import Tuple, Optional, Union
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


class BaseDataset(Dataset):
    def __init__(self,
                 data: Dataset,
                 transform: Optional[transforms.Compose] = None,
                 subset_size: Optional[int] = None
                ) -> None:
        if subset_size is not None:
            total_size = len(data)
            assert subset_size <= total_size, f"Subset size {subset_size} exceeds dataset size {total_size}."
            indices = torch.randperm(total_size)[:subset_size]
            data = Subset(data, indices)
        self.data = data
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, label = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


class MNIST(BaseDataset):
    def __init__(self,
                 root: Union[str, Path],
                 train: bool,
                 download: bool,
                 transform: Optional[transforms.Compose] = None,
                 subset_size: Optional[int] = None
                ) -> None:
        data = datasets.MNIST(root=root, train=train, download=download)
        super().__init__(data, transform, subset_size)


class CIFAR10(BaseDataset):
    def __init__(self,
                 root: Union[str, Path],
                 train: bool,
                 download: bool,
                 transform: Optional[transforms.Compose] = None,
                 subset_size: Optional[int] = None
                ) -> None:
        data = datasets.CIFAR10(root=root, train=train, download=download)
        super().__init__(data, transform, subset_size)