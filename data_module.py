import pytorch_lightning as pl
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader
from typing import Optional, Union
from pathlib import Path 


class _BaseCIFARDataModule(pl.LightningDataModule):
    """Shared logic for CIFAR-10 and CIFAR-100."""
    dataset_cls = None        # CIFAR10 or CIFAR100
    mean = std = None         # override in subclass

    def __init__(self, data_dir: str, batch_size: int = 256, num_workers: int = 4):
        super().__init__()
        self.data_dir, self.batch_size, self.num_workers = data_dir, batch_size, num_workers

        self.tf_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
        self.tf_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])

    # download once
    def prepare_data(self):
        self.dataset_cls(self.data_dir, train=True, download=True)
        self.dataset_cls(self.data_dir, train=False, download=True)

    # build the actual datasets
    def setup(self, stage: Optional[str] = None):
        self.train_set = self.dataset_cls(self.data_dir, train=True,
                                        transform=self.tf_train)
        self.val_set   = self.dataset_cls(self.data_dir, train=False,
                                        transform=self.tf_test)

    # loaders
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size,
                          shuffle=True,  num_workers=self.num_workers,
                          pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_set,   batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          pin_memory=True)


# ───────────────────────── concrete subclasses ────────────────────────────────
class CIFAR10DataModule(_BaseCIFARDataModule):
    dataset_cls = CIFAR10
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    def __init__(self, data_dir: str = "data/cifar10", **kw):
        super().__init__(data_dir=data_dir, **kw)


class CIFAR100DataModule(_BaseCIFARDataModule):
    dataset_cls = CIFAR100
    mean = (0.5071, 0.4867, 0.4408)
    std  = (0.2675, 0.2565, 0.2761)

    def __init__(self, data_dir: str = "data/cifar100", **kw):
        super().__init__(data_dir=data_dir, **kw)