from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class TranslatorDataLoader(LightningDataModule):
    """
    LightningDataModule for creating data loaders for training and validation in a translation task.

    Parameters:
        train_dataset (Dataset): Training dataset.
        dev_dataset (Dataset): Validation dataset.
        batch_size (int): Batch size for the data loaders.
        num_workers (int): Number of workers for the data loaders.

    Attributes:
        train_dataset (Dataset): Training dataset.
        dev_dataset (Dataset): Validation dataset.
        batch_size (int): Batch size for the data loaders.
    """

    def __init__(self, train_dataset, dev_dataset, batch_size, num_workers=6):
        super().__init__()
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            pin_memory = True,
            shuffle = True,
        )
    

    def val_dataloader(self):
        return DataLoader(
            self.dev_dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            pin_memory = True,
            shuffle = False,
        )