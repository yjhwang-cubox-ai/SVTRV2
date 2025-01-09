import lightning as L
from torch.utils.data import DataLoader
from data.lao_dataset import LaoDataset

class LaoDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_anno_files: list,
        val_anno_files: list = None,
        test_anno_files: list = None,
        batch_size: int = 32,
        num_workers: int = 4
    ):
        super().__init__()
        self.train_anno_files = train_anno_files
        self.val_anno_files = val_anno_files
        self.test_anno_files = test_anno_files
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        """데이터셋 준비"""
        if stage == 'fit' or stage is None:
            self.train_dataset = LaoDataset(self.train_anno_files, is_train=True)
            if self.val_anno_files:
                self.val_dataset = LaoDataset(self.val_anno_files, is_train=False)
        if stage == 'test' or stage is None:
            if self.test_anno_file:
                self.test_dataset = LaoDataset(self.test_anno_files, is_train=False)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )

    def val_dataloader(self):
        if hasattr(self, 'val_anno_files'):
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False
            )

    def test_dataloader(self):
        if hasattr(self, 'test_anno_files'):
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=False
            )