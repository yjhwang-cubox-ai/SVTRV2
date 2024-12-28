# Lightning Module 구현
# 일단 모델 config 생략

import torch
import lightning as L
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR

from utils.dictionary import Dictionary
from models.preprocessor.tps_preprocessor import STN
from models.encoder.svtr_encoder import SVTREncoder
from models.decoder.svtr_decoder import SVTRDecoder
from models.module_loss.ctc_module_loss import CTCModuleLoss

class SVTR(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        # preprocessing parameters -> Lightning Module 에서 처리해야 메모리 효율적, GPU 에서 직접 처리 가능
        self.register_buffer('mean', torch.tensor([127.5]).view(1, 1, 1, 1))
        self.register_buffer('std', torch.tensor([127.5]).view(1, 1, 1, 1))

        self.dictionary = Dictionary("dicts/lao_dict.txt")
        self.criterion = CTCModuleLoss(dictionary=self.dictionary)

        # model components
        self.preprocessor = STN(in_channels=3)
        self.encoder = SVTREncoder()
        self.decoder = SVTRDecoder(in_channels=192, dictionary=self.dictionary)

        self.base_lr = 5e-4 * 2048/2048
    
    def forward(self, x):
        x = self.preprocessor(x)
        x = self.encoder(x)
        x = self.decoder(out_enc = x)
        return x

    def preprocess(self, batch):
        img = batch['img'].float()
        img = (img - self.mean) / self.std
        batch['img'] = img
        return batch

    def training_step(self, batch, batch_idx):
        batch = self.preprocess(batch)
        x = self(batch['img'])
        train_loss = self.criterion(x, batch['label'])
        self.log('train_loss', train_loss, prog_bar=True, sync_dist=True)
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        batch = self.preprocess(batch)
        x = self(batch['img'])
        val_loss = self.criterion(x, batch['label'])
        self.log('val_loss', val_loss, prog_bar=True, sync_dist=True)
        return val_loss
    
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr = self.base_lr,
            betas = (0.9, 0.99),
            eps = 8e-8,
            weight_decay = 0.05
        )

        #warm up
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.5,
            end_factor=1.0,
            total_iters=2 * self.trainer.estimated_stepping_batches // self.trainer.max_epochs
        )

        remaining_epochs = self.trainer.max_epochs - 2
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=remaining_epochs * self.trainer.estimated_stepping_batches // self.trainer.max_epochs,
            eta_min=0
        )

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[2 * self.trainer.estimated_stepping_batches // self.trainer.max_epochs] #milestones: SequentialLR 에서 스케줄러를 전환하는 시점을 지정하는 파라미터
            ),
            'interval': 'step',
            'frequency': 1
        }
        
        return [optimizer], [scheduler]