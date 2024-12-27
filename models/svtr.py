# Lightning Module 구현
# 일단 모델 config 생략

import torch
import torch.nn as nn
import lightning as L
from typing import Dict, Any, Optional
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

        self.dictionary = Dictionary("dicts/lao_dict.txt")
        self.criterion = CTCModuleLoss(dictionary=self.dictionary)

        self.preprocessor = STN(in_channels=3)
        self.encoder = SVTREncoder()
        self.decoder = SVTRDecoder(in_channels=192, dictionary=self.dictionary)

        self.base_lr = 5e-4 * 2048/2048
    
    def forward(self, x):
        x = self.preprocessor(x)
        x = self.encoder(x)
        x = self.decoder(out_enc = x, training=self.training)
        return x

    def training_step(self, batch, batch_idx):
        image, text = batch
        x = self(image)
        train_loss = self.criterion(x, text)
        self.log('train_loss', train_loss)
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        image, text = batch
        x = self(image)
        val_loss = self.criterion(x, text)
        self.log('val_loss', val_loss)
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
        scheduler1 = LinearLR(
            optimizer,
            start_factor=0.5,
            end_factor=1.0,
            total_iters=2 * self.trainer.estimated_stepping_batches // self.trainer.max_epochs
        )

        remaining_epochs = self.trainer.max_epochs - 2
        scheduler2 = CosineAnnealingLR(
            optimizer,
            T_max=remaining_epochs * self.trainer.estimated_stepping_batches // self.trainer.max_epochs,
            eta_min=0
        )

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[scheduler1, scheduler2],
                milestones=[2 * self.trainer.estimated_stepping_batches // self.trainer.max_epochs] #milestones: SequentialLR 에서 스케줄러를 전환하는 시점을 지정하는 파라미터
            ),
            'interval': 'step',
            'frequency': 1
        }
        
        return [optimizer], [scheduler]