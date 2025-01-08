# Lightning Module 구현
# 일단 모델 config 생략

import torch
import torch.nn as nn
import lightning as L
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, ConstantLR

from data.dictionary import Dictionary
from models.preprocessor.tps_preprocessor import STN
from models.encoder.svtr_encoder import SVTREncoder
from models.decoder.svtr_decoder import SVTRDecoder
from models.module_loss.ctc_module_loss import CTCModuleLoss

class SVTRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dictionary = Dictionary("dicts/lao_dict.txt")
        self.preprocessor = STN(in_channels=3)
        self.encoder = SVTREncoder()
        self.decoder = SVTRDecoder(in_channels=192, dictionary=self.dictionary)
    
    def forward(self, x):
        x = self.preprocessor(x)
        x = self.encoder(x)
        x = self.decoder(out_enc=x)
        return x
    

class SVTR(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        # preprocessing parameters -> Lightning Module 에서 처리해야 메모리 효율적, GPU 에서 직접 처리 가능
        self.mean = torch.tensor([127.5]).view(1, 1, 1, 1)
        self.std = torch.tensor([127.5]).view(1, 1, 1, 1)

        self.dictionary = Dictionary("dicts/lao_dict.txt")
        self.criterion = CTCModuleLoss(dictionary=self.dictionary)

        self.model = SVTRModel()

        self.base_lr = 5e-4 * 2048/2048
    
    def forward(self, x):
        x = x.float()
        x = (x - self.mean.to(self.device)) / self.std.to(self.device)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = self(batch['img'])
        train_loss = self.criterion(x, batch['label'])

        #lr monitoring
        current_lr = self.optimizers().param_groups[0]['lr']

        batch_size = x.shape[0]

        self.log('train_loss', train_loss, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.log('learning_rate', current_lr)
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        x = self(batch['img'])
        val_loss = self.criterion(x, batch['label'])
        batch_size = x.shape[0]
        self.log('val_loss', val_loss, prog_bar=True, batch_size=batch_size, sync_dist=True)
        return val_loss
    
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr = self.base_lr,
            betas = (0.9, 0.99),
            eps = 8e-8,
            weight_decay = 0.05
        )

        # warmup
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.5,
            end_factor=1.0,
            total_iters=2
        )

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=19,
            eta_min=self.base_lr * 0.05
        )

        constant_scheduler = ConstantLR(
            optimizer,
            factor=1.0,  # 이전 스케줄러의 마지막 lr을 유지
            total_iters=0  # 무한히 지속
        )

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler, constant_scheduler],
                milestones=[2, 21]
            ),
            'interval': 'epoch',  # step에서 epoch로 변경
            'frequency': 1
        }
        
        return [optimizer], [scheduler]