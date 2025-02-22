# Lightning Module 구현
# 일단 모델 config 생략

import unicodedata
import torch
import torch.nn as nn
import lightning as L
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, LambdaLR

from data.dictionary import Dictionary
from models.preprocessor.tps_preprocessor import STN
from models.encoder.svtr_encoder import SVTREncoder
from models.decoder.svtr_decoder import SVTRDecoder
from models.postprocessor.ctc_postprocessor import CTCPostProcessor
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
        self.postprocessor = CTCPostProcessor(self.dictionary)

        # self.base_lr = 5e-4 * 2048/2048
        self.base_lr = 5e-4 * 2

        # 카운터 초기화
        self.val_correct = 0
        self.val_total = 0
    
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

        results = self.postprocessor(x)

        def lao_text_postprocess(pred_text):
            text = unicodedata.normalize("NFKC", pred_text)
            text = text.replace('ເເ','ແ')
            text = text.replace('ໍໍ', 'ໍ')  # 연속된 ໍ를 하나로
            text = text.replace('້ໍ', 'ໍ້')  # ້ໍ를 ໍ້로 변경
            text = text.replace('ຫນ','ໜ')
            text = text.replace('ຫມ','ໝ')
            
            return text

        correct = sum(lao_text_postprocess(result['text'])==lao_text_postprocess(label) for result, label in zip(results, batch['label']))

        # 누적 카운터 업데이트
        self.val_correct += correct
        self.val_total += batch_size

        self.log('val_loss', val_loss, prog_bar=True, batch_size=batch_size, sync_dist=True)

        return val_loss

    def on_validation_epoch_start(self):
        # validation epoch 시작할 때 카운터 초기화
        self.val_correct = 0
        self.val_total = 0

    def on_validation_epoch_end(self):
        # epoch 끝나고 전체 accuracy 계산
        val_accuracy = self.val_correct / self.val_total if self.val_total > 0 else 0
        self.log('word_acc', val_accuracy, prog_bar=True, sync_dist=True)
    
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
            T_max=49,
            eta_min=self.base_lr * 0.05
        )

        # Fixed LR 스케줄러
        def fixed_lr(epoch):
            return cosine_scheduler.get_last_lr()[0] / self.base_lr

        fixed_scheduler = LambdaLR(
            optimizer,
            lr_lambda=fixed_lr
        )

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler, fixed_scheduler],
                milestones=[2, 21]
            ),
            'interval': 'epoch',  # step에서 epoch로 변경
            'frequency': 1
        }
        
        return [optimizer], [scheduler]