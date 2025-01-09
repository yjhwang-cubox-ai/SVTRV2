from argparse import ArgumentParser
import yaml
import os
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from data.datamodule import LaoDataModule
from models.svtr_module import SVTR
import wandb

torch.set_float32_matmul_precision('medium')

def main(args):
    # 데이터 모듈 준비
    data_module = LaoDataModule(
        train_anno_files=args['data']['train'],
        val_anno_files=args['data']['val'],
        batch_size=args['data']['batch_size'],
        num_workers=args['data']['num_workers']
    )

    # 모델 생성
    model = SVTR()

    # 콜백 정의
    callbacks = [
        ModelCheckpoint(
            monitor='word_acc',
            mode='max',
            filename='{epoch}-{step}-{word_acc:.3f}',
            verbose=True
        )
    ]

    # wandb logger 설정
    logger = WandbLogger(
        project = args['wandb']['project_name'],\
        name = args['wandb']['wandb_name'],
        )

    trainer = L.Trainer(
        max_epochs = args['training']['max_epochs'],
        accelerator = 'gpu',
        devices = 1,
        num_nodes = 1,
        strategy='ddp_find_unused_parameters_true',
        callbacks = callbacks,
        logger = logger
    )
    
    trainer.fit(model = model, datamodule = data_module)

    wandb.finish()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', default='configs/config-231.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.environ['WANDB_API_KEY'] = config['wandb']['api-key']

    main(args=config)