from argparse import ArgumentParser
import yaml
import os
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from data.datamodule import LaoDataModule

from models.svtr_module import SVTR
import wandb

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
            monitor='val_loss',       # 모니터링할 지표
            mode='min',               # 'min'은 최소값이 최선임을 의미
            save_top_k=3,             # 최상의 k개 모델만 저장
            filename='best-checkpoint',  # 저장될 파일명 형식
            verbose=True              # 저장 시 로그 출력 여부
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
        devices = 8,
        num_nodes = 2,
        strategy='ddp_find_unused_parameters_true',
        callbacks = callbacks,
        logger = logger
    )
    
    trainer.fit(model = model, datamodule = data_module)

    wandb.finish()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.environ['WANDB_API_KEY'] = config['wandb']['api-key']

    main(args=config)