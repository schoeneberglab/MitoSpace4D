import argparse
import torch
import torch.backends.cudnn as cudnn
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from simclr.models import MitoSpace4DConvLSTM
from simclr.models_transformer import MitoSpace4DTransformer
from simclr.simclr import SimCLRRunner
from simclr.models_simple import Lightweight3DResNet
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from utils.utils import load_config
import warnings

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='MitoSpace4D')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--config', default='/u/earkfeld/MitoSpace4D/simclr/config.yaml', type=str,
                    help='Config path.')

def main():
    warnings.filterwarnings("ignore")  # supress all warnings
    args = parser.parse_args()
    cfg = load_config(args.config)

    assert cfg['training']['n_views'] == 2, "Only two view training is supported. Please use --n-views 2."

    torch.set_float32_matmul_precision('medium')

    dataset = ContrastiveLearningDataset(cfg['data_params']['data_path'], cfg)

    pick_labels = None

    train_dataset = dataset.get_dataset(cfg['data_params']['dataset_name'],
                                        cfg['training']['n_views'],
                                        flag='train', seed=None,
                                        pick_labels=pick_labels,
                                        samples_per_drug=cfg['data_params']['samples_per_drug'],
                                        timesteps=cfg['data_params']['timesteps'],
                                        zstacks=cfg['data_params']['zstacks'])
    
    val_dataset = dataset.get_dataset(cfg['data_params']['dataset_name'],
                                      cfg['training']['n_views'],
                                      flag='val', seed=None,
                                      pick_labels=pick_labels,
                                      samples_per_drug=cfg['data_params']['samples_per_drug'],
                                      timesteps=cfg['data_params']['timesteps'],
                                      zstacks=cfg['data_params']['zstacks'])

    train_loader = DataLoader(train_dataset, batch_size=cfg['training']['batch_size'], shuffle=True,
                              num_workers=cfg['training']['workers'], pin_memory=True, drop_last=True,
                              persistent_workers=cfg['training']['persistent_workers'])

    val_loader = DataLoader(val_dataset, batch_size=cfg['training']['batch_size'], shuffle=False,
                            num_workers=cfg['training']['workers'], pin_memory=True, drop_last=True,
                            persistent_workers=cfg['training']['persistent_workers'])

    # model = MitoSpace4DConvLSTM(
    #     in_channels=cfg['model_params']['in_channels'],
    #     out_dim=cfg['model_params']['out_dim'],
    #     cfg_aug=cfg['data_params']['transforms'],
    #     apply_aug=True
    # )

    # model = MitoSpace4DTransformer(cfg_aug=cfg['data_params']['transforms'], 
    #                                apply_aug=True)

    model = Lightweight3DResNet(embedding_size=2048, 
                                cfg_aug=cfg['data_params']['transforms'], 
                                apply_aug=True).cuda()

    for param in model.augment_pipeline.parameters():
        param.requires_grad = False
    for param in model.decoder.parameters():
        param.requires_grad = False

    tb_logger = pl_loggers.TensorBoardLogger(
        version=cfg["experiment_name"], save_dir=cfg["logging_params"]["save_path"]
    )

    ckpt_callback = ModelCheckpoint(
        monitor=cfg["training"]["ckpt_callback"]["monitor"],
        mode=cfg["training"]["ckpt_callback"]["mode"],
        save_top_k=cfg["training"]["ckpt_callback"]["save_top_k"],
        filename='{epoch}-{step}-{val_loss:.2f}',
        save_last=cfg["training"]["ckpt_callback"]["save_last"],
    )

    # load from checkpoint
    if cfg["training"]["continue_from_ckpt_wo_opt"] != 'None':
        # Note: It simply loads the checkpoint and doesn't load the optimizer or scheduler states
        # To continue training properly, add ckpt_path in the trainer.fit() call
        train_runner = SimCLRRunner.load_from_checkpoint(cfg["training"]["continue_from_ckpt_wo_opt"], 
                                                         cfg=cfg,
                                                         model=model)
        
        print(f"Loading checkpoints without optimizer from {cfg['training']['continue_from_ckpt_wo_opt']}")

    else:
        train_runner = SimCLRRunner(cfg, model)

    trainer = pl.Trainer(
        max_epochs=cfg["training"]["max_epochs"],
        accelerator=cfg["distributed"]["accelerator"],
        log_every_n_steps=13,
        logger=tb_logger,
        callbacks=[ckpt_callback],
        # precision=16,  # mixed precision training,
        precision="16-mixed",  # mixed precision training,
        num_nodes=cfg["distributed"]["num_nodes"],
        devices=cfg["distributed"]["num_gpus"],
        strategy=cfg["distributed"]["strategy"],
        sync_batchnorm=True,
    )

    trainer.fit(
        model=train_runner,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        #ckpt_path="/tscc/lustre/ddn/scratch/d5agarwal/projects/MitoSpace4D/runs/lightning_logs/resnetbilstm_encoded_normal_run2/checkpoints/epoch=1-step=512-val_loss=0.00.ckpt"
        # use this to load optimizer as well as model states
    )


if __name__ == "__main__":
    main()
