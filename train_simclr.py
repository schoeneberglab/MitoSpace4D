import argparse
import torch
import torch.backends.cudnn as cudnn
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from simclr.models import ResNetSimCLR3D
from simclr.simclr import SimCLRRunner
from torchsummary import summary
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from utils.utils import load_config
import warnings
from data_aug.mitospace_dataset import MitoSpaceDataModule

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='MitoSpace4D')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--config', default='/home/dhruvagarwal/projects/MitoSpace4D/simclr/config.yaml', type=str,
                    help='Config path.')


def main():
    warnings.filterwarnings("ignore")  # supress all warnings
    args = parser.parse_args()
    cfg = load_config(args.config)

    assert cfg['training']['n_views'] == 2, "Only two view training is supported. Please use --n-views 2."

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    dataset = ContrastiveLearningDataset(cfg['data_params']['data_path'], cfg)

    train_dataset = dataset.get_dataset(cfg['data_params']['dataset_name'],
                                        cfg['training']['n_views'],
                                        flag='train', seed=None,
                                        pick_labels=None,
                                        samples_per_drug=cfg['data_params']['samples_per_drug'],
                                        timesteps=cfg['data_params']['timesteps'],
                                        zstacks=cfg['data_params']['zstacks'])
    val_dataset = dataset.get_dataset(cfg['data_params']['dataset_name'],
                                      cfg['training']['n_views'],
                                      flag='val', seed=None,
                                      pick_labels=None,
                                      samples_per_drug=cfg['data_params']['samples_per_drug'],
                                      timesteps=cfg['data_params']['timesteps'],
                                      zstacks=cfg['data_params']['zstacks'])

    datamodule = MitoSpaceDataModule(train_datasets=[train_dataset],
                                     val_datasets=[val_dataset],
                                     batch_size=cfg['training']['batch_size'],
                                     num_workers=cfg['training']['workers'], pin_memory=True, drop_last=True)

    model = ResNetSimCLR3D(base_model=cfg['model_params']['arch'],
                           out_dim=cfg['model_params']['out_dim'],
                           pretrained=cfg['model_params']['pretrained'],
                           in_channels=cfg['model_params']['num_z'])

    print(summary(model.to('cuda'), (cfg["model_params"]["timesteps"],
                                     cfg["model_params"]["num_z"],
                                     cfg['data_params']['patch_size'], cfg['data_params']['patch_size'])))

    tb_logger = pl_loggers.TensorBoardLogger(
        version=cfg["experiment_name"], save_dir=cfg["logging_params"]["save_path"]
    )

    ckpt_callback = ModelCheckpoint(
        monitor=cfg["training"]["ckpt_callback"]["monitor"],
        mode=cfg["training"]["ckpt_callback"]["mode"],
        save_top_k=cfg["training"]["ckpt_callback"]["save_top_k"],
        filename='best_model',
        save_last=cfg["training"]["ckpt_callback"]["save_last"],
    )

    # load from checkpoint
    if cfg["training"]["continue_from_ckpt_wo_opt"] != 'None':
        # Note: It simply loads the checkpoint and doesn't load the optimizer or scheduler states
        # To continue training properly, add ckpt_path in the trainer.fit() call
        train_runner = SimCLRRunner.load_from_checkpoint(cfg["training"]["continue_from_ckpt_wo_opt"], cfg=cfg,
                                                         model=model)

    else:
        train_runner = SimCLRRunner(cfg, model)

    trainer = pl.Trainer(
        max_epochs=cfg["training"]["max_epochs"],
        accelerator=cfg["gpu"]["accelerator"],
        log_every_n_steps=13,
        logger=tb_logger,
        callbacks=[ckpt_callback],
        precision=16,  # mixed precision training
        devices=cfg["gpu"]["num_gpus"],
        strategy=cfg["gpu"]["strategy"],
        sync_batchnorm=True
    )
    trainer.fit(
        model=train_runner,
        datamodule=datamodule,
        # ckpt_path="/home/dhruvagarwal/projects/MitoSpace/runs/lightning_logs/20240503_noNorm_moreStrongAug_MixedPrec_cosineLR_1000epochs_normalizedFeats512dims/checkpoints/last-v1.ckpt"
        # use this to load optimizer as well as model states
    )


if __name__ == "__main__":
    main()
