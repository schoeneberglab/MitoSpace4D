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
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from utils.utils import load_config
import warnings
from tqdm import tqdm
from pytorch_lightning.profilers import AdvancedProfiler
from autoencoder.autoencoder import AutoEncoderRunner
from autoencoder.models import MitoSpace3DAutoencoder
from simclr.augmentations_exp import DataAugmentation

cudnn.benchmark = True


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
    cfg_aug = cfg['data_params']['transforms']
    device = torch.device('cuda')

    assert cfg['training']['n_views'] == 2, "Only two view training is supported. Please use --n-views 2."

    torch.set_float32_matmul_precision('medium')

    dataset = ContrastiveLearningDataset(cfg['data_params']['data_path'], cfg)
    

    train_dataset = dataset.get_dataset(cfg['data_params']['dataset_name'],
                                        cfg['training']['n_views'],
                                        flag='train', 
                                        seed=None,
                                        pick_labels=None,
                                        samples_per_drug=cfg['data_params']['samples_per_drug'],
                                        timesteps=cfg['data_params']['timesteps'],
                                        zstacks=cfg['data_params']['zstacks'])
    
    val_dataset = dataset.get_dataset(cfg['data_params']['dataset_name'],
                                      cfg['training']['n_views'],
                                      flag='val', 
                                      seed=None,
                                      pick_labels=None,
                                      samples_per_drug=cfg['data_params']['samples_per_drug'],
                                      timesteps=cfg['data_params']['timesteps'],
                                      zstacks=cfg['data_params']['zstacks'])

    train_loader = DataLoader(train_dataset, 
                              batch_size=cfg['training']['batch_size'], 
                              shuffle=True,
                              num_workers=cfg['training']['workers'], 
                              pin_memory=True, 
                              drop_last=True,
                              persistent_workers=cfg['training']['persistent_workers'])

    val_loader = DataLoader(val_dataset, 
                            batch_size=cfg['training']['batch_size'], 
                            shuffle=False,
                            num_workers=cfg['training']['workers'], 
                            pin_memory=True, 
                            drop_last=True,
                            persistent_workers=cfg['training']['persistent_workers'])
    
    #-- Decoder
    dec_checkpoint_path = "/u/earkfeld/MitoSpace4D/autoencoder/MitospaceAutoencoder.ckpt"
    decoder_model = MitoSpace3DAutoencoder()
    decoder = AutoEncoderRunner.load_from_checkpoint(dec_checkpoint_path, model=decoder_model)
    decoder = decoder.model.decoder
    decoder.eval()
    for param in decoder.parameters(): # Just to be sure
        param.requires_grad = False
    decoder.to(device)

    #-- Augmentation Pipeline
    augmentation_pipeline = DataAugmentation(cfg_aug, zero_mean_norm=True)
    augmentation_pipeline.to(device)

    pbar = tqdm(len(train_loader))
    for batch in train_loader:
        x, lbl = batch["images"], batch["classes"]
        x = x.to(device, non_blocking=True)
        with torch.no_grad():
            x = decoder(x)  # (b*n_views, t, c, z, h, w)
            x = augmentation_pipeline(x)  # (b*n_views, t, c, z, h, w)
        pbar.update(1)
    pbar.close()

    pbar = tqdm(len(val_loader))
    for batch in val_loader:
        x, lbl = batch["images"], batch["classes"]
        x = x.to(device, non_blocking=True)
        with torch.no_grad():
            x = decoder(x)  # (b*n_views, t, c, z, h, w)
            x = augmentation_pipeline(x)  # (b*n_views, t, c, z, h, w)
        pbar.update(1)

if __name__ == "__main__":
    main()
