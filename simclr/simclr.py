import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn.functional as F
import umap
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from sklearn.metrics import davies_bouldin_score
import pytorch_lightning as pl

from simclr.loss import SupConLoss, InfoNCELoss
from typing import Dict, Any, Tuple, List
from simclr.models import Small3DResNetLSTM

torch.manual_seed(0)


def load_resnet_model(cfg, ckpt_path, device='cuda', eval_mode=True):
    model = Small3DResNetLSTM(out_dim=cfg['model_params']['out_dim'],
                              in_channels=cfg["model_params"]["in_channels"]).to(device)

    model = SimCLRRunner.load_from_checkpoint(ckpt_path, model=model, cfg=cfg)
    # state_dict = torch.load(ckpt_path, map_location=device)
    # for key in list(state_dict.keys()):
    #     if 'model.' in key:
    #         state_dict[key.replace('model.', '')] = state_dict.pop(key)
    # model.load_state_dict(state_dict)

    if eval_mode:
        model.eval()

    return model


class SimCLRRunner(pl.LightningModule):
    def __init__(self, cfg: Dict, model: torch.nn.Module) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.loss = cfg['training']['loss']['name']

        self.intermediate_outputs = []

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          cfg['training']['lr'],
                                          weight_decay=cfg['training']['weight_decay'])

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                    T_max=cfg['training']['max_epochs'],
                                                                    eta_min=0, last_epoch=-1)

        self.data_bank = {"Train": [], "Val": []}

        print(f"###################### Using {self.loss} Loss For Training ##################")

        if self.loss == 'InfoNCELoss':
            self.criterion = InfoNCELoss(use_normalization=cfg['training']['loss']['use_normalization'],
                                         temperature=cfg['training']['loss']['temperature'])
        elif self.loss == 'SupConLoss':
            self.criterion = SupConLoss(use_normalization=cfg['training']['loss']['use_normalization'],
                                        temperature=cfg['training']['loss']['temperature'],
                                        base_temperature=cfg['training']['loss']['temperature'])

        self.train_draw_period = cfg['training']['print_interval']
        self.projector_period = cfg['training']['projector_interval']
        self.val_draw = False

    def flush_bank(self):
        self.data_bank = {"Train": [], "Val": []}

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = self.optimizer
        scheduler = self.scheduler

        return [optimizer], [{'scheduler': scheduler, 'interval': 'epoch'}]

    def plot_img(self, img_name: str, img: torch.Tensor) -> None:
        """
        Plot the image and log it to tensorboard

        The function normalizes the image tensor b/w [0-1] before plotting
        """
        img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))
        img = img.cpu()

        if len(img.shape) == 2:
            cm = plt.get_cmap('viridis')
            colored_image = cm(img)[:, :, :3]
        else:
            colored_image = img

        self.logger.experiment.add_image(img_name, colored_image, self.global_step, dataformats='HWC')

    def additional_log(self, batch: Dict[str, Any], key: str) -> None:
        """
        The function is meant to call all the additional logging functions

        Assumptions for the batch:
            1. The batch should contain the images and the classes
            2. The images are a list of size 2, where the first element is the augmentation-1 image and the second is the augmentation-2 image
            3. The classes are the labels for the images
            4. batch contains more than 1 sample; otherwise the random index selection will get stuck in an infinite loop
        """

        if isinstance(batch, Dict):
            assert len(batch["images"]) == 2, "The batch should contain 2 views"
            assert batch["images"][0].shape[0] > 1, "The batch should contain more than 1 sample"
        else:
            assert len(batch[0]) == 2, "The batch should contain 2 views"
            assert batch[0][0].shape[0] > 1, "The batch should contain more than 1 sample"

        if isinstance(batch, Dict):
            images, classes = batch["images"], batch["classes"]
        else:
            images, classes = batch

        # for now we only have either MitoTracker or TMRM channel
        # pick random time and random z
        num_timesteps = images[0].shape[0]
        num_z = images[0].shape[1]

        random_timestep = torch.randint(high=num_timesteps, size=(1,))[0]
        random_z = torch.randint(high=num_z, size=(1,))[0]

        #  random positive pair
        idx = torch.randint(high=images[0].shape[0], size=(1,))[0]
        pos1_mito = images[0][idx, random_timestep, random_z]  # aug-1
        pos2_mito = images[1][idx][random_timestep, random_z]  # aug-2

        #  random negative pair
        neg1_mito = images[0][idx][random_timestep, random_z]  # normal
        neg_idx = torch.randint(high=images[0].shape[0], size=(1,))[0]
        while neg_idx == idx:
            neg_idx = torch.randint(high=images[0].shape[0], size=(1,))[0]
        neg2_mito = images[1][neg_idx][random_timestep, random_z]  # aug

        # concat positive and negative pairs
        sep = 10
        pos_mito_pair = torch.zeros((images[0].shape[-2], images[0].shape[-1] * 2 + sep))
        pos_mito_pair[:, :images[0].shape[3]] = pos1_mito
        pos_mito_pair[:, images[0].shape[3] + 10:images[0].shape[3] * 2 + sep] = pos2_mito

        neg_mito_pair = torch.zeros((images[0].shape[-2], images[0].shape[-1] * 2 + sep))
        neg_mito_pair[:, :images[0].shape[3]] = neg1_mito
        neg_mito_pair[:, images[0].shape[3] + 10:images[0].shape[3] * 2 + sep] = neg2_mito

        # plot images
        self.plot_img(f"{key}/Positive MitoTracker", pos_mito_pair)
        self.plot_img(f"{key}/Negative MitoTracker", neg_mito_pair)

    def log_mitospace(self, batch):
        if isinstance(batch, Dict):
            assert len(batch["images"]) == 2, "The batch should contain 2 views"
            assert batch["images"][0].shape[0] > 1, "The batch should contain more than 1 sample"
        else:
            assert len(batch[0]) == 2, "The batch should contain 2 views"
            assert batch[0][0].shape[0] > 1, "The batch should contain more than 1 sample"

        if isinstance(batch, Dict):
            images, classes = batch["images"], batch["classes"]
        else:
            images, classes = batch

        # make the embeddings
        embds = batch["embeddings"]
        embds = embds.reshape(embds.shape[0], -1)
        embds = embds[:embds.shape[0] // 2]  # taking only one view
        labels = classes.cpu().numpy()
        labels = list(labels)

        num_timesteps = images[0].shape[0]
        num_z = images[0].shape[1]
        random_timestep = torch.randint(high=num_timesteps, size=(1,))[0]
        random_z = torch.randint(high=num_z, size=(1,))[0]

        images = images[0].cpu()  # taking only one view
        images = images[:, random_timestep, random_z, :, :].unsqueeze(1)  # taking a random timestep and random z
        images = (images + 1) / 2.

        self.logger.experiment.add_embedding(mat=embds.detach().cpu(),
                                             metadata=labels,
                                             label_img=images,
                                             tag='embeddings',
                                             global_step=self.global_step)

    def cross_entropy_2d(self, features, labels, batch_size, n_views):
        dvc = features.device
        features = F.normalize(features, dim=-1)
        temperature = 1

        features = torch.stack(torch.split(features, [batch_size for _ in range(n_views)]), dim=1)

        features = features.view(features.shape[0], features.shape[1], -1)

        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(dvc)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0).float()

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        logits = torch.div(torch.matmul(anchor_feature, contrast_feature.T), temperature)

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        entropy = -torch.sum(mask * F.log_softmax(logits, dim=1), dim=1)

        return entropy.mean()

    def augment(self, images: torch.Tensor, n_views: int) -> torch.Tensor:
        """
        Apply the same augmentation to generate multiple views (# n_views) of the same image
        """
        views = []
        for i in range(n_views):
            views.append(self.model.augment_data(images))

        views = torch.stack(views, dim=1)
        return views

    def batch_step(self, batch: Dict[str, Any], key: str = "Train") -> tuple[
        list[Any | None], Any | None, float, torch.Tensor]:
        """Common batch step for train, val, test"""

        images, classes = batch["images"], batch["classes"]
        images = self.augment(images, n_views=self.cfg['training']['n_views']) # (b, n_views, c, t, z, h, w)
        batch["images"] = images

        images = images.reshape(-1, *images.shape[2:]) # (b*n_views, c, t, z, h, w)
        images = images.transpose(1, 2)  # b, c, t, z, h, w -> b, t, c, z, h, w

        features, out = self.model(images)

        loss, cross_entropy, acc = None, None, None
        if self.loss == 'InfoNCELoss':
            loss, acc = self.criterion(out, bs=self.cfg['training']['batch_size'],
                                       n_views=self.cfg['training']['n_views'])
            cross_entropy = self.cross_entropy_2d(features.detach().cpu(), labels=classes.detach().cpu(),
                                                  batch_size=self.cfg['training']['batch_size'],
                                                  n_views=self.cfg['training']['n_views'])
        elif self.loss == 'SupConLoss':
            loss, acc = self.criterion(out, labels=classes,
                                       bs=self.cfg['training']['batch_size'],
                                       n_views=self.cfg['training']['n_views'])
            cross_entropy = self.cross_entropy_2d(features.detach().cpu(), labels=classes.detach().cpu(),
                                                  batch_size=self.cfg['training']['batch_size'],
                                                  n_views=self.cfg['training']['n_views'])

        features = F.normalize(features, dim=-1)
        try:
            db = davies_bouldin_score(features.reshape(features.shape[0], -1).cpu().detach().numpy(),
                                  np.array(list(classes.cpu().numpy()) + list(classes.cpu().numpy())))
        except:
            db = 1000 # random high number

        return [loss, cross_entropy], acc, db, features

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> STEP_OUTPUT:
        loss, acc, db, embds = self.batch_step(batch, "Train")
        batch["embeddings"] = embds

        learning_rate = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', learning_rate, on_step=True, on_epoch=False)

        self.log('Train/loss', loss[0])
        self.log('Train/loss_crossentropy', loss[1])
        self.log('Train/acc/top1', acc[0])
        self.log('Train/acc/top5', acc[1])
        self.log('Train/db_score', db)

        if self.global_step % self.train_draw_period == 0:
            self.additional_log(batch, "Train")
            self.val_draw = True

        if self.global_step % self.projector_period == 0 and self.global_step != 0:
            self.log_mitospace(batch)

        return loss[0]

    @torch.no_grad()
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> STEP_OUTPUT:
        loss, acc, db, embds = self.batch_step(batch, "Val")
        batch["embeddings"] = embds

        self.log('Val/loss', loss[0])
        self.log('Val/loss_crossentropy', loss[1])
        self.log('Val/acc/top1', acc[0])
        self.log('Val/acc/top5', acc[1])
        self.log('Val/db_score', db)

        if self.val_draw:
            self.additional_log(batch, "Val")
            self.val_draw = False

        return loss[0]
