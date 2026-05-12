from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from sklearn.metrics import davies_bouldin_score

from simclr.loss import InfoNCELoss, SupConLoss

torch.manual_seed(0)


class SimCLRRunner(pl.LightningModule):
    def __init__(self, cfg: Dict, model: torch.nn.Module) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.loss = cfg["training"]["loss"]["name"]

        self.intermediate_outputs = []

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            cfg["training"]["lr"],
            weight_decay=cfg["training"]["weight_decay"],
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg["training"]["max_epochs"],
            eta_min=0,
            last_epoch=-1,
        )

        self.data_bank = {"Train": [], "Val": []}

        print(
            f"###################### Using {self.loss} Loss For Training ##################"
        )

        if self.loss == "InfoNCELoss":
            self.criterion = InfoNCELoss(
                use_normalization=cfg["training"]["loss"]["use_normalization"],
                temperature=cfg["training"]["loss"]["temperature"],
            )
        elif self.loss == "SupConLoss":
            self.criterion = SupConLoss(
                use_normalization=cfg["training"]["loss"]["use_normalization"],
                temperature=cfg["training"]["loss"]["temperature"],
                base_temperature=cfg["training"]["loss"]["temperature"],
            )

        self.train_draw_period = cfg["training"]["print_interval"]
        self.projector_period = cfg["training"]["projector_interval"]
        self.val_draw = False

    def flush_bank(self):
        self.data_bank = {"Train": [], "Val": []}

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = self.optimizer
        scheduler = self.scheduler

        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def plot_img(self, img_name: str, img: torch.Tensor) -> None:
        """
        Plot the image and log it to tensorboard

        The function normalizes the image tensor b/w [0-1] before plotting
        """
        img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))
        img = img.cpu()

        if len(img.shape) == 2:
            cm = plt.get_cmap("viridis")
            colored_image = cm(img)[:, :, :3]
        else:
            colored_image = img

        self.logger.experiment.add_image(
            img_name, colored_image, self.global_step, dataformats="HWC"
        )

    def log_mitospace(self, batch):
        if isinstance(batch, Dict):
            assert len(batch["images"]) == 2, "The batch should contain 2 views"
            assert (
                batch["images"][0].shape[0] > 1
            ), "The batch should contain more than 1 sample"
        else:
            assert len(batch[0]) == 2, "The batch should contain 2 views"
            assert (
                batch[0][0].shape[0] > 1
            ), "The batch should contain more than 1 sample"

        if isinstance(batch, Dict):
            images, classes = batch["images"], batch["classes"]
        else:
            images, classes = batch

        # make the embeddings
        embds = batch["embeddings"]
        embds = embds.reshape(embds.shape[0], -1)
        embds = embds[: embds.shape[0] // 2]  # taking only one view
        labels = classes.cpu().numpy()
        labels = list(labels)

        num_timesteps = images[0].shape[0]
        num_z = images[0].shape[1]
        random_timestep = torch.randint(high=num_timesteps, size=(1,))[0]
        random_z = torch.randint(high=num_z, size=(1,))[0]

        images = images[0].cpu()  # taking only one view
        images = images[:, random_timestep, random_z, :, :].unsqueeze(
            1
        )  # taking a random timestep and random z
        images = (images + 1) / 2.0

        self.logger.experiment.add_embedding(
            mat=embds.detach().cpu(),
            metadata=labels,
            label_img=images,
            tag="embeddings",
            global_step=self.global_step,
        )

    def cross_entropy_2d(self, features, labels, batch_size, n_views):
        dvc = features.device
        features = F.normalize(features, dim=-1)
        temperature = 1

        features = torch.stack(
            torch.split(features, [batch_size for _ in range(n_views)]), dim=1
        )

        features = features.view(features.shape[0], features.shape[1], -1)

        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError("Num of labels does not match num of features")
        mask = torch.eq(labels, labels.T).float().to(dvc)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0).float()

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        logits = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), temperature
        )

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        entropy = -torch.sum(mask * F.log_softmax(logits, dim=1), dim=1)

        return entropy.mean()

    def batch_step(
        self, batch: Dict[str, Any], key: str = "Train"
    ) -> tuple[list[Any | None], Any | None, float, torch.Tensor]:
        """Common batch step for train, val, test"""

        images, classes = batch["images"], batch["classes"]

        features, out = self.model(images)

        loss, cross_entropy, acc = None, None, None
        if self.loss == "InfoNCELoss":
            loss, acc = self.criterion(out, bs=self.cfg["training"]["batch_size"])
            cross_entropy = self.cross_entropy_2d(
                features.detach().cpu(),
                labels=classes.detach().cpu(),
                batch_size=self.cfg["training"]["batch_size"],
                n_views=self.cfg["training"]["n_views"],
            )
        elif self.loss == "SupConLoss":
            loss, acc = self.criterion(
                out, labels=classes, bs=self.cfg["training"]["batch_size"]
            )
            cross_entropy = self.cross_entropy_2d(
                features.detach().cpu(),
                labels=classes.detach().cpu(),
                batch_size=self.cfg["training"]["batch_size"],
                n_views=self.cfg["training"]["n_views"],
            )

        features = F.normalize(features, dim=-1)
        try:
            db = davies_bouldin_score(
                features.reshape(features.shape[0], -1).cpu().detach().numpy(),
                np.array(list(classes.cpu().numpy()) + list(classes.cpu().numpy())),
            )
        except:
            db = 1000  # random high number

        return [loss, cross_entropy], acc, db, features

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> STEP_OUTPUT:
        loss, acc, db, embds = self.batch_step(batch, "Train")
        batch["embeddings"] = embds

        learning_rate = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("learning_rate", learning_rate, on_step=True, on_epoch=False)

        self.log("Train/loss", loss[0])
        self.log("Train/loss_crossentropy", loss[1])
        self.log("Train/acc/top1", acc[0])
        self.log("Train/acc/top5", acc[1])
        self.log("Train/db_score", db)

        return loss[0]

    @torch.no_grad()
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> STEP_OUTPUT:
        loss, acc, db, embds = self.batch_step(batch, "Val")
        batch["embeddings"] = embds

        self.log("Val/loss", loss[0])
        self.log("Val/loss_crossentropy", loss[1])
        self.log("Val/acc/top1", acc[0])
        self.log("Val/acc/top5", acc[1])
        self.log("Val/db_score", db)

        return loss[0]
