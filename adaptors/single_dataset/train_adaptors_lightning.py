import torch
import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from adaptor_dataset import AdaptorDataset
from model import MitoModel, MitoTemporalModel
import argparse
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
import pandas as pd
import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

import warnings
torch.manual_seed(1123)
warnings.filterwarnings("ignore")
warnings.filterwarnings('module')

class MitoLightningModule(pl.LightningModule):
    def __init__(self, model, task_type, num_classes=None, lr=0.0001, min_scale=None, max_scale=None):
        super().__init__()
        self.model = model
        self.task_type = task_type
        self.criterion = nn.CrossEntropyLoss() if task_type == "classification" else nn.L1Loss()
        self.lr = lr
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.num_classes = num_classes
        self.test_predictions = []
        self.test_targets = []
        self.is_final_validation = False
        print("Minimum and Maximum Values are", self.min_scale, self.max_scale)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        embeddings = batch['embedding']
        targets = batch['target'].unsqueeze(1) if self.task_type == "regression" else batch['target']
        outputs = self(embeddings)
        loss = self.criterion(outputs, targets)
        self.log("train_loss", loss, prog_bar=True)

        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("Learning Rate", lr, prog_bar=True, logger=True)

        if self.task_type == "classification":
            preds = torch.argmax(outputs, dim=1)
            acc = (preds == targets).float().mean()
            self.log("train_acc", acc, prog_bar=True)
            return loss

        elif self.task_type == "regression":
            if self.min_scale is not None and self.max_scale is not None:
                outputs = outputs * (self.max_scale - self.min_scale) + self.min_scale
                targets = targets * (self.max_scale - self.min_scale) + self.min_scale

                error = torch.abs(outputs - targets)
                self.log("train_error", error.mean(), prog_bar=True)

                # log the percentage error
                percentage_error = (error / targets) * 100
                self.log("train_percentage_error", percentage_error.mean(), prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):

        embeddings = batch['embedding']
        targets = batch['target']
        outputs = self(embeddings)
        loss = self.criterion(outputs, targets)
        self.log("val_loss", loss, prog_bar=True)

        if self.task_type == "classification":
            preds = torch.argmax(outputs, dim=1)
            acc = (preds == targets).float().mean()
            self.log("val_acc", acc, prog_bar=True)
            return loss

        elif self.task_type == "regression":
            if self.min_scale is not None and self.max_scale is not None:
                outputs = outputs * (self.max_scale - self.min_scale) + self.min_scale
                targets = targets * (self.max_scale - self.min_scale) + self.min_scale

                error = torch.abs(outputs - targets)
                self.log("val_error", error.mean(), prog_bar=True)

                # log the percentage error
                percentage_error = (error / (targets + 1e-8)) * 100
                self.log("val_percentage_error", percentage_error.mean(), prog_bar=True)

                if self.is_final_validation:
                    print("I am storing preds")
                    self.test_predictions.extend(outputs.cpu().numpy().flatten())
                    self.test_targets.extend(targets.cpu().numpy().flatten())

        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return [optimizer], [scheduler]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a MitoModel on MitoTNTDataset using PyTorch Lightning.")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to the CSV file.")
    parser.add_argument("--embeddings_file", type=str, required=True, help="Path to the .npy embeddings file.")
    parser.add_argument("--target_column", type=str, default="Label", help="Column name for the target variable.")
    parser.add_argument("--task_type", type=str, choices=["classification", "regression"], required=True)
    parser.add_argument("--num_classes", type=int, default=None, help="Number of classes for classification.")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for training.")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs to train.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    summer_dataset = AdaptorDataset(
        csv_file=args.csv_file,
        embeddings_file=args.embeddings_file,
        target_column=args.target_column,
        task_type=args.task_type,
        data='summer'
    )

    summer_train_size = int(0.8 * len(summer_dataset))
    summer_test_size = len(summer_dataset) - summer_train_size
    summer_train, summer_test = random_split(summer_dataset, [summer_train_size, summer_test_size])

    # for eg in summer_train:
    #     print(eg)
    #     continue
    # for eg in summer_test:
    #     continue

    train_loader = DataLoader(summer_train, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(summer_test, batch_size=args.batch_size, shuffle=False)

    input_dim = 2048
    # model = MitoModel(input_dim, num_classes=args.num_classes, task=args.task_type)
    model = MitoModel(input_dim, num_classes=args.num_classes, task=args.task_type)

    min_scale, max_scale = None, None
    if args.task_type == "regression":
        min_scale, max_scale = summer_dataset.get_scaling_params()

    save_dir = os.path.join('/home/dhruvagarwal/projects/MitoSpace4D/adaptors/single_dataset/figures',
                            f'{args.task_type}_figures', args.target_column)
    os.makedirs(save_dir, exist_ok=True)

    # Before creating the trainer, add the checkpoint callback lo
    checkpoint_callback = ModelCheckpoint(
        # dirpath=save_dir,
        filename=f'{args.target_column}_best_model',
        save_top_k=1,
        monitor='val_loss',
        mode='min',
        save_last=True
    )

    logger = pl_loggers.TensorBoardLogger(
        save_dir=save_dir,
        name='lightning_logs',
        default_hp_metric=False
    )
    pl_model = MitoLightningModule(model, args.task_type, args.num_classes, args.lr, min_scale, max_scale)

    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        accelerator=args.device,
        logger=logger,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=test_loader)
    print("Finished Training")

    if args.task_type == "regression":
        feature_name = args.target_column
        valid_model = MitoModel(input_dim, num_classes=args.num_classes, task=args.task_type)
        # best_ckpt = os.path.join(logger.log_dir, 'checkpoints', f'{args.target_column}_best_model.ckpt')
        best_ckpt = os.path.join(logger.log_dir, 'checkpoints', f'last.ckpt')
        loaded_state_dict = torch.load(best_ckpt)['state_dict']
        state_dict = {k.replace('model.', ''): v for k, v in loaded_state_dict.items()}
        valid_model.load_state_dict(state_dict)
        valid_model.eval()

        preds = []
        targets = []
        for batch in test_loader:
            embeddings, y = batch['embedding'], batch['target']
            y_hat = valid_model(embeddings)

            preds.append(y_hat)
            targets.append(y)

        preds = torch.cat(preds).squeeze()
        targets = torch.cat(targets)

        preds = preds.cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()

        preds = preds * (max_scale - min_scale) + min_scale
        targets = targets * (max_scale - min_scale) + min_scale

        # preds = np.exp(preds)
        # targets = np.exp(targets)
        preds = preds ** 3
        targets = targets ** 3

        # Create and save predictions DataFrame
        results_df = pd.DataFrame({
            'True_Value': targets,
            'Predicted_Value': preds
        })

        # Calculate final metrics
        results_df['MAE'] = np.abs(
            np.array(results_df['Predicted_Value']) -
            np.array(results_df['True_Value'])
        )

        # Calculate individual MAPE for each sample
        # Adding epsilon (1e-8) to prevent division by zero
        results_df['MAPE'] = np.abs(
            (results_df['True_Value'] - results_df['Predicted_Value']) /
            (results_df['True_Value'] + 1e-8)
        ) * 100

        results_df['RMAE'] = np.abs(
            (results_df['True_Value'] - results_df['Predicted_Value']) /
            (results_df['True_Value'].max() - results_df['True_Value'].min())
        )

        # Calculate and print overall metrics
        overall_mae = results_df['MAE'].mean()
        overall_mape = results_df['MAPE'].mean()
        overall_rmae = results_df['RMAE'].mean()

        print(f"\nOverall Model Performance:")
        print(f"Average MAE: {overall_mae:.4f}")
        print(f"Average MAPE: {overall_mape:.4f}")
        print(f"Average RMAE: {overall_rmae:.4f}")

        # Save predictions and model
        csv_path = os.path.join(save_dir, f'{feature_name}_predictions.csv')
        results_df.to_csv(csv_path, index=False)

        # checkpoint_path = os.path.join(save_dir, f'{feature_name}_final_model.ckpt')
        # trainer.save_checkpoint(checkpoint_path)

        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=results_df, x='True_Value', y='Predicted_Value', alpha=0.6)

        # Add a perfect prediction line (y=x)
        min_val = min(results_df['True_Value'].min(), results_df['Predicted_Value'].min())
        max_val = max(results_df['True_Value'].max(), results_df['Predicted_Value'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

        # Customize the plot
        plt.title(f'True vs Predicted Values for {feature_name}', fontsize=14)
        plt.xlabel('True Value', fontsize=12)
        plt.ylabel('Predicted Value', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save the plot
        plot_path = os.path.join(save_dir, f'{feature_name}_scatter_plot_1_3_scaling.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

        # __________________________________________________________________________________________________________________
        preds = []
        targets = []
        for batch in train_loader:
            embeddings, y = batch['embedding'], batch['target']
            y_hat = model(embeddings)

            preds.append(y_hat)
            targets.append(y)

        preds = torch.cat(preds).squeeze()
        targets = torch.cat(targets)

        preds = preds.cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()

        preds = preds * (max_scale - min_scale) + min_scale
        targets = targets * (max_scale - min_scale) + min_scale

        preds = preds ** 3
        targets = targets ** 3
        # preds = np.exp(preds)
        # targets = np.exp(targets)

        # Create and save predictions DataFrame
        train_results_df = pd.DataFrame({
            'True_Value': targets,
            'Predicted_Value': preds
        })

        min_val = min(results_df['True_Value'].min(), results_df['Predicted_Value'].min())
        max_val = max(results_df['True_Value'].max(), results_df['Predicted_Value'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        sns.scatterplot(data=train_results_df, x='True_Value', y='Predicted_Value', alpha=0.6)

        # Customize the plot
        plt.title(f'Train test - True vs Predicted Values for {feature_name}', fontsize=14)
        plt.xlabel('True Value', fontsize=12)
        plt.ylabel('Predicted Value', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save the plot
        plot_path = os.path.join(save_dir, f'{feature_name}_scatter_plot_train_1_3_scaling.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
