import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import joblib
from tqdm import trange
import os
import os.path as osp
import dask.dataframe as dd
import argparse


def get_label_colormap():
    colors = {}
    with open("/home/earkfeld/Projects/MitoSpace4D/extraction_utils/colors.txt", "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 6:
                date, label, index, r, g, b = parts
                if float(r) >= 1 or float(g) >= 1 or float(b) >= 1:
                    colors[int(index)] = [float(r) / 255, float(g) / 255, float(b) / 255]
                else:
                    colors[int(index)] = [float(r), float(g), float(b)]
            else:
                print("Invalid line format:", line)
    return colors


class NonlinearRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, dropout_prob=0.1):
        super(NonlinearRegressor, self).__init__()
        if hidden_dim is None:
            hidden_dim = input_dim

        self.regressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.regressor(x)


class LinearRegressor(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressor, self).__init__()
        self.regressor = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.regressor(x)


def build_regressor(regressor_type, input_dim, hidden_dim=None, dropout_prob=0.1):
    if regressor_type == "linear":
        return LinearRegressor(input_dim=input_dim)
    elif regressor_type == "nonlinear":
        return NonlinearRegressor(input_dim=input_dim, hidden_dim=hidden_dim, dropout_prob=dropout_prob)
    else:
        raise ValueError(f"Unknown regressor_type '{regressor_type}'. Must be 'linear' or 'nonlinear'.")


def l2_normalize(x, eps=1e-9):
    """Row-wise L2 normalization for a 2D feature matrix."""
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (norms + eps)


def build_optimizer_and_scheduler(regressor_type, model, lr, weight_decay, epochs):
    """AdamW+cosine for both probes. AdamW is scale-invariant and converges reliably on
    L2-normed features for MSE regression; an SGD+momentum recipe (DINO-style for
    classification) is too brittle here and can diverge into a negative-R2 checkpoint."""
    if regressor_type == "linear":
        lr = 1e-3 if lr is None else lr
        weight_decay = 1e-4 if weight_decay is None else weight_decay
    elif regressor_type == "nonlinear":
        lr = 1e-3 if lr is None else lr
        weight_decay = 1e-4 if weight_decay is None else weight_decay
    else:
        raise ValueError(f"Unknown regressor_type '{regressor_type}'. Must be 'linear' or 'nonlinear'.")
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    return optimizer, scheduler

"/mnt/aquila/others/Users/Eric/4dms_manuscript-v2_aquila/data/ms4d_2024v3_supcon_190eps"
def train_model(embeddings, targets, save_dir, regression_target="target", desc=None, conditions=None, labels=None,
                regressor_type="nonlinear",
                feature_norm="standardize",
                epochs=100,
                batch_size=32,
                lr=None,
                weight_decay=None,
                hidden_dim=None,
                dropout_rate=0.1,
                test_split=0.2, ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Pass conditions and labels through the split if provided so we can evaluate per-condition later
    if conditions is not None and labels is not None:
        X_train, X_val, y_train, y_val, cond_train, cond_val, lab_train, lab_val = train_test_split(
            embeddings, targets, conditions, labels, test_size=test_split, random_state=42
        )
    elif conditions is not None:
        X_train, X_val, y_train, y_val, cond_train, cond_val = train_test_split(
            embeddings, targets, conditions, test_size=test_split, random_state=42
        )
        lab_val = None
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            embeddings, targets, test_size=test_split, random_state=42
        )
        cond_val = None
        lab_val = None

    feature_scaler = None
    if feature_norm == "standardize":
        feature_scaler = StandardScaler()
        X_train = feature_scaler.fit_transform(X_train)
        X_val = feature_scaler.transform(X_val)
    elif feature_norm == "l2":
        X_train = l2_normalize(X_train)
        X_val = l2_normalize(X_val)
    elif feature_norm != "none":
        raise ValueError(f"Unknown feature_norm '{feature_norm}'. Must be 'standardize', 'l2', or 'none'.")
    print(f"Feature normalization: {feature_norm}")

    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1))
    y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1))

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train_scaled))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val_scaled))

    # Only drop the last batch if it would be size 1 (which breaks BatchNorm); otherwise keep all data.
    drop_last = (len(X_train) % batch_size) == 1
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    print(f"Train samples: {len(X_train)} | Val samples: {len(X_val)} | Train batches/epoch: {len(train_loader)}")

    model = build_regressor(
        regressor_type=regressor_type,
        input_dim=embeddings.shape[1],
        hidden_dim=hidden_dim,
        dropout_prob=dropout_rate,
    ).to(device)
    print(f"Using {regressor_type} regressor.")

    # criterion = nn.L1Loss()
    criterion = nn.MSELoss()
    optimizer, scheduler = build_optimizer_and_scheduler(
        regressor_type=regressor_type,
        model=model,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
    )

    best_val_loss = float('inf')
    best_epoch = 0

    # Construct the output directory path inside the embeddings dir
    if desc is not None:
        output_dir = osp.join(save_dir, f"regressor_outputs_{regression_target}-{desc}")
    else:
        output_dir = osp.join(save_dir, f"regressor_outputs_{regression_target}")

    # Ensure directory exists before saving
    os.makedirs(output_dir, exist_ok=True)

    for epoch in trange(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                v_out = model(vx)
                val_loss += criterion(v_out, vy).item() * vx.size(0)

        avg_train_loss = train_loss / len(X_train)
        avg_val_loss = val_loss / len(X_val)
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:03d} | Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), osp.join(output_dir, 'best_regressor.pth'))

    model.load_state_dict(torch.load(osp.join(output_dir, 'best_regressor.pth')))
    joblib.dump(target_scaler, osp.join(output_dir, 'target_scaler.pkl'))
    if feature_scaler is not None:
        joblib.dump(feature_scaler, osp.join(output_dir, 'feature_scaler.pkl'))
    return model, target_scaler, X_val, y_val, cond_val, lab_val, best_epoch

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

def evaluate_and_plot(model, X_val, y_val, val_conditions, val_labels, target_scaler, title=None, save_path=None,
                      save_boxplot_path=None, save_csv_path=None, generate_plot=True):
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        inputs = torch.FloatTensor(X_val).to(device)
        preds_scaled = model(inputs).cpu().numpy()

    preds_orig = target_scaler.inverse_transform(preds_scaled).flatten()

    r2 = r2_score(y_val, preds_orig)

    abs_errors = np.abs(y_val - preds_orig)
    mae = abs_errors.mean()
    mae_std = abs_errors.std()

    print("\n--- Final Performance Metrics ---")
    print(f"Global R-squared Score: {r2:.4f}")
    print(f"Global MAE: {mae * 100:.2f}% ± {mae_std * 100:.2f}%")

    results_df = None
    if val_conditions is not None:
        unique_conditions = np.unique(val_conditions)
        condition_results = []
        raw_error_data = []

        for cond in unique_conditions:
            mask = (val_conditions == cond)
            y_val_cond = y_val[mask]
            preds_cond = preds_orig[mask]

            if val_labels is not None:
                cond_label = val_labels[mask][0]
            else:
                cond_label = -1

            if len(y_val_cond) > 1:
                r2_cond = r2_score(y_val_cond, preds_cond)
            else:
                r2_cond = np.nan

            abs_errors_cond = np.abs(y_val_cond - preds_cond)
            mae_cond = abs_errors_cond.mean()
            mae_std_cond = abs_errors_cond.std()

            raw_error_data.append(pd.DataFrame({
                'Condition': cond,
                'Label': cond_label,
                'Absolute Error (%)': abs_errors_cond * 100
            }))

            condition_results.append({
                'condition': cond,
                'label': cond_label,
                'n_samples': len(y_val_cond),
                'r2_score': r2_cond,
                'mae': mae_cond,
                'mae_std': mae_std_cond,
            })

        condition_results.append({
            'condition': 'global',
            'label': -1,
            'n_samples': len(y_val),
            'r2_score': r2,
            'mae': mae,
            'mae_std': mae_std,
        })

        results_df = pd.DataFrame(condition_results)

        if save_csv_path:
            results_df.to_csv(save_csv_path, index=False)
            print(f"\nSaved per-condition metrics to: {save_csv_path}")
            print(results_df.head(10))

    if generate_plot:

        # 1. Regression scatter plot
        # plt.figure(figsize=(5, 7))
        plt.figure(figsize=(8, 6), dpi=300)
        sns.regplot(x=y_val,
                    y=preds_orig,
                    scatter_kws={'alpha': 0.3},
                    line_kws={'color': 'red', 'alpha': 0.5},
                    ci=None,
                    )
        plt.ylabel('Predicted Normalized Intensity')
        plt.xlabel('Actual Normalized Intensity')
        # plt.title(f'Regression Results (Global R²: {r2:.4f})')
        plt.grid(True, linestyle='--', alpha=0.6)
        # plt.xlim(0, 0.8)
        # plt.ylim(0, 0.8)

        lims = (0, 0.5)
        plt.ylim(lims)
        plt.xlim(lims)


        if save_path:
            plt.savefig(save_path)
            print(f"Saved regression plot to: {save_path}")

        plt.show()

        # 2. Per-condition MAE boxplot
        if val_conditions is not None and len(raw_error_data) > 0:

            plot_df = pd.concat(raw_error_data, ignore_index=True)
            plot_df = plot_df.sort_values('Label')
            plot_df['x_tick'] = plot_df['Condition'].astype(str)

            plt.figure(figsize=(16, 4), dpi=300)

            label_colors = get_label_colormap()  # Assuming this is defined elsewhere in your code
            unique_mapping = plot_df[['x_tick', 'Label']].drop_duplicates()
            palette = {row['x_tick']: label_colors.get(int(row['Label']), [0.5, 0.5, 0.5]) for _, row in
                       unique_mapping.iterrows()}

            # Sort the dataframe according to absolute error to ensure the boxplot colors are ordered by error magnitude
            median_order = plot_df.groupby('x_tick')['Absolute Error (%)'].median().sort_values().index
            plot_df['x_tick'] = pd.Categorical(plot_df['x_tick'], categories=median_order, ordered=True)
            plot_df = plot_df.sort_values('x_tick')

            # sns.boxplot(
            #     data=plot_df,
            #     x='x_tick',
            #     y='Absolute Error (%)',
            #     # palette=palette,
            #     color="skyblue",
            #     # hue='x_tick',
            #     dodge=False,
            #     legend=False,
            #     showfliers=False,
            #     width=0.6
            # )

            # Use color code #87CEEB

            # sns.boxplot(
            #     data=plot_df,
            #     x='x_tick',
            #     y='Absolute Error (%)',
            #     color="skyblue",
            #     dodge=False,
            #     legend=False,
            #     showfliers=False,
            #     width=0.6,
            #     medianprops={'color': 'red'}  # <--- Add this line
            # )
            sns.boxplot(
                data=plot_df,
                x='x_tick',
                y='Absolute Error (%)',
                color="#87CEEB",  # Updated to use the hex code
                saturation=1.0,
                dodge=False,
                legend=False,
                showfliers=False,
                # width=0.6,
                linewidth=1.0,
                medianprops={'color': 'red'},
            )
            # plt.grid(True, linestyle='--', alpha=0.6)
            plt.grid(True, axis='y', linestyle='--', alpha=0.6)

            plt.xticks(rotation=45, ha='right')
            plt.xlabel('Condition / Label', fontweight='bold', fontsize=14)
            plt.ylabel('NAPE (%)', fontweight='bold', fontsize=14)
            # plt.title('NAPE Distribution by Condition', fontsize=14)
            plt.ylim(0, 12)
            plt.tight_layout()

            if save_boxplot_path:
                plt.savefig(save_boxplot_path)
                print(f"Saved MAE boxplot to: {save_boxplot_path}")

            plt.show()

    return r2, mae


def normalize(x):
    """Normalize array or tensor to [0, 1] range."""
    return (x - x.min()) / (x.max() - x.min() + 1e-9)


def main(local_df,
         embeddings_dir,
         training_cfg,
         regression_target="tmrm_intensities",
         desc=None,
         pick_labels=None,
         exclude_labels=None,
         generate_plot=True,
         frame_index=-1,
         seed=1123):
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("Dataframe columns:", list(local_df.columns))

    if exclude_labels:
        print(f"Excluding labels: {exclude_labels}")
        local_df = local_df[~local_df['labels'].isin(exclude_labels)]

    if pick_labels:
        local_df = local_df[local_df['labels'].isin(pick_labels)]

    min_class_size = local_df['labels'].value_counts().min()
    print(f"Balancing dataset: sampling {min_class_size} samples per class.")
    local_df = local_df.groupby('labels').sample(n=min_class_size, random_state=seed).reset_index(drop=True)

    # 1. Force everything into clean NumPy arrays and stack the sequences
    embeddings = np.stack(local_df['embeddings'].values)
    labels = local_df['labels'].to_numpy()
    label_names = local_df['label_names'].to_numpy(dtype=str)

    # Use np.stack for targets because they are sequences (time frames) too!
    target_vals = np.stack(local_df[regression_target].values)

    # 2. Get specified frame only
    if embeddings.ndim == 3:
        embeddings = embeddings[:, frame_index, :]
        target_vals = target_vals[:, frame_index]
    elif embeddings.ndim == 2 and isinstance(embeddings[0, frame_index], (list, np.ndarray)):
        embeddings = np.stack([emb[frame_index] for emb in embeddings])
        target_vals = np.array([tgt[frame_index] for tgt in target_vals])

    # >>> EXPERIMENT percentile filtering
    # # Keep 0.01 - 99.9 percentile range for  to avoid extreme outliers dominating the regression
    # high_val = np.percentile(target_vals, 99.9, axis=0)
    # low_val = np.percentile(target_vals, 0.01, axis=0)
    # idxs_to_keep =  (target_vals <= high_val) & (target_vals >= low_val)
    # print(f"Filtering out {np.sum(~idxs_to_keep)} extreme outliers outside 0.01-99.9 percentile range.")
    # embeddings = embeddings[idxs_to_keep]
    # labels = labels[idxs_to_keep]
    # label_names = label_names[idxs_to_keep]
    # target_vals = target_vals[idxs_to_keep]
    # <<< EXPERIMENT percentile filtering

    # Normalize targets globally across the dataset to [0, 1] range
    target_vals = normalize(target_vals)

    model, scaler, X_val, y_val, val_conditions, val_labels, best_epoch = train_model(
        embeddings,
        target_vals,
        save_dir=embeddings_dir,
        regression_target=regression_target,
        desc=desc,
        conditions=label_names,
        labels=labels,
        **training_cfg
    )

    if desc is not None:
        output_dir = osp.join(embeddings_dir, f"regressor_outputs_{regression_target}-{desc}")
        csv_out_path = osp.join(output_dir, f"regressor-{regression_target}-{desc}.csv")
        plot_out_path = osp.join(output_dir, f"regression_plot-{regression_target}-{desc}.png")
        boxplot_out_path = osp.join(output_dir, f"nape_boxplot-{regression_target}-{desc}.png")
    else:
        output_dir = osp.join(embeddings_dir, f"regressor_outputs_{regression_target}")
        csv_out_path = osp.join(output_dir, f"regressor-{regression_target}.csv")
        plot_out_path = osp.join(output_dir, f"regression_plot-{regression_target}.png")
        boxplot_out_path = osp.join(output_dir, f"nape_boxplot-{regression_target}.png")

    print(f"\nEvaluating best model from checkpoint at epoch {best_epoch}...")

    evaluate_and_plot(
        model,
        X_val,
        y_val,
        val_conditions,
        val_labels,
        scaler,
        save_path=plot_out_path,
        save_boxplot_path=boxplot_out_path,
        save_csv_path=csv_out_path,
        generate_plot=generate_plot
    )


def get_label_colormap():
    colors = {}
    with open(f"/home/earkfeld/Projects/MitoSpace4D/extraction_utils/colors.txt", "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 6:
                date, label, index, r, g, b = parts
                if float(r) >= 1 or float(g) >= 1 or float(b) >= 1:
                    colors[int(index)] = [float(r) / 255, float(g) / 255, float(b) / 255]
                else:
                    colors[int(index)] = [float(r), float(g), float(b)]
            else:
                print("Invalid line format:", line)
    return colors


if __name__ == "__main__":
    # embeddings_dir = '/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/data/ms2d_2024v3'
    # embeddings_dir = "/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/data/ms3d_2024v3_225eps"
    # embeddings_dir = "/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/data/ms4d_2024v3_252eps"
    # embeddings_dir = "/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/data/ms4d_2024v3_supcon_190eps"
    # embeddings_dir = "/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/data/ms4d_2024v3_zero-shot_241eps"
    # embeddings_dir = '/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/data/ms4d_2024v3_resnet_252eps'
    # embeddings_dir = "/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/data/ms4d_2024v3_tscrambled_284eps"

    embeddings_dir = "/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/data/ms4d_2024v3_random_init"

    regression_target = "tmrm_intensities"
    generate_plot = True

    pick_labels = None
    exclude_labels = None
    # desc = None
    frame_index = -1
    # desc = "frame=-1_mnae-plot_subset"
    desc = "frame=-1_filtered_sorted_recolored"

    # desc = "non_extreme_conditions"
    # exclude_labels = [4, 5, 6, 9, 11, 12, 13, 19, 23, 24]  # Overrides pick_labels

    # training_cfg = {
    #     "epochs": 500,
    #     "batch_size": 2048,
    #     "lr": 1e-3,
    #     "hidden_dim": 1048,
    #     "dropout_rate": 0.2,
    #     "test_split": 0.2
    # }

    training_cfg = {
        "regressor_type": "linear",     # "linear" or "nonlinear"
        "feature_norm": "standardize",  # "standardize", "l2", or "none"
        "epochs": 100,
        "batch_size": 2048,
        "test_split": 0.2,
        # lr / weight_decay / hidden_dim / dropout_rate omitted -> recipe defaults
    }

    data_infile = osp.join(embeddings_dir, "embeddings+metadata.parquet")
    filter_infile = "/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/data/2024v3_exclude_paths.parquet"

    if osp.exists(data_infile):
        print(f"Loading data from {data_infile}...")
        df = pd.read_parquet(data_infile)
    else:
        print("Creating datafile from embeddings and intensities...")
        intensities_infile = "/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/data/2024v3_channel_intensities.parquet"
        embeddings = np.load(osp.join(embeddings_dir, "embeddings.npy"))
        labels = np.load(osp.join(embeddings_dir, "labels.npy"))
        label_names = np.load(osp.join(embeddings_dir, "label_names.npy"))
        image_paths = np.loadtxt(osp.join(embeddings_dir, 'image_paths.csv'), dtype=str).tolist()

        df = pd.read_parquet(intensities_infile)
        df = df.rename(columns={'morph_path': 'image_paths'})

        df_embeddings = pd.DataFrame({
            'image_paths': image_paths,
            'labels': labels,
            'embeddings': embeddings.tolist(),
            'label_names': [label_names[lbl] for lbl in labels],
        })

        df = df.merge(df_embeddings, on='image_paths', how='inner')
        combined_outfile = osp.join(embeddings_dir, "embeddings+metadata.parquet")
        df.to_parquet(combined_outfile, index=False)

        print(f"Combined data saved to {combined_outfile}. Loading as Dask DataFrame for efficient processing...")
        del embeddings, labels, label_names, image_paths, df_embeddings, df
        df = pd.read_parquet(combined_outfile)

    df_exclude = pd.read_parquet(filter_infile)
    print(len(df))
    df = df[~df['image_paths'].isin(df_exclude['image_paths'])]
    print(len(df))

    main(
        local_df=df,
        embeddings_dir=embeddings_dir,
        training_cfg=training_cfg,
        regression_target=regression_target,
        desc=desc,
        pick_labels=pick_labels,
        exclude_labels=exclude_labels,
        generate_plot=generate_plot,
        frame_index=frame_index,
        seed=1123
    )