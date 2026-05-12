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


# class Regressor(nn.Module):
#     def __init__(self, input_dim, hidden_dim=256, dropout_prob=0.3):
#         super(Regressor, self).__init__()
#
#         self.regressor = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(p=dropout_prob),
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(p=dropout_prob),
#             nn.Linear(hidden_dim // 2, 1)
#         )
#
#     def forward(self, x):
#         return self.regressor(x)

class NonlinearRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout_prob=0.3):
        super(NonlinearRegressor, self).__init__()

        self.regressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.regressor(x)


class LinearRegressor(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressor, self).__init__()
        self.regressor = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.regressor(x)


def build_regressor(regressor_type, input_dim, hidden_dim=256, dropout_prob=0.3):
    if regressor_type == "linear":
        return LinearRegressor(input_dim=input_dim)
    elif regressor_type == "nonlinear":
        return NonlinearRegressor(input_dim=input_dim, hidden_dim=hidden_dim, dropout_prob=dropout_prob)
    else:
        raise ValueError(f"Unknown regressor_type '{regressor_type}'. Must be 'linear' or 'nonlinear'.")


def train_model(embeddings, targets, save_dir, regression_target="target", desc=None, conditions=None, labels=None,
                regressor_type="nonlinear",
                epochs=100,
                batch_size=32,
                lr=1e-3,
                hidden_dim=256,
                dropout_rate=0.3,
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

    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1))
    y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1))

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train_scaled))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val_scaled))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = build_regressor(
        regressor_type=regressor_type,
        input_dim=embeddings.shape[1],
        hidden_dim=hidden_dim,
        dropout_prob=dropout_rate,
    ).to(device)
    print(f"Using {regressor_type} regressor.")

    # criterion = nn.L1Loss()
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5)

    best_val_loss = float('inf')
    best_epoch = 0

    # Shared output directory for all regression targets — per-target files get
    # namespaced filenames so multiple targets coexist in the same directory.
    if desc is not None:
        output_dir = osp.join(save_dir, f"regressor_outputs_features-{desc}")
    else:
        output_dir = osp.join(save_dir, "regressor_outputs_features")

    # Ensure directory exists before saving
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = osp.join(output_dir, f"best_regressor-{regression_target}.pth")

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
        scheduler.step(avg_val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:03d} | Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), ckpt_path)

    model.load_state_dict(torch.load(ckpt_path))
    joblib.dump(target_scaler, osp.join(output_dir, f"target_scaler-{regression_target}.pkl"))
    return model, target_scaler, X_val, y_val, cond_val, lab_val, best_epoch

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

def evaluate_and_plot(model, X_val, y_val, val_conditions, val_labels, target_scaler, title=None, save_path=None,
                      save_boxplot_path=None, save_csv_path=None, generate_plot=True, feature_name="feature"):
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

    # Normalized Absolute Percent Error: |y - y_hat| / |y| * 100, excluding samples where |y| is ~0
    eps = 1e-8
    valid = np.abs(y_val) > eps
    if valid.any():
        abs_pct_errors = np.abs(y_val[valid] - preds_orig[valid]) / np.abs(y_val[valid]) * 100.0
        nape = abs_pct_errors.mean()
        nape_std = abs_pct_errors.std()
    else:
        nape = np.nan
        nape_std = np.nan

    print("\n--- Final Performance Metrics ---")
    print(f"Global R-squared Score: {r2:.4f}")
    print(f"Global MAE: {mae:.4f} ± {mae_std:.4f}")
    print(f"Global NAPE: {nape:.2f}% ± {nape_std:.2f}%")

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

            valid_cond = np.abs(y_val_cond) > eps
            if valid_cond.any():
                pct_cond = np.abs(y_val_cond[valid_cond] - preds_cond[valid_cond]) / np.abs(y_val_cond[valid_cond]) * 100.0
                nape_cond = pct_cond.mean()
                nape_std_cond = pct_cond.std()
            else:
                nape_cond = np.nan
                nape_std_cond = np.nan

            raw_error_data.append(pd.DataFrame({
                'Condition': cond,
                'Label': cond_label,
                'Absolute Error': abs_errors_cond,
            }))

            condition_results.append({
                'condition': cond,
                'label': cond_label,
                'n_samples': len(y_val_cond),
                'r2_score': r2_cond,
                'mae': mae_cond,
                'mae_std': mae_std_cond,
                'nape': nape_cond,
                'nape_std': nape_std_cond,
            })

        condition_results.append({
            'condition': 'global',
            'label': -1,
            'n_samples': len(y_val),
            'r2_score': r2,
            'mae': mae,
            'mae_std': mae_std,
            'nape': nape,
            'nape_std': nape_std,
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
        plt.ylabel(f'Predicted {feature_name}')
        plt.xlabel(f'Actual {feature_name}')
        plt.grid(True, linestyle='--', alpha=0.6)

        lo = float(min(y_val.min(), preds_orig.min()))
        hi = float(max(y_val.max(), preds_orig.max()))
        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        plt.xlim(lo - pad, hi + pad)
        plt.ylim(lo - pad, hi + pad)
        plt.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                 '--', color='gray', alpha=0.4)


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
            median_order = plot_df.groupby('x_tick')['Absolute Error'].median().sort_values().index
            plot_df['x_tick'] = pd.Categorical(plot_df['x_tick'], categories=median_order, ordered=True)
            plot_df = plot_df.sort_values('x_tick')

            sns.boxplot(
                data=plot_df,
                x='x_tick',
                y='Absolute Error',
                color="#87CEEB",
                saturation=1.0,
                dodge=False,
                legend=False,
                showfliers=False,
                linewidth=1.0,
                medianprops={'color': 'red'},
            )
            plt.grid(True, axis='y', linestyle='--', alpha=0.6)

            plt.xticks(rotation=45, ha='right')
            plt.xlabel('Condition / Label', fontweight='bold', fontsize=14)
            plt.ylabel(f'Absolute Error ({feature_name})', fontweight='bold', fontsize=14)
            plt.tight_layout()

            if save_boxplot_path:
                plt.savefig(save_boxplot_path)
                print(f"Saved MAE boxplot to: {save_boxplot_path}")

            plt.show()

    return r2, mae, nape, results_df


def main(local_df,
         embeddings_dir,
         training_cfg,
         regression_targets=("tmrm_intensities",),
         desc=None,
         pick_labels=None,
         exclude_labels=None,
         generate_plot=True,
         frame_index=-1,
         seed=1123):
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

    # Single shared output directory for all regression targets.
    if desc is not None:
        output_dir = osp.join(embeddings_dir, f"regressor_outputs_features-{desc}")
        combined_csv_path = osp.join(output_dir, f"regressor-all_features-{desc}.csv")
    else:
        output_dir = osp.join(embeddings_dir, "regressor_outputs_features")
        combined_csv_path = osp.join(output_dir, "regressor-all_features.csv")
    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    for regression_target in regression_targets:
        print(f"\n=== Regressing feature: {regression_target} ===")

        target_df = local_df[local_df[regression_target].notna()]
        n_dropped = len(local_df) - len(target_df)
        if n_dropped > 0:
            print(f"Dropped {n_dropped} rows with NaN in '{regression_target}'.")

        min_class_size = target_df['labels'].value_counts().min()
        print(f"Balancing dataset: sampling {min_class_size} samples per class.")
        target_df = target_df.groupby('labels').sample(n=min_class_size, random_state=seed).reset_index(drop=True)

        embeddings = np.stack(target_df['embeddings'].values)
        labels = target_df['labels'].to_numpy()
        label_names = target_df['label_names'].to_numpy(dtype=str)

        # Reduce temporal embeddings to a single frame (targets are scalar per cell)
        if embeddings.ndim == 3:
            embeddings = embeddings[:, frame_index, :]
        elif embeddings.ndim == 2 and isinstance(embeddings[0, frame_index], (list, np.ndarray)):
            embeddings = np.stack([emb[frame_index] for emb in embeddings])

        # Check if the target is a numpy array list or sequence
        if target_df[regression_target][0].ndim > 0:
            # Get the last value only
            target_vals = np.array([val[frame_index] for val in target_df[regression_target].values])
        else:
            target_vals = target_df[regression_target].to_numpy().astype(np.float32)

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
            plot_out_path = osp.join(output_dir, f"regression_plot-{regression_target}-{desc}.png")
            boxplot_out_path = osp.join(output_dir, f"error_boxplot-{regression_target}-{desc}.png")
        else:
            plot_out_path = osp.join(output_dir, f"regression_plot-{regression_target}.png")
            boxplot_out_path = osp.join(output_dir, f"error_boxplot-{regression_target}.png")

        print(f"\nEvaluating best model from checkpoint at epoch {best_epoch}...")

        _, _, _, results_df = evaluate_and_plot(
            model,
            X_val,
            y_val,
            val_conditions,
            val_labels,
            scaler,
            feature_name=regression_target,
            save_path=plot_out_path,
            save_boxplot_path=boxplot_out_path,
            save_csv_path=None,  # aggregated below into a single combined CSV
            generate_plot=generate_plot
        )

        if results_df is not None:
            results_df.insert(0, 'feature', regression_target)
            all_results.append(results_df)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(combined_csv_path, index=False)
        print(f"\nSaved combined per-feature metrics to: {combined_csv_path}")
        print(combined.head(20))


if __name__ == "__main__":
    # embeddings_dir = '/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/data/ms2d_2024v3'
    # embeddings_dir = "/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/data/ms3d_2024v3_225eps"
    embeddings_dir = "/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/data/ms4d_2024v3_252eps"
    # embeddings_dir = "/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/data/ms4d_2024v3_supcon_190eps"
    # embeddings_dir = "/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/data/ms4d_2024v3_zero-shot_241eps"
    # embeddings_dir = '/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/data/ms4d_2024v3_resnet_252eps'
    # embeddings_dir = "/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/data/ms4d_2024v3_tscrambled_284eps"

    # embeddings_dir = "/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/data/ms4d_2024v3_random_init"

    # Columns must already exist in embeddings+metadata.parquet and be scalar per-cell values.
    # regression_targets = ["segment_length_mean", "fragment_diffusivity_mean"]
    # regression_targets = ["tmrm_intensities"]
    regression_targets = [
        "segment_diffusivity_mean",
        "fragment_diffusivity_mean",
        "node_diffusivity_mean",
        "fusion_rate_mean",
        "fission_rate_mean",
        "total_node_count_mean",
        "graph_efficiency_mean",
        "fragment_diameter_mean",
        "segment_length_mean",
        "graph_clustering_coefficient_mean",
        "fragment_branching_index_mean",
        "graph_density_mean",
        "fragment_length_mean",
    ]
    generate_plot = True

    pick_labels = None
    exclude_labels = None
    # desc = None
    frame_index = -1
    # desc = "frame=-1_mnae-plot_subset"
    desc = "frame=-1_all-features"

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
        "regressor_type": "nonlinear",   # "linear" | "nonlinear"
        "epochs": 500,
        "batch_size": 2048,
        "lr": 1e-3,
        "hidden_dim": 1024,
        "dropout_rate": 0.2,
        "test_split": 0.2
    }

    data_infile = osp.join(embeddings_dir, "embeddings+metadata_vis_joined.parquet")
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

    print(df.columns)

    df_exclude = pd.read_parquet(filter_infile)
    print(len(df))
    df = df[~df['image_paths'].isin(df_exclude['image_paths'])]
    print(len(df))

    main(
        local_df=df,
        embeddings_dir=embeddings_dir,
        training_cfg=training_cfg,
        regression_targets=regression_targets,
        desc=desc,
        pick_labels=pick_labels,
        exclude_labels=exclude_labels,
        generate_plot=generate_plot,
        frame_index=frame_index,
        seed=1123
    )