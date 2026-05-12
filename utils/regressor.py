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
from sklearn.linear_model import Ridge
import joblib
from tqdm import trange
import os
import os.path as osp

def get_label_colormap():
    colors = {}
    with open("/home/earkfeld/Projects/MitoSpace4D/metadata/colors.txt", "r") as file:
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

class RidgeRegressor(nn.Module):
    """Closed-form L2-penalised linear regression. Fits via sklearn's Ridge and
    copies the learned (coef, intercept) into a wrapped nn.Linear so the rest of
    the pipeline (forward pass, eval, state_dict, parameters) sees a normal
    PyTorch model. Skip the gradient training loop for this one in `train_model`."""
    def __init__(self, input_dim, alpha=1.0):
        super(RidgeRegressor, self).__init__()
        self.regressor = nn.Linear(input_dim, 1)
        self.alpha = alpha

    def forward(self, x):
        return self.regressor(x)

    def fit_closed_form(self, X, y):
        """X: (N, D) numpy array, y: (N,) numpy array. Both should already be
        feature- and target-scaled to match what the gradient probes see."""
        sk_ridge = Ridge(alpha=self.alpha, fit_intercept=True)
        sk_ridge.fit(X, y)
        with torch.no_grad():
            self.regressor.weight.copy_(
                torch.from_numpy(sk_ridge.coef_).float().reshape(1, -1)
            )
            self.regressor.bias.copy_(
                torch.from_numpy(np.atleast_1d(sk_ridge.intercept_)).float()
            )


def build_regressor(regressor_type, input_dim, hidden_dim=256, dropout_prob=0.3, ridge_alpha=1.0):
    if regressor_type == "linear":
        return LinearRegressor(input_dim=input_dim)
    elif regressor_type == "nonlinear":
        return NonlinearRegressor(input_dim=input_dim, hidden_dim=hidden_dim, dropout_prob=dropout_prob)
    elif regressor_type == "ridge":
        return RidgeRegressor(input_dim=input_dim, alpha=ridge_alpha)
    else:
        raise ValueError(
            f"Unknown regressor_type '{regressor_type}'. Must be 'linear', 'nonlinear', or 'ridge'."
        )


def stack_input_features(df, feature_columns, frame_index):
    """Concatenate one or more feature columns from df into a single (N, D_total)
    matrix. Each column may hold scalars, 1D vectors, 2D (T, D) sequences, or
    per-row object arrays of vectors; sequences are reduced to `frame_index`."""
    parts = []
    for col in feature_columns:
        first = df[col].iloc[0]
        if hasattr(first, 'ndim') and first.ndim > 0:
            arr = np.stack(df[col].values)
            if arr.ndim == 3:                       # (N, T, D)
                arr = arr[:, frame_index, :]
            elif arr.ndim == 2 and isinstance(df[col].iloc[0][frame_index], (list, np.ndarray)):
                # (N, T) object dtype with per-row 1D vectors → take the frame
                arr = np.stack([v[frame_index] for v in df[col].values])
            elif arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            # else: already (N, D) — no time axis to reduce
        else:
            # Scalar per-row column
            arr = df[col].to_numpy(dtype=np.float32).reshape(-1, 1)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        parts.append(arr.astype(np.float32))
    X = np.concatenate(parts, axis=-1)
    print(f"  Input features {list(feature_columns)} → shape {X.shape}")
    return X


def train_model(embeddings, targets, save_dir, regression_target="target", desc=None, conditions=None, labels=None,
                regressor_type="nonlinear",
                epochs=100,
                batch_size=32,
                lr=1e-3,
                hidden_dim=256,
                dropout_rate=0.3,
                test_split=0.2,
                ridge_alpha=1.0, ):
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

    feature_scaler = StandardScaler()
    X_train = feature_scaler.fit_transform(X_train)
    X_val = feature_scaler.transform(X_val)

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
        ridge_alpha=ridge_alpha,
    ).to(device)
    print(f"Using {regressor_type} regressor.")

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

    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = osp.join(output_dir, f"best_regressor-{regression_target}.pth")

    if regressor_type == "ridge":
        model.fit_closed_form(X_train, y_train_scaled.ravel())
        torch.save(model.state_dict(), ckpt_path)
        joblib.dump(target_scaler, osp.join(output_dir, f"target_scaler-{regression_target}.pkl"))
        joblib.dump(feature_scaler, osp.join(output_dir, f"feature_scaler-{regression_target}.pkl"))
        print(f"Closed-form ridge fit (alpha={ridge_alpha}). Saved weights to {ckpt_path}")
        return model, target_scaler, X_val, y_val, cond_val, lab_val, 0

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
    joblib.dump(feature_scaler, osp.join(output_dir, f"feature_scaler-{regression_target}.pkl"))
    return model, target_scaler, X_val, y_val, cond_val, lab_val, best_epoch

def evaluate_and_plot(model, X_val, y_val, val_conditions, val_labels, target_scaler, title=None, save_path=None,
                      save_boxplot_path=None, save_csv_path=None, generate_plot=True, show_plot=True,
                      feature_name="feature", y_min=None, y_max=None):
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

    # Normalized Absolute Percent Error: |y - y_hat| / |y_max - y_min| * 100,
    # where (y_min, y_max) span the entire dataset (passed in by caller) so the
    # denominator is constant across samples and conditions.
    eps = 1e-9
    if y_min is None:
        y_min = float(np.min(y_val))
    if y_max is None:
        y_max = float(np.max(y_val))
    y_range = y_max - y_min
    if y_range > eps:
        abs_pct_errors = np.abs(y_val - preds_orig) / y_range * 100.0
        nape = abs_pct_errors.mean()
        nape_std = abs_pct_errors.std()
    else:
        abs_pct_errors = np.array([])
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

            if y_range > eps and len(y_val_cond) > 0:
                pct_cond = np.abs(y_val_cond - preds_cond) / y_range * 100.0
                nape_cond = pct_cond.mean()
                nape_std_cond = pct_cond.std()
            else:
                pct_cond = np.array([])
                nape_cond = np.nan
                nape_std_cond = np.nan

            # Boxplot uses per-sample NAPE (%); samples with |y| ≤ eps are
            # excluded since percent error is undefined there.
            raw_error_data.append(pd.DataFrame({
                'Condition': cond,
                'Label': cond_label,
                'NAPE (%)': pct_cond,
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

        # 1. Regression scatter plot — `y_val` and `preds_orig` are already in
        # the normalized [0, 1] space set up in main(), so we plot them directly
        # with axes locked to [0, 1] and y=x as the diagonal.
        plt.figure(figsize=(8, 6), dpi=300)
        sns.regplot(x=y_val,
                    y=preds_orig,
                    scatter_kws={'alpha': 0.3},
                    line_kws={'color': 'red', 'alpha': 0.5},
                    ci=None,
                    )
        plt.ylabel(f'Predicted {feature_name} (min-max normalized)')
        plt.xlabel(f'Actual {feature_name} (min-max normalized)')
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)


        if save_path:
            plt.savefig(save_path)
            print(f"Saved regression plot to: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

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

            # Sort the dataframe according to NAPE so the boxplot is ordered by error magnitude
            median_order = plot_df.groupby('x_tick')['NAPE (%)'].median().sort_values().index
            plot_df['x_tick'] = pd.Categorical(plot_df['x_tick'], categories=median_order, ordered=True)
            plot_df = plot_df.sort_values('x_tick')

            sns.boxplot(
                data=plot_df,
                x='x_tick',
                y='NAPE (%)',
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
            plt.ylabel(f'NAPE (%) — {feature_name}', fontweight='bold', fontsize=14)
            plt.tight_layout()

            if save_boxplot_path:
                plt.savefig(save_boxplot_path)
                print(f"Saved MAE boxplot to: {save_boxplot_path}")

            if show_plot:
                plt.show()
            else:
                plt.close()

    return r2, mae, nape, results_df, abs_pct_errors


def main(local_df,
         embeddings_dir,
         training_cfg,
         regression_targets=("tmrm_intensities",),
         feature_columns=("embeddings",),
         desc=None,
         pick_labels=None,
         exclude_labels=None,
         generate_plot=True,
         show_plot=True,
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
    # Per-target per-sample NAPE arrays — one entry per regression target,
    # used to draw a single cross-feature NAPE boxplot after the loop.
    all_nape_per_sample = []

    for regression_target in regression_targets:
        print(f"\n=== Regressing feature: {regression_target} ===")

        # Drop rows missing either any input feature column or the regression target.
        required_cols = list(feature_columns) + [regression_target]
        target_df = local_df.dropna(subset=required_cols)
        n_dropped = len(local_df) - len(target_df)
        if n_dropped > 0:
            print(f"Dropped {n_dropped} rows with NaN in {required_cols}.")

        min_class_size = target_df['labels'].value_counts().min()
        print(f"Balancing dataset: sampling {min_class_size} samples per class.")
        target_df = target_df.groupby('labels').sample(n=min_class_size, random_state=seed).reset_index(drop=True)

        labels = target_df['labels'].to_numpy()
        label_names = target_df['label_names'].to_numpy(dtype=str)

        # Stack the configured feature columns into a single (N, D_total) input matrix.
        if feature_columns != ("embeddings",):
            inputs = stack_input_features(target_df, feature_columns, frame_index)
        else:
            inputs = target_df

        # Check if the target is a numpy array list or sequence
        if target_df[regression_target][0].ndim > 0:
            # Get the last value only
            target_vals = np.array([val[frame_index] for val in target_df[regression_target].values])
        else:
            target_vals = target_df[regression_target].to_numpy().astype(np.float32)

        # Restrict to the 1st–99th percentile of target values so extreme outliers
        # don't dominate MSE/MAE and skew per-condition NAPE.
        # lo_thr, hi_thr = np.percentile(target_vals, [1, 99])
        # keep_mask = (target_vals >= lo_thr) & (target_vals <= hi_thr)
        # n_kept = int(keep_mask.sum())
        # n_dropped_pct = len(target_vals) - n_kept
        #
        # print(f"Filtering target to 1st–99th pct of {regression_target}: "
        #       f"[{lo_thr:.4g}, {hi_thr:.4g}], kept {n_kept}/{len(target_vals)} "
        #       f"(dropped {n_dropped_pct}).")
        # target_vals = target_vals[keep_mask]
        # if isinstance(inputs, pd.DataFrame):
        #     inputs = inputs.iloc[keep_mask].reset_index(drop=True)
        # else:
        #     inputs = inputs[keep_mask]
        # labels = labels[keep_mask]
        # label_names = label_names[keep_mask]

        # Min-max normalize target_vals to [0, 1] once, here. The rest of the
        # pipeline (training, scaling, NAPE, plotting) then operates entirely in
        # the normalized space, so y_range = 1 and the plot axes map directly to
        # [0, 1] without any per-call re-normalization.
        y_min_raw = float(target_vals.min())
        y_max_raw = float(target_vals.max())
        y_range_raw = y_max_raw - y_min_raw
        if y_range_raw > 0:
            target_vals = ((target_vals - y_min_raw) / y_range_raw).astype(np.float32)
            print(f"Normalized {regression_target} to [0, 1] using raw range "
                  f"[{y_min_raw:.4g}, {y_max_raw:.4g}].")
        y_min_global = 0.0
        y_max_global = 1.0

        model, scaler, X_val, y_val, val_conditions, val_labels, best_epoch = train_model(
            inputs,
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

        _, _, _, results_df, nape_per_sample = evaluate_and_plot(
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
            generate_plot=generate_plot,
            show_plot=show_plot,
            y_min=y_min_global,
            y_max=y_max_global,
        )

        if results_df is not None:
            results_df.insert(0, 'feature', regression_target)
            all_results.append(results_df)

        if nape_per_sample is not None and len(nape_per_sample) > 0:
            all_nape_per_sample.append({
                'feature': regression_target,
                'nape': np.asarray(nape_per_sample),
            })

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(combined_csv_path, index=False)
        print(f"\nSaved combined per-feature metrics to: {combined_csv_path}")
        print(combined.head(20))

    # === Cross-feature NAPE boxplot ===
    # One box per regression target instead of one box per condition. Mirrors the
    # styling of the per-condition boxplot (same color, median highlighting, etc.)
    # but the x-axis ticks are feature names. Per-condition / per-target boxplots
    # produced inside `evaluate_and_plot` are unchanged.
    if generate_plot and all_nape_per_sample:
        nape_dfs = [
            pd.DataFrame({'feature': entry['feature'], 'NAPE (%)': entry['nape']})
            for entry in all_nape_per_sample
        ]
        cross_feature_df = pd.concat(nape_dfs, ignore_index=True)

        # Order boxes by the original `regression_targets` ordering so the x-axis
        # matches the order the user listed the targets in (rather than e.g.
        # sorting by median NAPE).
        feature_order = [t for t in regression_targets
                         if t in set(cross_feature_df['feature'])]
        cross_feature_df['feature'] = pd.Categorical(
            cross_feature_df['feature'], categories=feature_order, ordered=True
        )
        cross_feature_df = cross_feature_df.sort_values('feature')

        fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
        sns.boxplot(
            data=cross_feature_df,
            x='feature',
            y='NAPE (%)',
            color="#87CEEB",
            saturation=1.0,
            dodge=False,
            legend=False,
            showfliers=False,
            linewidth=1.0,
            medianprops={'color': 'red'},
            ax=ax,
        )

        # Lock the plotted region (axes box) to a 12:5 width:height aspect ratio
        # — independent of figsize / tick label sizes — so the figure always
        # renders with the same plot proportions.
        ax.set_box_aspect(5 / 12)
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        # Replace feature-name x-tick labels with 1-based indices matching the
        # `regression_targets` order (so the legend / accompanying CSV is the
        # source of truth for which index is which feature).
        ax.set_xticks(range(len(feature_order)))
        ax.set_xticklabels([str(i + 1) for i in range(len(feature_order))], rotation=0)
        ax.set_xlabel('Regression target index', fontweight='bold', fontsize=14)
        ax.set_ylabel('NAPE (%)', fontweight='bold', fontsize=14)
        fig.tight_layout()

        if desc is not None:
            cross_feature_path = osp.join(output_dir, f"nape_boxplot-all_features-{desc}.png")
        else:
            cross_feature_path = osp.join(output_dir, "nape_boxplot-all_features.png")
        plt.savefig(cross_feature_path)
        print(f"Saved cross-feature NAPE boxplot to: {cross_feature_path}")
        if show_plot:
            plt.show()
        else:
            plt.close()


if __name__ == "__main__":
    # embeddings_dir = '/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/data/ms2d_2024v3'
    # embeddings_dir = "/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/data/ms3d_2024v3_225eps"
    # embeddings_dir = "/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/data/ms4d_2024v3_252eps"
    # embeddings_dir = "/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/data/ms4d_2024v3_supcon_190eps"
    # embeddings_dir = "/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/data/ms4d_2024v3_zero-shot_241eps"
    # embeddings_dir = '/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/data/ms4d_2024v3_resnet_252eps'
    # embeddings_dir = "/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/data/ms4d_2024v3_tscrambled_284eps"

    # embeddings_dir = "/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/data/ms4d_2024v3_random_init"

    # embeddings_dir = "/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/data/ms4d_2024v3_vary-intensity=0.1"
    # embeddings_dir = "/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/data/ms4d_2024v3-binarized_252eps"

    # embeddings_dir = "/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/data/mitotnt_2024v3"

    embeddings_dir = "/home/earkfeld/Projects/MitoSpace4D/manuscript_v2/data/ms4d_testing"

    # Columns must already exist in embeddings+metadata.parquet and be scalar per-cell values.
    # regression_targets = ["segment_length_mean", "fragment_diffusivity_mean"]
    # regression_targets = ["tmrm_intensities"]
    # Input feature columns from the parquet — concatenated along the feature dim.
    # Mix and match: embeddings + scalars (e.g. ["embeddings", "tmrm_intensities"]),
    # several embedding columns, or just scalar features alone.
    feature_columns = ["embeddings"]

    # regression_targets = ["morph_intensities"]
    regression_targets = [
        "fragment_diameter_mean",
        "fragment_length_mean",
        "segment_length_mean",
        "total_node_count_mean",
        "fragment_tortuosity_mean",
        "fragment_branchpoint_to_endpoint_ratio_mean",
        "graph_density_mean",
        "graph_efficiency_mean",
        "fragment_diffusivity_mean",
        "segment_diffusivity_mean",
        "node_diffusivity_mean",
        "fusion_rate_mean",
        "fission_rate_mean",
    ]

    generate_plot = True
    show_plot = False

    pick_labels = None
    exclude_labels = None
    # desc = None
    frame_index = -1

    # desc = "frame=-1_mnae-plot_subset"
    # desc = "frame=-1_tmrm_v4"
    desc = "frame=-1_mitotnt-feats_v3"
    # desc = "tmp"

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

        print(f"Combined data saved to {combined_outfile}.")
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
        feature_columns=feature_columns,
        desc=desc,
        pick_labels=pick_labels,
        exclude_labels=exclude_labels,
        generate_plot=generate_plot,
        show_plot=show_plot,
        frame_index=frame_index,
        seed=1123
    )