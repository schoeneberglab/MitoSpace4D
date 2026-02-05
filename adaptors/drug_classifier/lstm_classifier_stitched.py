import os.path as osp
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from sklearn.model_selection import train_test_split
from tqdm import trange
import einops
import pandas as pd


class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=1):
        super(LinearClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        return self.linear(x)


class NonlinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=None, dropout=0.2):
        super(NonlinearClassifier, self).__init__()

        if hidden_dim is None:
            hidden_dim = input_dim

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.network(x)


class BiLSTMClassifier(nn.Module):
    """
    BiLSTM classifier for per-frame embeddings.

    Expected input:
      x: (N, T, D) where
         N = batch size
         T = sequence length (frames)
         D = embedding dimension
    Output:
      logits: (N, C)
    """
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_layers: int = 1,
        lstm_dropout: float = 0.0,   # only applied if num_layers > 1
        head_dropout: float = 0.2,
        pool: str = "last",          # "last" or "mean"
    ):
        super().__init__()

        if pool not in ("last", "mean"):
            raise ValueError(f"pool must be 'last' or 'mean', got {pool}")

        self.pool = pool

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=(lstm_dropout if num_layers > 1 else 0.0),
            bidirectional=True,
            batch_first=True,
        )

        # BiLSTM => 2 * hidden_dim
        self.head = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=head_dropout),
            nn.Linear(2 * hidden_dim, num_classes),
        )

    def forward(self, x):
        # x: (N, T, D)
        out, _ = self.lstm(x)  # out: (N, T, 2H)

        if self.pool == "mean":
            pooled = out.mean(dim=1)      # (N, 2H)
        else:
            pooled = out[:, -1, :]        # (N, 2H) (forward last + backward corresponding to first)

        return self.head(pooled)


def _l2_normalize_np(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


def train_model(model, data_loader, val_loader=None, num_epochs=10, learning_rate=1e-3, device="cuda",
                show_loss=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
    model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    best_epoch = 0

    for epoch in trange(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)  # (N,) long

            optimizer.zero_grad()
            logits = model(inputs)      # (N, C)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(data_loader.dataset)

        if val_loader is not None:
            model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for v_inputs, v_labels in val_loader:
                    v_inputs = v_inputs.to(device)
                    v_labels = v_labels.to(device)  # (N,) long
                    v_logits = model(v_inputs)
                    v_loss = criterion(v_logits, v_labels)
                    val_running_loss += v_loss.item() * v_inputs.size(0)

            val_loss = val_running_loss / len(val_loader.dataset)
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1

        if show_loss:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    if val_loader is not None:
        model.load_state_dict(best_model_wts)

    print(f"Best validation loss: {best_loss:.4f} at epoch {best_epoch}")
    return model


def evaluate_model(model, data_loader, device="cuda", show_acc=False):
    device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    class_correct = {}
    class_total = {}

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)          # (N,) long
            logits = model(inputs)              # (N, C)
            predicted = logits.argmax(dim=1)    # (N,)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for t, p in zip(labels.view(-1), predicted.view(-1)):
                t_val = t.item()
                if t_val not in class_total:
                    class_total[t_val] = 0
                    class_correct[t_val] = 0
                class_total[t_val] += 1
                if t_val == p.item():
                    class_correct[t_val] += 1

    accuracy = correct / total
    class_accuracies = {cls: class_correct[cls] / class_total[cls] for cls in class_total}

    if show_acc:
        print(f"Accuracy: {accuracy:.4f}")
    return accuracy, class_accuracies


def split_dataset(embeddings, labels, split_perc=0.9, per_label=True, seed=1123):
    if per_label:
        unique_labels = np.unique(labels)
        class_counts = [int((labels == lbl).sum()) for lbl in unique_labels]
        n_per_class = min(class_counts)
        train_n = int(np.floor(n_per_class * split_perc))

        rng = np.random.RandomState(seed)
        train_indices = []
        val_indices = []
        for lbl in unique_labels:
            lbl_indices = np.where(labels == lbl)[0]
            if len(lbl_indices) > n_per_class:
                lbl_indices = rng.choice(lbl_indices, n_per_class, replace=False)
            train_idx, val_idx = train_test_split(
                lbl_indices, train_size=train_n, random_state=seed, shuffle=True
            )
            train_indices.extend(train_idx)
            val_indices.extend(val_idx)

        train_indices = np.array(train_indices, dtype=int)
        val_indices = np.array(val_indices, dtype=int)
    else:
        all_indices = np.arange(len(labels))
        train_indices, val_indices = train_test_split(
            all_indices, train_size=split_perc, random_state=seed, shuffle=True
        )
        train_indices = np.array(train_indices, dtype=int)
        val_indices = np.array(val_indices, dtype=int)

    return (
        embeddings[train_indices],
        labels[train_indices],
        embeddings[val_indices],
        labels[val_indices],
    )


def balance_label_counts(embeddings, labels, seed=1123, shuffle=True):
    embeddings = np.asarray(embeddings)
    labels = np.asarray(labels)

    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError(f"embeddings has {embeddings.shape[0]} rows but labels has length {labels.shape[0]}")

    rng = np.random.default_rng(seed)

    unique_labels = np.unique(labels)
    label_counts = {lbl: np.sum(labels == lbl) for lbl in unique_labels}
    min_count = min(label_counts.values())

    indices = []
    for lbl in unique_labels:
        lbl_indices = np.where(labels == lbl)[0]
        selected = rng.choice(lbl_indices, size=min_count, replace=False)
        indices.append(selected)

    indices = np.concatenate(indices)

    if shuffle:
        rng.shuffle(indices)

    return embeddings[indices], labels[indices]


if __name__ == "__main__":
    from torch.utils.data import TensorDataset, DataLoader

    seed = 112358
    np.random.seed(seed)

    n_iterations = 1

    # -- dataset params
    split_frac = 0.9
    # -- train params
    num_epochs = 100
    hidden_dim = 512          # BiLSTM hidden size (per direction uses this as hidden_size; output is 2*hidden_dim)
    batch_size = 1024
    dropout = 0.3
    probe_type = "bilstm"     # "linear" or "nonlinear" or "bilstm"

    data_dirs = [
        # "/home/earkfeld/Projects/MitoSpace4D/runs/20260116_kinetics-raw_kinetics-resnet3d_ablated_tmrm_extract_tmrm"
        # "/home/earkfeld/Projects/MitoSpace4D/runs/20260117_2024v2-raw_kinetics-resnet3d_ablated_tmrm_extract_tmrm"
        "/home/earkfeld/Projects/MitoSpace4D/runs/20260117_2024v2-4D-embeddings_2024v2-161eps_ablated-tmrm"
    ]

    accuracies = []
    per_class_all = []
    for iteration in trange(n_iterations):
        current_seed = seed + iteration

        features = []
        labels_raw = []

        for data_dir in data_dirs:
            print("Loading data from ", data_dir, "...")
            # embeddings_path = osp.join(data_dir, "embeddings_raw.npy")
            embeddings_path = osp.join(data_dir, "embeddings_resnet.npy")
            labels_path = osp.join(data_dir, "labels.npy")
            features_raw = np.load(embeddings_path)
            lbls = np.load(labels_path)
            img_paths = np.loadtxt(osp.join(data_dir, "image_paths.csv"), delimiter=",", dtype=str)

            df = pd.DataFrame({
                "img_path": img_paths,
                # "labels": lbls
            })

            df["path_id"] = df['img_path'].apply(lambda x: x.split("ed_data/")[-1][:-6])
            df["movie_id"] = df['img_path'].apply(lambda x: int(x.split("ed_data/")[-1].split(".npy")[0].split("-")[-1]))

            unique_path_ids = np.unique(df["path_id"])
            new_lbls = []
            new_features_raw = []
            for uid in unique_path_ids:
                rowset = df[df["path_id"] == uid]
                rowset = rowset.sort_values(by="movie_id", ascending=True)

                # Get the index of the first movie_id in the main dataframe
                first_index = rowset.index[0]
                lbl = lbls[first_index]
                new_lbls.append(lbl)

                # Stack the embeddings for each frame using the indices
                features_raw_for_uid = features_raw[rowset.index]
                new_features_raw.append(np.concatenate(features_raw_for_uid, axis=0))


            # Ensure (N, T, D) for BiLSTM.
            # - If already (N, T, D): optionally shuffle time.
            # - If (N, D): make it (N, 1, D).
            if features_raw.ndim == 3:
                # Shuffle the time dimension (if you want the scrambled-time condition).
                rng = np.random.default_rng(current_seed)
                features_raw = rng.permutation(features_raw, axis=1)  # (N, T, D)

                # Keep every 3rd frame
                # features_raw = features_raw[:, ::3, :]

                # L2-normalize per frame embedding (along D)
                features_raw = _l2_normalize_np(features_raw, axis=-1)
            elif features_raw.ndim == 2:
                # (N, D) -> (N, 1, D)
                features_raw = features_raw[:, None, :]
                features_raw = _l2_normalize_np(features_raw, axis=-1)
            else:
                raise ValueError(f"Unsupported embeddings shape: {features_raw.shape}")

            features.append(features_raw)
            labels_raw.append(lbls)

        features = np.concatenate(features, axis=0)     # (N, T, D)
        labels_raw = np.concatenate(labels_raw, axis=0)

        unique_classes, labels = np.unique(labels_raw, return_inverse=True)
        n_classes = len(unique_classes)

        # Balance labels
        features, labels = balance_label_counts(features, labels, shuffle=True, seed=current_seed)

        # Stratified per-label split with enforced balance
        train_features, train_labels, val_features, val_labels = split_dataset(
            features, labels, split_perc=split_frac, per_label=True, seed=current_seed
        )

        train_dataset = TensorDataset(
            torch.tensor(train_features, dtype=torch.float32),  # (N, T, D)
            torch.tensor(train_labels, dtype=torch.long),
        )

        val_dataset = TensorDataset(
            torch.tensor(val_features, dtype=torch.float32),    # (N, T, D)
            torch.tensor(val_labels, dtype=torch.long),
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Train/evaluate
        if probe_type == "bilstm":
            model = BiLSTMClassifier(
                input_dim=train_features.shape[-1],   # D
                num_classes=n_classes,
                hidden_dim=hidden_dim,
                num_layers=1,
                lstm_dropout=0.0,
                head_dropout=dropout,
                pool="last",  # or "mean"
            )
        elif probe_type == "nonlinear":
            # For MLP probes, flatten (N, T, D) -> (N, T*D)
            tr_flat = einops.rearrange(train_features, "n t d -> n (t d)")
            va_flat = einops.rearrange(val_features, "n t d -> n (t d)")

            train_dataset = TensorDataset(
                torch.tensor(tr_flat, dtype=torch.float32),
                torch.tensor(train_labels, dtype=torch.long),
            )
            val_dataset = TensorDataset(
                torch.tensor(va_flat, dtype=torch.float32),
                torch.tensor(val_labels, dtype=torch.long),
            )
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            model = NonlinearClassifier(
                input_dim=tr_flat.shape[1],
                hidden_dim=tr_flat.shape[1] if hidden_dim is None else hidden_dim,
                num_classes=n_classes,
                dropout=dropout,
            )
        else:
            # Linear probe also needs flattened inputs
            tr_flat = einops.rearrange(train_features, "n t d -> n (t d)")
            va_flat = einops.rearrange(val_features, "n t d -> n (t d)")

            train_dataset = TensorDataset(
                torch.tensor(tr_flat, dtype=torch.float32),
                torch.tensor(train_labels, dtype=torch.long),
            )
            val_dataset = TensorDataset(
                torch.tensor(va_flat, dtype=torch.float32),
                torch.tensor(val_labels, dtype=torch.long),
            )
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            model = LinearClassifier(input_dim=tr_flat.shape[1], num_classes=n_classes)

        model = train_model(
            model,
            train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            learning_rate=0.001,
            device="cuda",
        )

        accuracy, class_accs = evaluate_model(model, val_loader, device="cuda")
        accuracies.append(accuracy)
        per_class_all.append(class_accs)

    accuracies = np.array(accuracies)
    print(f"Mean accuracy over {n_iterations} iterations: {accuracies.mean():.4f} ± {accuracies.std():.4f}")
    print("All accuracies:", accuracies.tolist())

    if per_class_all:
        print("-" * 30)
        print("Per-class Average Accuracies:")
        all_keys = set().union(*[d.keys() for d in per_class_all])
        for k in sorted(all_keys):
            vals = [d[k] for d in per_class_all if k in d]
            mean_val = np.mean(vals)
            std_val = np.std(vals)

            label_str = k
            if 'unique_classes' in locals() and k < len(unique_classes):
                label_str = unique_classes[k]

            print(f"{label_str}: {mean_val:.4f} ± {std_val:.4f}")