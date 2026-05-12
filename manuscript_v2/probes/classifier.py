import os.path as osp
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from sklearn.model_selection import train_test_split
from tqdm import trange
import einops


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
        # Enforce equal per-class train/test counts deterministically
        class_counts = [int((labels == lbl).sum()) for lbl in unique_labels]
        n_per_class = min(class_counts)
        train_n = int(np.floor(n_per_class * split_perc))

        rng = np.random.RandomState(seed)
        train_indices = []
        val_indices = []
        for lbl in unique_labels:
            lbl_indices = np.where(labels == lbl)[0]
            # If any class is larger, subsample to n_per_class to keep balance
            if len(lbl_indices) > n_per_class:
                lbl_indices = rng.choice(lbl_indices, n_per_class, replace=False)
            # Use an integer train size to avoid rounding mismatches across classes
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

    # Re-shuffle the labels
    if shuffle:
        rng.shuffle(indices)

    return embeddings[indices], labels[indices]


if __name__ == "__main__":
    from torch.utils.data import TensorDataset, DataLoader

    seed = 112358
    np.random.seed(seed)

    n_iterations = 10

    # -- dataset params
    split_frac = 0.9
    # -- train params
    num_epochs = 150
    hidden_dim = 512
    batch_size = 1024
    dropout = 0.3
    # dropout=0.
    probe_type = "nonlinear"  # "linear" or "nonlinear"

    # MS4D hidden_dim=512: 0.8012 ± 0.0080 (nonlinear, n=10, seed+=iteration)
    # DP hidden_dim=512: 0.2722 ± 0.0207 (nonlinear, n=10, seed+=iteration)

    # data_dir = "/home/earkfeld/Desktop/cell_paint_feats/a549"
    # data_dir = "/home/earkfeld/Desktop/cell_paint_feats/hela"

    # data_dir = "/mnt/DATA_01/Eric/mitospace4d_data/runs/embeddings_cancer_r20250929_10frames_modified_labels_for_eval"
    # data_dir = "/mnt/DATA_01/Eric/mitospace4d_data/runs/embeddings_cancer_r20251002_single_frames_modified_labels_for_eval"
    # data_dir = "/home/earkfeld/Projects/MitoSpace4D/runs/exp0_modified_embeddings_cancer-pten_trial4_2024v2-model_ablated-tmrm_eps162_r20251220"
    # data_dir = "/runs/20260109_pten-t4_resnet_embeddings_2024v2-model_tscrambled_modified_labels"

    # data_dir = "/home/earkfeld/Projects/MitoSpace4D/adaptors/pten_classifier/deepprofiler_features/2024v2/2024v2_deepprofiler"
    # data_dir = "/home/earkfeld/Projects/MitoSpace4D/runs/archived_linuxc_embeddings/embeddings_2024v2-encoded_ablated-tmrm_eps162_r20251120"
    # data_dir = "/home/earkfeld/Projects/MitoSpace4D/runs/20260112_2024v2-all_embeddings_resnet3d-kinetics-300eps_ablated-tmrm/"
    # data_dir = "/home/earkfeld/Projects/MitoSpace4D/runs/20260121_liver-drugs_4D-embeddings_2024v2-model"

    data_dirs = [
        # "/home/earkfeld/Projects/MitoSpace4D/runs/20260121_liver-drugs_3D-embeddings_Kinetics3D-model",
        # "/home/earkfeld/Projects/MitoSpace4D/runs/20260117_2024v2-raw_kinetics-resnet3d_ablated_tmrm_extract_tmrm"

        # "/home/earkfeld/Projects/MitoSpace4D/runs/20260121_liver-drugs_4D-embeddings_2024v2-model",
        # "/home/earkfeld/Projects/MitoSpace4D/runs/20260117_2024v2-4D-embeddings_2024v2-161eps_ablated-tmrm"

        # "/home/earkfeld/Projects/MitoSpace4D/runs/20260120_kinetics-60frame-embeddings_2024v2-ft-kinetics-60frame",
        # "/home/earkfeld/Projects/MitoSpace4D/runs/20260116_kinetics-raw_kinetics-resnet3d_ablated_tmrm_extract_tmrm"

        # "/home/earkfeld/Projects/MitoSpace4D/runs/20260117_2024v2-4D-embeddings_2024v2-161eps_ablated-tmrm"
        # "/home/earkfeld/Projects/MitoSpace4D/runs/20260127_2024v2-4D-embeddings_2024v2-161eps_tscrambled_ablated-tmrm"

        # "/home/earkfeld/Projects/MitoSpace4D/data/ms2d_liver_embeddings"
        "/home/earkfeld/Projects/MitoSpace4D/data/2024v2_llsm2confocal_embeddings"
    ]

    accuracies = []
    per_class_all = []
    for iteration in trange(n_iterations):
        current_seed = seed + iteration

        features = []
        labels_raw = []

        for data_dir in data_dirs:
            print("Loading data from ", data_dir, "...")
            embeddings_path = osp.join(data_dir, "embeddings_raw.npy")
            labels_path = osp.join(data_dir, "labels.npy")
            features_raw = np.load(embeddings_path)
            lbls = np.load(labels_path)

            # print(features_raw.shape)

            if len(features_raw.shape) == 3:
                # features_raw = features_raw[:, -1, :]

                features_raw = np.mean(features_raw, axis=1)

                # Shuffle the time dimension
                # rng = np.random.default_rng(current_seed)
                # features_raw = rng.permutation(features_raw, axis=1)
                # features_raw = einops.rearrange(features_raw, "n t d -> n (t d)")

                print(features_raw.shape)

            features.append(features_raw)
            labels_raw.append(lbls)

        features = np.concatenate(features, axis=0)
        labels_raw = np.concatenate(labels_raw, axis=0)

        unique_classes, labels = np.unique(labels_raw, return_inverse=True)
        n_classes = len(unique_classes)

        # n_classes = len(np.unique(labels))
        # print(n_classes)

        # Normalize features
        features = features / np.linalg.norm(features, axis=-1, keepdims=True)
        # print(features.shape)

        # Balance labels
        features, labels = balance_label_counts(features, labels, shuffle=True, seed=current_seed)

        # Stratified per-label split with enforced balance
        train_features, train_labels, val_features, val_labels = split_dataset(
            features, labels, split_perc=split_frac, per_label=True, seed=current_seed
        )

        train_dataset = TensorDataset(
            torch.tensor(train_features, dtype=torch.float32),
            torch.tensor(train_labels, dtype=torch.long),
        )

        val_dataset = TensorDataset(
            torch.tensor(val_features, dtype=torch.float32),
            torch.tensor(val_labels, dtype=torch.long),
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Train/evaluate
        if probe_type == "nonlinear":
            model = NonlinearClassifier(input_dim=features.shape[1], hidden_dim=hidden_dim, num_classes=n_classes, dropout=dropout)
        else:
            model = LinearClassifier(input_dim=features.shape[1], num_classes=n_classes)

        # Pass val_loader to train_model
        model = train_model(model, train_loader, val_loader=val_loader, num_epochs=num_epochs, learning_rate=0.001,
                            device="cuda")

        # Evaluation uses the best checkpoint loaded by train_model
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
