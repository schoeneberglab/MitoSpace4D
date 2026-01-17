import os.path as osp
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
from tqdm import trange


class LinearBinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(LinearBinaryClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


class SimpleNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(SimpleNNClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


def train_model(model, data_loader, val_loader=None, num_epochs=10, learning_rate=0.001, device="cuda",
                show_loss=False):
    criterion = nn.BCELoss()  # Binary Cross-Entropy
    # criterion = nn.BCEWithLogitsLoss() if isinstance(model, LinearBinaryClassifier) else criterion
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
    model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(data_loader.dataset)

        # Validation phase
        if val_loader is not None:
            model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for v_inputs, v_labels in val_loader:
                    v_inputs = v_inputs.to(device)
                    v_labels = v_labels.to(device).float().unsqueeze(1)
                    v_outputs = model(v_inputs)
                    v_loss = criterion(v_outputs, v_labels)
                    val_running_loss += v_loss.item() * v_inputs.size(0)

            val_loss = val_running_loss / len(val_loader.dataset)

            # Deep copy the model if it's the best loss so far
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        if show_loss:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # Load best model weights
    if val_loader is not None:
        model.load_state_dict(best_model_wts)

    return model


def evaluate_model(model, data_loader, device="cuda", show_acc=False):
    device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    if show_acc:
        print(f"Accuracy: {accuracy:.4f}")
    return accuracy


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


if __name__ == "__main__":
    from torch.utils.data import TensorDataset, DataLoader

    np.random.seed(1123)
    n_iterations = 10  # 100

    # -- dataset params
    # n_samples = 465
    # n_samples = None
    split_frac = 0.5  # 0.9
    n_samples = 696 * 2

    # -- train params
    num_epochs = 100  # 50
    hidden_dim = 256  # 128
    probe_type = "nonlinear"  # "linear" or "nonlinear"

    # data_dir = "/home/earkfeld/Desktop/cell_paint_feats/a549"
    # data_dir = "/home/earkfeld/Desktop/cell_paint_feats/hela"

    # data_dir = "/mnt/DATA_01/Eric/mitospace4d_data/runs/embeddings_cancer_r20250929_10frames_modified_labels_for_eval"
    # data_dir = "/mnt/DATA_01/Eric/mitospace4d_data/runs/embeddings_cancer_r20251002_single_frames_modified_labels_for_eval"
    # data_dir = "/home/earkfeld/Projects/MitoSpace4D/runs/exp0_modified_embeddings_cancer-pten_trial4_2024v2-model_ablated-tmrm_eps162_r20251220"
    # data_dir = "/runs/20260109_pten-t4_resnet_embeddings_2024v2-model_tscrambled_modified_labels"

    is_mitospace = True
    pick_datasets = [
        "20251215-1", "20251215-2", "20251215-3",  # Normal
        "20251216-1", "20251216-2", "20251216-3",  # Normal
        # "20251217-1", "20251217-2", "20251217-3", # Cisplatin
        # "20251218-1", "20251218-2", "20251218-3", # Cisplatin
    ]
    # pick_datasets = None

    accuracies = []
    for iteration in trange(n_iterations):
        # embeddings_path = osp.join(data_dir, "embeddings_raw.npy" if is_mitospace else "embeddings.npy")
        embeddings_path = osp.join(data_dir, "embeddings_resnet.npy" if is_mitospace else "embeddings.npy")
        labels_path = osp.join(data_dir, "labels.npy")
        features = np.load(embeddings_path)
        labels = np.load(labels_path)

        # Keep only the last frame
        features = features[:, -1, :]

        print(features.shape)

        # concatenate along dim 1
        # features = features.reshape(features.shape[0], -1)

        if not is_mitospace:
            # standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(features)
            # Normalize the along the feature axis (N, D) -> (N, D)
            X_normalized = normalize(X_scaled, norm='max', axis=1)
            features = X_normalized

        # Optional filtering by dataset substrings
        if is_mitospace:
            df_img_paths = pd.read_csv(f"{data_dir}/image_paths.csv", header=None)
            img_paths = df_img_paths[0].tolist()
            img_paths = np.array(img_paths)

            if pick_datasets is not None:
                keep_idxs = [i for i, path in enumerate(img_paths) if any(ds in path for ds in pick_datasets)]
                keep_idxs = np.array(keep_idxs, dtype=int)
                features = features[keep_idxs]
                labels = labels[keep_idxs]
                img_paths = img_paths[keep_idxs]

            # #-- EXPERIMENT
            # if "single_frames" in data_dir:
            #     img_paths = np.array(img_paths)
            #
            #     print(f"image paths: {img_paths.shape[0]}")
            #     print(f"features: {features.shape[0]}")
            #     print(f"labels: {labels.shape[0]}")
            #
            #     # keep only every 10th frame (the last frame of each sequence, e.g. frame 10, 20, 30, ...)
            #     img_paths = img_paths[9::10]
            #     features = features[9::10]
            #     labels = labels[9::10]
            #     print(f"new image paths: {img_paths.shape[0]}")
            #     print(f"new features: {features.shape[0]}")
            #     print(f"new labels: {labels.shape[0]}")
            #
            # if "10frames" in data_dir:
            #     # Get only the last frame of each sequence for classification
            #     for i in range(features.shape[0]):
            #         features[i] = features[i, -1]
            #
            #     features = features[:, 0, :]  # drop the time dimension (we only have 1 frame embedding now)

        # Map labels to {0,1}
        unique_labels = np.unique(labels)
        assert len(unique_labels) == 2, f"Expected binary labels, got {unique_labels}"
        label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
        labels = np.array([label_map[lbl] for lbl in labels], dtype=int)

        # ----- Ensure global class balance (and honor n_samples if set) -----
        idx0 = np.where(labels == 0)[0]
        idx1 = np.where(labels == 1)[0]

        if n_samples is not None:
            n_each = min(len(idx0), len(idx1), n_samples // 2)
            idx0_bal = np.random.choice(idx0, n_each, replace=False)
            idx1_bal = np.random.choice(idx1, n_each, replace=False)
            sel = np.concatenate([idx0_bal, idx1_bal])
            np.random.shuffle(sel)
            features = features[sel]
            labels = labels[sel]
        else:
            if len(idx0) != len(idx1):
                n_each = min(len(idx0), len(idx1))
                idx0_bal = np.random.choice(idx0, n_each, replace=False)
                idx1_bal = np.random.choice(idx1, n_each, replace=False)
                sel = np.concatenate([idx0_bal, idx1_bal])
                np.random.shuffle(sel)
                features = features[sel]
                labels = labels[sel]

        print(features.shape)
        print(labels.shape)

        # Print the number of samples per class
        # print("Samples per class:", {k: (labels == k).sum() for k in np.unique(labels)})

        # # Normalize the feature dimensions
        # features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        #
        # # Normalize to 0-1 range
        # feat_min = features.min(axis=0)
        # feat_max = features.max(axis=0)
        # features = (features - feat_min) / (feat_max - feat_min + 1e-8)

        # Print the min/max values for the features
        # print("Feature min/max:", features.min(), features.max())

        # Stratified per-label split with enforced balance
        train_features_all, train_labels_all, test_features, test_labels = split_dataset(
            features, labels, split_perc=split_frac, per_label=True, seed=1123
        )

        # Further split Training into 90/10 Train/Validation
        train_features, val_features, train_labels, val_labels = train_test_split(
            train_features_all, train_labels_all, train_size=0.9,
            random_state=1123, shuffle=True, stratify=train_labels_all
        )

        # Print the number of samples per class in train and test sets
        print("Train samples per class:", {k: (train_labels == k).sum() for k in np.unique(train_labels)})
        print("Val samples per class:", {k: (val_labels == k).sum() for k in np.unique(val_labels)})
        print("Test samples per class:", {k: (test_labels == k).sum() for k in np.unique(test_labels)})

        # Build datasets/loaders from the split (no extra random_split)
        train_dataset = TensorDataset(
            torch.tensor(train_features, dtype=torch.float32),
            torch.tensor(train_labels, dtype=torch.float32),
        )
        val_dataset = TensorDataset(
            torch.tensor(val_features, dtype=torch.float32),
            torch.tensor(val_labels, dtype=torch.float32),
        )
        test_dataset = TensorDataset(
            torch.tensor(test_features, dtype=torch.float32),
            torch.tensor(test_labels, dtype=torch.float32),
        )

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Train/evaluate
        if probe_type == "nonlinear":
            model = SimpleNNClassifier(input_dim=features.shape[1], hidden_dim=hidden_dim)
        else:
            model = LinearBinaryClassifier(input_dim=features.shape[1])

        # Pass val_loader to train_model
        model = train_model(model, train_loader, val_loader=val_loader, num_epochs=num_epochs, learning_rate=0.001,
                            device="cuda")

        # Evaluation uses the best checkpoint loaded by train_model
        accuracy = evaluate_model(model, test_loader, device="cuda")
        accuracies.append(accuracy)

    accuracies = np.array(accuracies)
    print(f"Mean accuracy over {n_iterations} iterations: {accuracies.mean():.4f} ± {accuracies.std():.4f}")
    print("All accuracies:", accuracies.tolist())