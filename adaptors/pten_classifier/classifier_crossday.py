import os.path as osp
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import copy
from sklearn.model_selection import train_test_split
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
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
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


def get_dataset_features(datasets, features, labels, img_paths):
    keep_idxs = [i for i, path in enumerate(img_paths) if any(ds in path for ds in datasets)]
    keep_idxs = np.array(keep_idxs, dtype=int)
    features = features[keep_idxs]
    labels = labels[keep_idxs]
    img_paths = img_paths[keep_idxs]
    return features, labels, img_paths


def balance_classes(features, labels, n_samples=None):
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
    return features, labels


if __name__ == "__main__":
    from torch.utils.data import TensorDataset, DataLoader

    np.random.seed(1123)

    n_iterations = 10
    num_epochs = 50
    hidden_dim = 256
    learning_rate = 0.001
    probe_type = "nonlinear"  # "linear" or "nonlinear"
    show_loss = False

    n_samples = None

    # data_dir = "/home/earkfeld/Desktop/cell_paint_feats/a549"
    # data_dir = "/home/earkfeld/Desktop/cell_paint_feats/hela"
    # data_dir = "/mnt/DATA_01/Eric/mitospace4d_data/runs/embeddings_cancer_r20250929_10frames_modified_labels_for_eval"
    # data_dir = "/mnt/DATA_01/Eric/mitospace4d_data/runs/embeddings_cancer_r20251002_single_frames_modified_labels_for_eval"
    # data_dir = "/home/earkfeld/Projects/MitoSpace4D/runs/exp0_modified_embeddings_cancer-pten_trial4_2024v2-model_ablated-tmrm_eps162_r20251220"
    # data_dir = "/runs/20260109_pten-t4_resnet_embeddings_2024v2-model_tscrambled_modified_labels"

    # data_dir = "/home/earkfeld/Projects/MitoSpace4D/runs/20260109_pten-t4_resnet_embeddings_2024v2-model_modified_labels"

    data_dir = "/home/earkfeld/Projects/MitoSpace4D/runs/20260111_pten_embeddings_resnet3d-kinetics-300eps_ablated-tmrm_modified_labels"
    # data_dir = "/home/earkfeld/Projects/MitoSpace4D/adaptors/pten_classifier/deepprofiler_features/PTEN_deepprofiler_pooled-clones"

    is_mitospace = "deepprofiler" not in data_dir

    train_datasets = ['20251215']
    test_datasets = ['20251216']
    pick_datasets = train_datasets + test_datasets

    embeddings_path = osp.join(data_dir, "embeddings_raw.npy")
    labels_path = osp.join(data_dir, "labels.npy")
    raw_features = np.load(embeddings_path)
    raw_labels = np.load(labels_path)

    if is_mitospace:
        # raw_features = raw_features[:, -1, :]
        # raw_features = raw_features.reshape(raw_features.shape[0], -1)
        raw_features = raw_features.mean(axis=1)
        # raw_features = raw_features.max(axis=1)

        # idx = np.argmax(np.abs(raw_features), axis=1, keepdims=True)
        # max_mag_features = np.take_along_axis(raw_features, idx, axis=1)
        # raw_features = max_mag_features.squeeze(axis=1)

    accuracies = []

    # Set up progress bar with
    pbar = trange(n_iterations, desc="Acc: XX.X%")
    for i, iteration in enumerate(pbar):
        features = raw_features.copy()
        labels = raw_labels.copy()

        df_img_paths = pd.read_csv(f"{data_dir}/image_paths.csv", header=None)
        img_paths = df_img_paths[0].tolist()
        img_paths = np.array(img_paths)

        if pick_datasets is not None:
            keep_idxs = [i for i, path in enumerate(img_paths) if any(ds in path for ds in pick_datasets)]
            keep_idxs = np.array(keep_idxs, dtype=int)
            features = features[keep_idxs]
            labels = labels[keep_idxs]
            img_paths = img_paths[keep_idxs]

        # Map labels to {0,1}
        unique_labels = np.unique(labels)
        assert len(unique_labels) == 2, f"Expected binary labels, got {unique_labels}"
        label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
        labels = np.array([label_map[lbl] for lbl in labels], dtype=int)

        # L2 normalization
        features = features / np.linalg.norm(features, axis=1, keepdims=True)

        # Separate the train dataset using the dates and image paths
        train_idxs = np.array([i for i, path in enumerate(img_paths) if any(ds in path for ds in train_datasets)],
                              dtype=int)
        test_idxs = np.array([i for i, path in enumerate(img_paths) if any(ds in path for ds in test_datasets)],
                             dtype=int)

        train_features_all = features[train_idxs]
        train_labels_all = labels[train_idxs]
        test_features = features[test_idxs]
        test_labels = labels[test_idxs]

        # Get the min number of samples
        # min_n_samples = min(min(np.bincount(train_labels_all)), min(np.bincount(test_labels)))
        train_features_all, train_labels_all = balance_classes(train_features_all, train_labels_all)
        test_features, test_labels = balance_classes(test_features, test_labels)

        # Print the number of samples per class in train, val and test sets
        if i == 0:
            print("\nTrain samples per class:", {k: (train_labels_all == k).sum() for k in np.unique(train_labels_all)})
            print("Test/Val samples per class:", {k: (test_labels == k).sum() for k in np.unique(test_labels)})

        # Build datasets/loaders from the split (no extra random_split)
        train_dataset = TensorDataset(
            torch.tensor(train_features_all, dtype=torch.float32),
            torch.tensor(train_labels_all, dtype=torch.float32),
        )
        test_dataset = TensorDataset(
            torch.tensor(test_features, dtype=torch.float32),
            torch.tensor(test_labels, dtype=torch.float32),
        )

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Train/evaluate
        if probe_type == "nonlinear":
            model = SimpleNNClassifier(input_dim=features.shape[1], hidden_dim=hidden_dim)
        else:
            model = LinearBinaryClassifier(input_dim=features.shape[1])

        # Pass test_loader as val_loader to train_model
        model = train_model(model, train_loader, val_loader=test_loader, num_epochs=num_epochs, learning_rate=learning_rate,
                            device="cuda", show_loss=show_loss)

        # Evaluation uses the best checkpoint
        accuracy = evaluate_model(model, test_loader, device="cuda")
        accuracies.append(accuracy)

        # Set the progress bar description
        pbar.set_description(f"Acc: {np.mean(accuracies)*100:.1f}%")

    accuracies = np.array(accuracies)
    print(f"Mean accuracy over {n_iterations} iterations: {accuracies.mean()*100:.1f}% ± {accuracies.std()*100:.1f}%")
    print("All accuracies:", accuracies.tolist())