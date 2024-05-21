import numpy as np
import umap
from tqdm import tqdm
from sklearn.neighbors import KDTree
import pandas as pd
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, confusion_matrix, pairwise_distances
import torch.nn.functional as F

from data_aug.mitospace_dataset import *
from simclr.resnet_simclr import *
import argparse
from utils.utils import *
from data_aug.dataset_utils import get_mitospace_data_loaders
import torch
from train_simclr import SimCLRRunner

from utils.vis import plot_cm
from torch.utils.data import DataLoader

global device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from scipy.stats import entropy
from evaluate import *

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description='MitoSpace Evaluation')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument('--config', default='/home/dhruvagarwal/projects/MitoSpace/simclr/config.yaml',
                    type=str, help='Config path.')
parser.add_argument('--evaluate_set', default='test',
                    type=str, help='Set on which to run evaluation')
parser.add_argument('--dist_metric', default='cosine',
                    type=str, help='Metric to use for distance calculation between embeddings')

if __name__ == "__main__":
    args = parser.parse_args()
    cfg = load_config(args.config)
    proj_dir = "/home/dhruvagarwal/projects/MitoSpace/"

    model = ResNetSimCLR(base_model=cfg['model_params']['arch'], out_dim=cfg['model_params']['out_dim'],
                         in_channels=cfg["model_params"]["in_channels"]).to(device)

    checkpoint_path = f"{proj_dir}/runs/lightning_logs/{cfg['experiment_name']}/checkpoints/best_model-v1.ckpt"
    top_ns = cfg["evaluate"]["top_ns"]
    dataset_name = cfg["evaluate"]["dataset"]

    drug_labels_dict = {}
    label_drug_dict = {}
    with open(f"{proj_dir}/extraction_utils/drugs_to_labels.txt", 'r') as f:
        for line in f:
            drug, label = line.split()
            drug_labels_dict[drug] = int(label)
            label_drug_dict[int(label)] = drug

    print(f"Running for {dataset_name} for top {top_ns} accuracies and checkpoint path: {checkpoint_path}")

    model = SimCLRRunner.load_from_checkpoint(
        checkpoint_path, model=model, cfg=cfg
    )

    loaders_reference = get_mitospace_data_loaders(
        '/home/dhruvagarwal/projects/MitoSpace/data/Cal27NewHiroAndre/20240503',
        shuffle=False, batch_size=50, to_load=["all"],
        pick_labels=None)

    loaders_eval = get_mitospace_data_loaders(
        '/home/dhruvagarwal/projects/MitoSpace/data/Cal27NewHiroAndre/20240512',
        shuffle=False, batch_size=50, to_load=["all"],
        pick_labels=None
    )

    train_loader, eval_loader = (loaders_reference["all"], loaders_eval["all"])

    train_embeddings, train_images, train_labels = extract_embeddings_from_model(train_loader, model.model,
                                                                                 normalize_embeddings=True,
                                                                                 get_images=False,
                                                                                 get_labels=True,
                                                                                 messup_tmrm=False,
                                                                                 visualise_model_layer=False)

    eval_embeddings, eval_images, eval_labels = extract_embeddings_from_model(eval_loader, model.model,
                                                                              normalize_embeddings=True,
                                                                              get_images=False, get_labels=True,
                                                                              messup_tmrm=False,
                                                                              visualise_model_layer=False)

    # Evaluation on cosine similarity
    if args.dist_metric == 'cosine':
        dist_matrix_ref, dist_matrix_ref_idxs = cosine_distance(train_embeddings, train_embeddings, weighted=True,
                                                                temperature=cfg["training"]["loss"]["temperature"])
        preds_ref = nearest_neighbor_evaluation(train_labels, train_labels, top_ns, dist_matrix_ref,
                                                dist_matrix_ref_idxs, num_neighbors=[1500])
        cm_ref = plot_cm(train_labels, preds_ref[1], label_drug_dict, verbose=False)  # top 1 confusion matrix of ref set

        sample_size = 100  # number of samples for each drug to construct the cmap

        # group the eval_embeddings by eval_labels
        eval_embeddings_grouped = {}
        for i, label in enumerate(eval_labels):
            if label not in eval_embeddings_grouped:
                eval_embeddings_grouped[label] = [eval_embeddings[i]]
            else:
                eval_embeddings_grouped[label].append(eval_embeddings[i])

        for label in eval_embeddings_grouped:
            eval_embeddings_grouped[label] = np.array(eval_embeddings_grouped[label])
            # shuffle the eval_embeddings
            np.random.shuffle(eval_embeddings_grouped[label])

        drug_acc = {}
        for label in eval_embeddings_grouped:
            if label not in drug_acc:
                drug_acc[label] = [0, 0]  # correct, total

            start_idx = 0
            while start_idx < eval_embeddings_grouped[label].shape[0]:
                eval_embeddings_cur = eval_embeddings_grouped[label][start_idx: start_idx + sample_size]
                dist_matrix_eval, dist_matrix_eval_idxs = cosine_distance(eval_embeddings_cur, train_embeddings,
                                                                          weighted=True,
                                                                          temperature=cfg["training"]["loss"]["temperature"])

                eval_labels_cur = np.array([label] * eval_embeddings_cur.shape[0])
                preds_eval = nearest_neighbor_evaluation(eval_labels_cur, train_labels, top_ns, dist_matrix_eval,
                                                         dist_matrix_eval_idxs, num_neighbors=[1500], verbose=False)
                cm_eval = plot_cm(eval_labels_cur, preds_eval[1], label_drug_dict, verbose=False, make_plot=False)  # top 1 confusion matrix of samples wrt train set
                cm_eval = cm_eval[label]

                # for every row in the cm_eval find the best match with the rows in cm_ref using distributional similarity
                # and then calculate the accuracy

                eps = 1e-8
                best_match = -1
                best_match_score = np.inf

                for j in range(cm_ref.shape[0]):
                    ev, rf = cm_eval, cm_ref[j]
                    if np.sum(ev) == 0 or np.sum(rf) == 0:
                        # these distributions are for drugs that are not present in the ref set; so skip them!
                        continue

                    ev, rf = ev + eps, rf + eps
                    score = entropy(ev, rf)
                    if score < best_match_score:
                        best_match_score = score
                        best_match = j

                if best_match == label:
                    drug_acc[label][0] += 1
                drug_acc[label][1] += 1

                start_idx += sample_size

            print(f"Drug: {label}, Accuracy: {drug_acc[label][0] / drug_acc[label][1]}")

        print("Overall accuracy: ", sum([drug_acc[label][0] for label in drug_acc]) / sum([drug_acc[label][1] for label in drug_acc]))