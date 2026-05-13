import colorsys
import os
import os.path as osp
import random
from random import shuffle
from typing import List, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from scipy.cluster.hierarchy import fcluster, linkage
from skimage.restoration import denoise_tv_bregman
from sklearn.preprocessing import MinMaxScaler

def get_drug_label_maps(fpath):
    """
    Returns drug label maps.
    :param fpath: Path to the file containing drug-label mappings.
    :return: (drug_labels_dict, label_drug_dict)
    """
    drug_labels_dict = {}
    label_drug_dict = {}
    with open(fpath, "r") as f:
        for line in f:
            folder, drug, label = line.split()
            drug_labels_dict[drug] = int(label)
            label_drug_dict[int(label)] = drug

    return drug_labels_dict, label_drug_dict


def load_config(config_path):
    with open(config_path, "r") as file:
        try:
            cfg = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    return cfg


def topKfrequent(nums, weights, k, weighted=False):
    d = dict()

    for i, n in enumerate(nums):
        if weighted:
            d[n] = d.setdefault(n, 0) + weights[i]
        else:
            d[n] = d.setdefault(n, 0) + 1

    sortedNumsKeys = sorted(d.keys(), key=lambda x: d[x], reverse=True)
    return sortedNumsKeys[:k]


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

