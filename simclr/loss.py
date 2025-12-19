import torch
from torch import nn
import torch.nn.functional as F
from utils.utils import accuracy
from torch import Tensor
from typing import Tuple
import torch.distributed as dist
import diffdist


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07, use_normalization: bool = True,
                 n_views: int = 2, distributed: bool = True) -> None:
        """
        Initialize the loss function for InfoNCE loss.
        """
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.normalize = use_normalization
        self.cross_entropy = nn.CrossEntropyLoss()
        self.n_views = n_views
        self.distributed = distributed

    def forward(self, features: Tensor, bs: int) -> Tuple[Tensor, Tuple[float, float]]:
        """
        Compute loss for model and accuracy (top-1 and top-5) for the batch of features.
        The accuracy are based on the logits produced and the labels from the batch.

        Note: The loss only works for n_views = 2.

        returns: Tuple of ( loss and Tuple (top-1 accuracy, top-5 accuracy) )
        """

        if self.normalize:
            features = F.normalize(features, dim=1).to(features.dtype)

        if self.distributed:
            world_size = dist.get_world_size()
            gathered_features = [torch.zeros_like(features) for _ in range(world_size)]
            gathered_features = diffdist.functional.all_gather(gathered_features, features)

            gathered_features = torch.cat(gathered_features, dim=0)

            features_sorted = gathered_features.view(world_size, self.n_views, bs, -1).permute(1, 0, 2, 3).reshape(-1,
                                                                                                                   gathered_features.size(
                                                                                                                       -1))
            features = features_sorted

        bs = features.shape[0] // self.n_views

        labels = torch.arange(bs).repeat(self.n_views).to(features.device)

        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool, device=features.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        logits = logits / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

        loss = self.cross_entropy(logits, labels)

        # acc = accuracy(logits, labels, topk=(1, 5))
        acc = accuracy(logits, labels, topk=(1,3))

        return loss, (acc[0].item(), acc[1].item())


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature: float = 0.07, contrast_mode: str = 'all',
                 base_temperature: float = 0.07, use_normalization: bool = True) -> None:
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.normalize = use_normalization

    def forward(self, features: Tensor, bs: int, n_views: int, labels: Tensor = None, mask: Tensor = None) -> Tuple[
        Tensor, Tuple[float, float]]:
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Note: The accuracies computed doesn't make a lot of sense and is just a proxy of the model's performance.
        Do not rely on it for any meaningful metric. It's computed just so that InfoNCE loss and SupConLoss
         both have same return type.

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            bs: batch size.
            n_views: number of views.
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            Tuple of ( loss and Tuple (top-1 accuracy, top-5 accuracy) )
        """

        if self.normalize:
            features = F.normalize(features, dim=-1)

        features = torch.stack(torch.split(features, [bs for _ in range(n_views)]), dim=1)

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute a proxy for accuracy
        acc_logits = torch.div(torch.matmul(features[:, 0].clone(), features[:, 0].clone().T), self.temperature)
        pred_sorted_sim = torch.argsort(acc_logits, dim=1, descending=True)
        gt_labels_sorted = mask.clone().argsort(dim=1, descending=True)
        acc = torch.sum(torch.eq(pred_sorted_sim, gt_labels_sorted)) / (batch_size * batch_size)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point.
        # Edge case e.g.:-
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan]
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss, (acc * 100, acc * 100)
