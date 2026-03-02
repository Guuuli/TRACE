import numpy as np
from torch.nn.modules.loss import _Loss
from torch import Tensor
import torch.nn.functional as F
import torch
from torcheval.metrics.functional.ranking import retrieval_precision, reciprocal_rank
from torchmetrics.functional.retrieval.recall import retrieval_recall

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 *
                (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def SMAPE(pred, true):
    return np.mean(200 * np.abs(pred - true) / (np.abs(pred) + np.abs(true) + 1e-8))

def forecast_metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    smape = SMAPE(pred, true)
    return {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape, "mspe": mspe, "smape": smape}

def compute_accuracy(logits, labels):
    """
    Compute accuracy given logits and ground truth labels.

    Args:
        logits (torch.Tensor): shape [B, n_class], raw output from the model
        labels (torch.Tensor): shape [B], ground truth labels as class indices

    Returns:
        float: accuracy value between 0 and 1
    """
    preds = torch.argmax(logits, dim=1)  # shape: [B]
    valid_mask = labels != -100          # shape: [B]
    if valid_mask.sum() == 0:
        return 0.0  # No valid labels, avoid division by zero

    correct = (preds[valid_mask] == labels[valid_mask]).sum().item()
    total = valid_mask.sum().item()

    return correct / total

def compute_accuracy_stats(logits, labels):
    preds = torch.argmax(logits, dim=1)
    valid_mask = labels != -100
    correct = (preds[valid_mask] == labels[valid_mask]).sum()
    total = valid_mask.sum()
    return correct, total

class sMAPELoss(_Loss):
    __constants__ = ["reduction"]

    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super().__init__(size_average, reduce, reduction)

    def _abs(self, input):
        return F.l1_loss(input, torch.zeros_like(input), reduction="none")

    def _divide_no_nan(self, a: float, b: float) -> float:
        """
        Auxiliary funtion to handle divide by 0
        """
        div = a / b
        div[div != div] = 0.0
        div[div == float("inf")] = 0.0
        return div

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        delta_y = self._abs(input - target)
        scale = self._abs(target) + self._abs(input)
        error = self._divide_no_nan(delta_y, scale)
        error = 200 * torch.nanmean(error)

        return error

def compute_classification_loss(cls_logits, class_labels):
    '''
    cls_logits: [B, n_classes]
    class_labels: [B]
    '''
    # Classification loss only on labeled samples
    labeled_mask =(class_labels != -100)  # shape [B]
    if labeled_mask.any():
        cls_loss = F.cross_entropy(
            cls_logits[labeled_mask], 
            class_labels[labeled_mask]
        )
    else:
        cls_loss = torch.tensor(0.0, device=cls_logits.device)
    return cls_loss


def reciprocal_rank_tensor(similarity, gt):
    ranking = similarity.argsort(dim=1, descending=True)  # [N, N]

    match = (ranking == gt.unsqueeze(1)) 

    ranks = match.float().argmax(dim=1) + 1  
    return (1.0 / ranks).mean()


def retrieval_precision_tensor(similarity, gt, k):
    topk = similarity.topk(k, dim=1).indices  # [N, k]
    match = (topk == gt.unsqueeze(1))  # [N, k]
    return match.any(dim=1).float().mean()


def retrieval_recall_tensor(similarity, gt, k):
    return retrieval_precision_tensor(similarity, gt, k)



def compute_precision_at_k(indices, candidate_labels, query_labels, k):
    retrieved_labels_topk = candidate_labels[indices[:, :k]]  # [N, k]
    matches = (retrieved_labels_topk == query_labels.unsqueeze(1))  # [N, k]
    precision = matches.sum(dim=1).float() / k  # 每个样本的 precision
    return precision.mean()  # scalar

# MRR
def compute_mrr(indices, candidate_labels, query_labels):
    reciprocal_ranks = []
    for i in range(indices.size(0)):
        retrieved_labels = candidate_labels[indices[i]]  # [topk]
        correct_ranks = (retrieved_labels == query_labels[i]).nonzero(as_tuple=True)
        if correct_ranks[0].numel() > 0:
            rank = correct_ranks[0][0].item() + 1  # rank starts at 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
    return torch.tensor(reciprocal_ranks).mean()