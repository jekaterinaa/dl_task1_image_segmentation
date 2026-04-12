import torch
from torch import Tensor
import torch.nn as nn


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


def dice_loss_weighted(input: Tensor, target: Tensor, weights: Tensor = None, multiclass: bool = False):
    """
    input: Predicted probabilities, shape (N, C, H, W).
    target: One-hot encoded ground truth, shape (N, C, H, W).
    weights: Class weights, shape (C,). If None, no weighting is applied.
    multiclass: Whether to compute multiclass Dice loss.
    """
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    dice_per_class = 1 - fn(input, target, reduce_batch_first=True)

    if weights is not None:
        dice_per_class = dice_per_class * weights

    return dice_per_class.mean()


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss on logits.
    weights: tensor of shape (num_classes,) or None
    """
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        n, c, h, w = logits.shape
        logits = logits.permute(0, 2, 3, 1).reshape(-1, c)
        targets = targets.view(-1)

        log_p = torch.nn.functional.log_softmax(logits, dim=1)
        p = torch.exp(log_p)

        log_p_t = log_p[torch.arange(log_p.shape[0]), targets]
        p_t = p[torch.arange(p.shape[0]), targets]

        focal_factor = (1.0 - p_t) ** self.gamma

        loss = -focal_factor * log_p_t

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
        

def final_loss(focal_loss_function, outputs, masks, class_weights, alpha=0.3, beta=0.7):
    """
    outputs: logits, shape (N, 4, H, W)
    masks:   class indices, shape (N, H, W), values 0..3
    alpha, beta: weights for Focal and Dice losses
    """
    focal = focal_loss_function(outputs, masks)

    probs = torch.softmax(outputs, dim=1)
    one_hot = torch.nn.functional.one_hot(
        masks, num_classes=4
    ).permute(0, 3, 1, 2).float()

    d = dice_loss_weighted(probs, one_hot, class_weights, multiclass=True)

    return alpha * focal + beta * d