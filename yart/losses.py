"""
YART: Your Another Reranker Trainer
Loss function implementations for cross-encoder training.
"""

from typing import Dict, Literal, Type

import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginRankingLoss(nn.Module):
    """
    Implementation of a margin ranking loss for multiple negatives.
    """

    def __init__(self, margin: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scores: Tensor of shape (batch_size, num_samples)
            labels: Tensor of shape (batch_size, num_samples)

        Returns:
            Loss value
        """
        batch_size, num_samples = scores.shape

        # Identify positive and negative scores
        pos_mask = labels > 0.5
        neg_mask = ~pos_mask

        # Ensure we have positives and negatives
        if not torch.any(pos_mask) or not torch.any(neg_mask):
            return torch.tensor(0.0, device=scores.device)

        # Get positive scores (usually the first one in each row)
        pos_scores = scores.masked_select(pos_mask).view(batch_size, -1)
        # Get negative scores
        neg_scores = scores.masked_select(neg_mask).view(batch_size, -1)

        # Compute pairwise margins for each positive-negative pair
        pos_scores_expanded = pos_scores.unsqueeze(2)  # (batch_size, pos_count, 1)
        neg_scores_expanded = neg_scores.unsqueeze(1)  # (batch_size, 1, neg_count)

        # Calculate loss with margin: max(0, margin - (pos - neg))
        loss = F.relu(self.margin - (pos_scores_expanded - neg_scores_expanded))

        # Reduce as specified
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class MultipleNegativesRankingLoss(nn.Module):
    """
    Multiple negatives ranking loss where for each query, the first sample is positive
    and the rest are negatives.
    """

    def __init__(self, margin: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scores: Tensor of shape (batch_size, num_samples)
            labels: Tensor of shape (batch_size, num_samples), where first column
                   is expected to be the positive sample

        Returns:
            Loss value
        """
        batch_size, num_samples = scores.shape

        # Get positive scores (first column by convention)
        positive_scores = scores[:, 0]

        # All other scores are negative
        negative_scores = scores[:, 1:]

        # Calculate loss: max(0, margin - (pos - neg))
        losses = F.relu(self.margin - (positive_scores.unsqueeze(1) - negative_scores))

        # Reduce as specified
        if self.reduction == "mean":
            return losses.mean()
        elif self.reduction == "sum":
            return losses.sum()
        else:
            return losses


class SoftCrossEntropyLoss(nn.Module):
    """
    Soft cross entropy loss that takes soft labels.
    """

    def __init__(self, temperature: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scores: Tensor of shape (batch_size, num_samples)
            labels: Tensor of shape (batch_size, num_samples), containing soft labels

        Returns:
            Loss value
        """
        # Scale logits by temperature
        logits = scores / self.temperature

        # Compute log softmax and softmax of labels
        log_probs = F.log_softmax(logits, dim=1)
        target_probs = F.softmax(labels / self.temperature, dim=1)

        # Compute cross entropy
        loss = -(target_probs * log_probs).sum(dim=1)

        # Reduce as specified
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


# Register available loss functions
loss_registry: Dict[str, Type[nn.Module]] = {
    "cross_entropy": nn.CrossEntropyLoss,
    "mse": nn.MSELoss,
    "margin_ranking_loss": MarginRankingLoss,
    "multiple_negatives_ranking_loss": MultipleNegativesRankingLoss,
    "bce_with_logits": nn.BCEWithLogitsLoss,
    "soft_cross_entropy": SoftCrossEntropyLoss,
}


def get_loss_fn(loss_name: str, **kwargs) -> nn.Module:
    """
    Factory function to get a loss function by name.

    Args:
        loss_name: Name of the loss function
        **kwargs: Additional arguments to pass to the loss function constructor

    Returns:
        Loss function instance
    """
    if loss_name not in loss_registry:
        raise ValueError(
            f"Loss function {loss_name} not found. Available options: {list(loss_registry.keys())}"
        )

    loss_cls = loss_registry[loss_name]
    return loss_cls(**kwargs)
