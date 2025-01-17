import torch
from typing import List

def compute_ndcg(predictions: torch.Tensor, labels: torch.Tensor, k: int = 10) -> float:
    """
    Normalized Discounted Cumulative Gain@K for binary relevance.
    """
    device = predictions.device
    _, indices = torch.sort(predictions, descending=True)
    batch_labels = torch.gather(labels, -1, indices[:, :k])
    discounts = torch.log2(torch.arange(k, dtype=torch.float32, device=device) + 2.0)
    dcg = (batch_labels / discounts).sum(dim=-1)
    sorted_labels, _ = torch.sort(labels, descending=True)
    ideal_labels = sorted_labels[:, :k]
    idcg = (ideal_labels / discounts).sum(dim=-1)
    ndcg = dcg / (idcg + 1e-8)  # epsilon to avoid div by zero
    return ndcg.mean().item()

def compute_precision_at_k(predictions: torch.Tensor, labels: torch.Tensor, k: List[int]) -> List[float]:
    """
    Compute Precision@K for multiple K values.
    """
    max_k = max(k)
    
    _, indices = torch.sort(predictions, descending=True)
    batch_labels = torch.gather(labels, -1, indices[:, :max_k])
    precisions = []
    for ki in k:
        precision_at_ki = batch_labels[:, :ki].mean().item()
        precisions.append(precision_at_ki)
    
    return precisions

def compute_metrics(predictions: torch.Tensor, labels: torch.Tensor, k_values: List[int] = [1, 5, 10]) -> dict:
    metrics = {}
    for k in k_values:
        metrics[f'ndcg@{k}'] = compute_ndcg(predictions, labels, k)
    precisions = compute_precision_at_k(predictions, labels, k_values)
    for k, precision in zip(k_values, precisions):
        metrics[f'precision@{k}'] = precision
    
    return metrics 