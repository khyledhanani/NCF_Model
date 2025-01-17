import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, SubsetRandomSampler
from typing import Union, Dict, List
import logging
import os
import numpy as np
from metrics import compute_metrics

def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str,
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for user_indices, item_indices, labels in val_loader:
            user_indices = user_indices.to(device)
            item_indices = item_indices.to(device)
            labels = labels.float().to(device)
            
            predictions = model(user_indices, item_indices)
            loss = criterion(predictions, labels)
            
            total_loss += loss.item()
            all_predictions.append(predictions)
            all_labels.append(labels)
    
    predictions = torch.cat(all_predictions)
    labels = torch.cat(all_labels)
    
    metrics = compute_metrics(predictions, labels, k_values)
    metrics['loss'] = total_loss / len(val_loader)
    
    return metrics

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    for batch_idx, (user_indices, item_indices, labels) in enumerate(train_loader):
        user_indices = user_indices.to(device)
        item_indices = item_indices.to(device)
        labels = labels.float().to(device)
        
        optimizer.zero_grad()
        predictions = model(user_indices, item_indices)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_predictions.append(predictions.detach())
        all_labels.append(labels)
    
    predictions = torch.cat(all_predictions)
    labels = torch.cat(all_labels)
    metrics = compute_metrics(predictions, labels, k_values)
    metrics['loss'] = total_loss / len(train_loader)
    
    return metrics

def train_ncf(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    learning_rate: float = 0.001,
    device: str = 'cuda',
    logger: Union[logging.Logger, None] = None,
    checkpoint_dir: str = 'checkpoints',
    patience: int = 3,
    k_values: List[int] = [1, 5, 10],
    distributed: bool = False,
    local_rank: int = -1
):
    if distributed:
        if local_rank != -1:
            device = torch.device(f'cuda:{local_rank}')
            dist.init_process_group(backend='nccl')
            model = model.to(device)
            model = DDP(model, device_ids=[local_rank])
    else:
        model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    
    for epoch in range(epochs):

        # training pass
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, k_values)
        
        # validation pass
        val_metrics = evaluate(model, val_loader, criterion, device, k_values)
        
        # learning rate scheduler
        scheduler.step(val_metrics['loss'])
        
        if logger:
            metrics_str = f'Epoch {epoch+1}/{epochs}\n'
            metrics_str += 'Train Metrics:\n'
            for metric, value in train_metrics.items():
                metrics_str += f'  {metric}: {value:.4f}\n'
            metrics_str += 'Validation Metrics:\n'
            for metric, value in val_metrics.items():
                metrics_str += f'  {metric}: {value:.4f}\n'
            logger.info(metrics_str)
        
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            if not distributed or local_rank == 0:
                torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if logger:
                    logger.info(f'Early stopping triggered after {epoch+1} epochs')
                break
        
        if (epoch + 1) % 5 == 0 and (not distributed or local_rank == 0):
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }, checkpoint_path)
    
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    return model

def cross_validate(
    model_class: type,
    dataset: torch.utils.data.Dataset,
    n_splits: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    device: str,
    logger: Union[logging.Logger, None] = None,
    **model_kwargs
) -> Dict[str, List[float]]:
    """
    Perform k-fold cross-validation.
    """
    dataset_size = len(dataset)
    indices = np.arange(dataset_size)
    fold_sizes = np.full(n_splits, dataset_size // n_splits, dtype=int)
    fold_sizes[:dataset_size % n_splits] += 1
    current = 0
    results = []
    
    for fold in range(n_splits):
        start, stop = current, current + fold_sizes[fold]
        val_indices = indices[start:stop]
        train_indices = np.concatenate([indices[:start], indices[stop:]])
        current = stop
        
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        model = model_class(**model_kwargs)
        model = train_ncf(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            device=device,
            logger=logger
        )
        
        final_metrics = evaluate(model, val_loader, nn.BCELoss(), device)
        results.append(final_metrics)
        
        if logger:
            logger.info(f'Fold {fold + 1} Results:')
            for metric, value in final_metrics.items():
                logger.info(f'  {metric}: {value:.4f}')
    
    aggregated_results = {}
    for metric in results[0].keys():
        values = [r[metric] for r in results]
        aggregated_results[metric] = {
            'mean': np.mean(values),
            'std': np.std(values)
        }
    
    return aggregated_results 