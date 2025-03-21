import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from typing import Tuple, Union


### FAKE DATA GENERATED BY CHATGPT FOR TESTING PURPOSES

def get_data_loader(
    num_users: int,
    num_items: int,
    batch_size: int = 256,
    num_samples: int = 100000,
    val_ratio: float = 0.1,
    return_dataset: bool = False
) -> Union[TensorDataset, Tuple[DataLoader, DataLoader]]:
    users = torch.randint(0, num_users, (num_samples,))
    items = torch.randint(0, num_items, (num_samples,))
    labels = torch.randint(0, 2, (num_samples,)).float()
    
    dataset = TensorDataset(users, items, labels)
    
    if return_dataset:
        return dataset
    
    val_size = int(num_samples * val_ratio)
    train_size = num_samples - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader 