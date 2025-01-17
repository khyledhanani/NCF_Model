import torch
import torch.nn as nn
from typing import List

class NCF(nn.Module):
    def __init__(
        self, 
        num_users: int, 
        num_items: int, 
        embedding_dim: int = 8, 
        layers: List[int] = [64, 32, 16, 8]
    ):
        super(NCF, self).__init__()
        
        self.user_gmf_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_gmf_embedding = nn.Embedding(num_items, embedding_dim)
        self.user_mlp_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_mlp_embedding = nn.Embedding(num_items, embedding_dim)
        
        # batch norm implemented w dropout
        self.mlp_layers = nn.ModuleList()
        input_dim = 2 * embedding_dim
        for output_dim in layers:
            self.mlp_layers.append(nn.Linear(input_dim, output_dim))
            self.mlp_layers.append(nn.BatchNorm1d(output_dim))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.Dropout(p=0.2))
            input_dim = output_dim
        
        self.mlp = nn.Sequential(*self.mlp_layers)
        
        self.prediction = nn.Linear(layers[-1] + embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
    
    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        user_gmf_embed = self.user_gmf_embedding(user_indices)
        item_gmf_embed = self.item_gmf_embedding(item_indices)
        gmf_output = user_gmf_embed * item_gmf_embed
        
        user_mlp_embed = self.user_mlp_embedding(user_indices)
        item_mlp_embed = self.item_mlp_embedding(item_indices)
        mlp_input = torch.cat([user_mlp_embed, item_mlp_embed], dim=-1)
        mlp_output = self.mlp(mlp_input)
        
        concat = torch.cat([gmf_output, mlp_output], dim=-1)
        
        prediction = self.sigmoid(self.prediction(concat))
        return prediction.squeeze() 