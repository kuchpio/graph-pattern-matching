from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.utils import to_dense_batch
import torch.nn as nn
import torch.nn.functional as F

class GraphMatchingModel(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.conv1 = GINConv(nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU(), nn.Dropout(0.5)))
        self.conv2 = GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.5)))
        self.fc_g1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_g2 = nn.Linear(hidden_dim, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
        self.out = nn.Linear(hidden_dim, 2)

    def forward(self, g1, g2):
        h1 = self.conv1(g1.x, g1.edge_index)
        h1 = self.conv2(h1, g1.edge_index)
        h1 = global_mean_pool(h1, g1.batch)
        h1 = self.fc_g1(h1)

        h2 = self.conv1(g2.x, g2.edge_index)
        h2 = self.conv2(h2, g2.edge_index)
        h2 = global_mean_pool(h2, g2.batch)
        h2 = self.fc_g2(h2)

        h1 = h1.unsqueeze(0)
        h2 = h2.unsqueeze(0) 

        attn_output, _ = self.cross_attention(h1, h2, h2)
        attn_output = attn_output.squeeze(0)
        attn_output = F.relu(attn_output)

        logits = self.out(attn_output)
        return logits