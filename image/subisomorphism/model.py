import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool

class SubgraphGINModel(torch.nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.conv1 = GINConv(torch.nn.Sequential(torch.nn.Linear(1, hidden_dim), torch.nn.ReLU()))
        self.conv2 = GINConv(torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU()))
        self.matching_layer = torch.nn.Linear(hidden_dim, hidden_dim)  

    def forward(self, g1, g2):
        h1 = self.conv1(g1.x, g1.edge_index)
        h1 = F.relu(h1)
        h1 = self.conv2(h1, g1.edge_index)
        h1 = global_add_pool(h1, g1.batch)  

        h2 = self.conv1(g2.x, g2.edge_index)
        h2 = F.relu(h2)
        h2 = self.conv2(h2, g2.edge_index)
        h2 = global_add_pool(h2, g2.batch) 

        matching = torch.abs(h1 - h2) 
        return matching 