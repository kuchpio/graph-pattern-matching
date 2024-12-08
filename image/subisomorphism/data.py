import pytorch_lightning as pl
import torch
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import random

def generate_isomorphic_graph_pairs(num_pairs=1000, min_nodes=3, max_nodes=25):
    pairs = []
    labels = []
    for idx in range(num_pairs):
        num_nodes = random.randint(min_nodes, max_nodes)
        g1 = torch.randint(0, 2, (num_nodes, num_nodes))
        g1 = (g1 + g1.T) % 2
        g1.fill_diagonal_(0)
        if idx % 2 == 0:
            perm = torch.randperm(num_nodes)
            g2 = g1[perm][:, perm]
            labels.append(1)
        else:
            g2 = g1.clone()
            num_changes = random.randint(1, max(1, num_nodes // 2))
            for _ in range(num_changes):
                u, v = random.randint(0, num_nodes - 1), random.randint(0, num_nodes - 1)
                if u != v:
                    g2[u, v] = 1 - g2[u, v]
                    g2[v, u] = g2[u, v]
            labels.append(0)
        edge_index_1 = g1.nonzero(as_tuple=False).t()
        edge_index_2 = g2.nonzero(as_tuple=False).t()
        pairs.append((Data(edge_index=edge_index_1, x=torch.ones((num_nodes, 1))),
                      Data(edge_index=edge_index_2, x=torch.ones((num_nodes, 1)))))
    return pairs, torch.tensor(labels, dtype=torch.long)

class PairDataset(Dataset):
    def __init__(self, pairs, labels):
        self.pairs = pairs
        self.labels = labels

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        g1, g2 = self.pairs[idx]
        label = self.labels[idx]
        return g1, g2, label

def pair_collate_fn(batch):
    g1_list = []
    g2_list = []
    labels = []
    for g1, g2, label in batch:
        g1_list.append(g1)
        g2_list.append(g2)
        labels.append(label)
    batch_g1 = Batch.from_data_list(g1_list)
    batch_g2 = Batch.from_data_list(g2_list)
    labels = torch.tensor(labels, dtype=torch.long)
    return batch_g1, batch_g2, labels

class GraphDataModule(pl.LightningDataModule):
    def __init__(self, train_pairs, train_labels, test_pairs, test_labels, batch_size=32):
        super().__init__()
        self.train_dataset = PairDataset(train_pairs, train_labels)
        self.test_dataset = PairDataset(test_pairs, test_labels)
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=pair_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=pair_collate_fn)
