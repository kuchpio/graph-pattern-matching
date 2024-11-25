import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Data
import pytorch_lightning as pl
import random

import torch
from torch_geometric.data import Data
import random

def generate_subisomorphic_graph_pairs(num_pairs=1000, min_nodes=5, max_nodes=20, min_subgraph_size=3, max_subgraph_size=18):
    pairs = []
    labels = []
    
    for _ in range(num_pairs):
        # Losowanie liczby wierzchołków dla G2
        num_nodes = random.randint(min_nodes, max_nodes)  # Liczba wierzchołków w G2
        subgraph_size = random.randint(min_subgraph_size, min(num_nodes, max_subgraph_size))  # Rozmiar podgrafu
        
        # Generowanie grafu G2
        g2 = torch.randint(0, 2, (num_nodes, num_nodes))
        g2 = (g2 + g2.T) % 2  # Upewnienie się, że graf jest nieskierowany
        g2.fill_diagonal_(0)  # Brak pętli własnych
        edge_index_2 = (g2.nonzero(as_tuple=False)).t()

        # Generowanie G1 (subizomorficzne lub nie)
        if random.random() > 0.5:  # G1 jest podgrafem G2
            sub_nodes = torch.randperm(num_nodes)[:subgraph_size]  # Wybór podgrafu
            g1 = g2[sub_nodes][:, sub_nodes]  # Wydzielenie podgrafu z G2
            edge_index_1 = (g1.nonzero(as_tuple=False)).t()
            labels.append(1)  # Etykieta 1 dla subizomorfizmu
        else:  # G1 nie jest podgrafem G2
            g1 = torch.randint(0, 2, (subgraph_size, subgraph_size))
            g1 = (g1 + g1.T) % 2  # Upewnienie się, że graf jest nieskierowany
            g1.fill_diagonal_(0)  # Brak pętli własnych
            edge_index_1 = (g1.nonzero(as_tuple=False)).t()
            labels.append(0)  # Etykieta 0 dla grafu nieizomorficznego

        pairs.append((
            Data(edge_index=edge_index_1, x=torch.ones((subgraph_size, 1))),
            Data(edge_index=edge_index_2, x=torch.ones((num_nodes, 1)))
        ))

    return pairs, torch.tensor(labels, dtype=torch.long)


class GraphDataModule(pl.LightningDataModule):
    def __init__(self, train_pairs, train_labels, test_pairs, test_labels, batch_size=32):
        super().__init__()
        self.train_pairs = train_pairs
        self.train_labels = train_labels
        self.test_pairs = test_pairs
        self.test_labels = test_labels
        self.batch_size = batch_size

    def train_dataloader(self):
        dataset = list(zip(self.train_pairs, self.train_labels))
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        dataset = list(zip(self.test_pairs, self.test_labels))
        return DataLoader(dataset, batch_size=self.batch_size)
