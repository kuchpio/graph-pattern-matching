import sys
import torch
from torch_geometric.data import Data, Batch
import torch.nn.functional as F
from model import GraphMatchingModel 
import pytorch_lightning as pl

class GraphIsomorphismModel(pl.LightningModule):
    def __init__(self, hidden_dim=32, lr=1e-3):
        super().__init__()
        self.model = GraphMatchingModel(hidden_dim)
        self.lr = lr

    def forward(self, g1, g2):
        return self.model(g1, g2)


if __name__ == "__main__":
    checkpoint_path = "checkpoints/gin-epochepoch=2.ckpt"

    if len(sys.argv) < 3:
        print("Za mało argumentów.")
        print("Oczekiwane: classify.py <liczba_krawedzi_g1> <krawedzie_g1> <liczba_krawedzi_g2> <krawedzie_g2>")
        sys.exit(1)

    N1 = int(sys.argv[1])
    edges_g1 = sys.argv[2:2 + 2 * N1]
    edges_g1 = list(map(int, edges_g1))

    N2 = int(sys.argv[2 + 2 * N1])
    edges_g2 = sys.argv[2 + 2 * N1 + 1:2 + 2 * N1 + 1 + 2 * N2]
    edges_g2 = list(map(int, edges_g2))

    edge_index_g1 = torch.tensor([edges_g1[0::2], edges_g1[1::2]], dtype=torch.long)
    edge_index_g2 = torch.tensor([edges_g2[0::2], edges_g2[1::2]], dtype=torch.long)

    num_nodes_g1 = max(edge_index_g1.max().item(), 0) + 1
    num_nodes_g2 = max(edge_index_g2.max().item(), 0) + 1

    g1 = Data(edge_index=edge_index_g1, x=torch.ones((num_nodes_g1, 1)))
    g2 = Data(edge_index=edge_index_g2, x=torch.ones((num_nodes_g2, 1)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g1 = g1.to(device)
    g2 = g2.to(device)

    g1.batch = torch.zeros(g1.num_nodes, dtype=torch.long, device=device)
    g2.batch = torch.zeros(g2.num_nodes, dtype=torch.long, device=device)

    model = GraphIsomorphismModel.load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()

    with torch.no_grad():
        preds = model(g1, g2)
        predicted_label = preds.argmax(dim=1).item()

    print(predicted_label)
