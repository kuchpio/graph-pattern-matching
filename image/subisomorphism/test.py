
from lightning_module import SubgraphIsomorphismModel
import torch
from torch_geometric.data import Data

def test_model(model, test_pairs, test_labels, device):
    model.eval()
    model = model.to(device) 
    correct = 0
    total = len(test_labels)
    
    for i in range(len(test_pairs)):
        g1, g2 = test_pairs[i]
        label = test_labels[i]

        g1 = g1.to(device)
        g2 = g2.to(device)
        label = label.to(device)
        
        g1.batch = torch.zeros(g1.x.size(0), dtype=torch.long, device=device)
        g2.batch = torch.zeros(g2.x.size(0), dtype=torch.long, device=device)
        
        with torch.no_grad():
            preds = model(g1, g2)
            predicted_label = preds.argmax(dim=1).item()
        print(predicted_label)
        if predicted_label == label.item():
            correct += 1
    
    accuracy = correct / total
    print(f"Dokładność na zbiorze testowym: {accuracy:.2f}")

last_checkpoint_path = "checkpoints/gin-epochepoch=20-val_lossval_loss=0.00.ckpt"

trained_model = SubgraphIsomorphismModel.load_from_checkpoint(last_checkpoint_path)

test_graphs = [
    # Subizomorficzne grafy (G1 jest podgrafem G2)
    (
        Data(edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]), x=torch.ones((3, 1))),  # G1 (ścieżka 0-1-2)
        Data(edge_index=torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]]), x=torch.ones((4, 1)))  # G2 (cykl 4-wierzchołkowy)
    ),
    # Nieizomorficzne grafy (G1 nie jest podgrafem G2)
    (
        Data(edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]), x=torch.ones((3, 1))),  # G1 (ścieżka 0-1-2)
        Data(edge_index=torch.tensor([[0, 1, 2, 3, 3, 0], [1, 0, 3, 2, 0, 3]]), x=torch.ones((4, 1)))  # G2 (graf 4-wierzchołkowy bez podgrafu 0-1-2)
    ),
    # Izomorficzne grafy (G1 = G2)
    (
        Data(edge_index=torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]]), x=torch.ones((3, 1))),  # G1 (trójkąt 0-1-2)
        Data(edge_index=torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]]), x=torch.ones((3, 1)))  # G2 (identyczny trójkąt 0-1-2)
    ),
    # Subizomorficzne grafy (G1 jest podgrafem G2 z dodanym wierzchołkiem)
    (
        Data(edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]), x=torch.ones((3, 1))),  # G1 (ścieżka 0-1-2)
        Data(edge_index=torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]), x=torch.ones((4, 1)))  # G2 (ścieżka 0-1-2-3)
    ),
    # Nieizomorficzne grafy (G1 nie jest podgrafem G2, ponieważ struktura jest inna)
    (
        Data(edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]), x=torch.ones((3, 1))),  # G1 (ścieżka 0-1-2)
        Data(edge_index=torch.tensor([[0, 1, 1, 3, 3, 0], [1, 0, 3, 1, 0, 3]]), x=torch.ones((4, 1)))  # G2 (cykl 4-wierzchołkowy z brakiem ścieżki 0-1-2)
    )
]

test_labels = torch.tensor([1, 0, 1, 1, 0], dtype=torch.long)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_labels = test_labels.to(device)

test_model(trained_model, test_graphs, test_labels, device)
