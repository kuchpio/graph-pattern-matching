import torch
from torch_geometric.data import Data
from lightning_module import GraphIsomorphismModel
from torch_geometric.data import Data, Batch

def test_model(model, test_pairs, test_labels, device):
    model.eval()
    model = model.to(device) 
    correct = 0
    total = len(test_labels)
    
    for i in range(len(test_pairs)):
        g1, g2 = test_pairs[i]
        label = test_labels[i].to(device)

        g1 = g1.to(device)
        g2 = g2.to(device)

        batch_g1 = Batch.from_data_list([g1])
        batch_g2 = Batch.from_data_list([g2])
        
        with torch.no_grad():
            preds = model(batch_g1, batch_g2)
            predicted_label = preds.argmax(dim=1).item()
        if predicted_label == label.item():
            correct += 1
        print(predicted_label, end=" ")
    
    accuracy = correct / total
    print(f"\nDokładność na zbiorze testowym: {accuracy:.2f}")



test_graphs = [
    (
        Data(edge_index=torch.tensor([[0, 1], [1, 0]]), x=torch.ones((2, 1))),
        Data(edge_index=torch.tensor([[1, 0], [0, 1]]), x=torch.ones((2, 1)))
    ),
    (
        Data(edge_index=torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]]), x=torch.ones((3, 1))),
        Data(edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]), x=torch.ones((3, 1)))
    ),
    (
        Data(edge_index=torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]]), x=torch.ones((3, 1))),
        Data(edge_index=torch.tensor([[1, 2, 2, 0, 0, 1], [2, 1, 0, 2, 1, 0]]), x=torch.ones((3, 1)))
    ),
    (
        Data(edge_index=torch.tensor([[0, 1], [1, 0]]), x=torch.ones((3, 1))),
        Data(edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]), x=torch.ones((3, 1)))
    ),
    (
        Data(edge_index=torch.tensor([[0, 1], [1, 0]]), x=torch.ones((2, 1))),
        Data(edge_index=torch.tensor([[0, 1], [1, 0]]), x=torch.ones((2, 1)))
    )
]
test_labels = torch.tensor([1, 0, 1, 0, 1], dtype=torch.long)


for i in range(0,15):
    last_checkpoint_path = f"checkpoints/gin-epochepoch={i}.ckpt"
    trained_model = GraphIsomorphismModel.load_from_checkpoint(last_checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_labels = test_labels.to(device)
    print("#"*50)
    test_model(trained_model, test_graphs, test_labels, device)
    print("#"*50)