import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from model import SubgraphGINModel

class SubgraphIsomorphismModel(pl.LightningModule):
    def __init__(self, hidden_dim=32, lr=1e-3):
        super().__init__()
        self.gnn = SubgraphGINModel(hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, 2) 
        self.lr = lr

    def forward(self, g1, g2):
        match_score = self.gnn(g1, g2) 
        return self.fc(match_score)  

    def training_step(self, batch, batch_idx):
        (g1, g2), labels = batch
        preds = self(g1, g2)
        loss = F.cross_entropy(preds, labels)
        self.log("train_loss", loss)
        return loss


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)