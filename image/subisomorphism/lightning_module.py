import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from model import GraphMatchingModel

class GraphIsomorphismModel(pl.LightningModule):
    def __init__(self, hidden_dim=32, lr=1e-3):
        super().__init__()
        self.model = GraphMatchingModel(hidden_dim)
        self.lr = lr

    def forward(self, g1, g2):
        return self.model(g1, g2)

    def training_step(self, batch, batch_idx):
        g1, g2, labels = batch
        preds = self(g1, g2)
        loss = F.cross_entropy(preds, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        g1, g2, labels = batch
        preds = self(g1, g2)
        val_loss = F.cross_entropy(preds, labels)
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)