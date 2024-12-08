from sklearn.model_selection import train_test_split
from data import generate_isomorphic_graph_pairs
from data import GraphDataModule
from lightning_module import GraphIsomorphismModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

graph_pairs, labels = generate_isomorphic_graph_pairs(num_pairs=100000)
train_pairs, test_pairs, train_labels, test_labels = train_test_split(
    graph_pairs, labels, test_size=0.2, random_state=42
)

datamodule = GraphDataModule(train_pairs, train_labels, test_pairs, test_labels)
model = GraphIsomorphismModel()

checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints", 
    filename="gin-epoch{epoch}", 
    save_top_k=-1, 
    save_weights_only=False, 
    verbose=True,
)

trainer = pl.Trainer(
    max_epochs=20, 
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    callbacks=[checkpoint_callback]
)
trainer.fit(model, datamodule)


