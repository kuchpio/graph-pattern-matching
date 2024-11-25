from sklearn.model_selection import train_test_split
from data import generate_subisomorphic_graph_pairs
from data import GraphDataModule
from lightning_module import SubgraphIsomorphismModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


graph_pairs, labels = generate_subisomorphic_graph_pairs(num_pairs=50000)

train_pairs, test_pairs, train_labels, test_labels = train_test_split(
    graph_pairs, labels, test_size=0.2, random_state=42
)

datamodule = GraphDataModule(train_pairs, train_labels, test_pairs, test_labels)
model = SubgraphIsomorphismModel()

checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints", 
    filename="gin-epoch{epoch:02d}-val_loss{val_loss:.2f}", 
    save_top_k=-1, 
    save_weights_only=False, 
    verbose=True,
)

trainer = pl.Trainer(
    max_epochs=30, 
    accelerator="gpu",
    callbacks=[checkpoint_callback]
    )
trainer.fit(model, datamodule)


