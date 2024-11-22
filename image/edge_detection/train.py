import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
import torchvision.transforms as transforms
from model import UNet
from data import EdgeDetectionDataset

def compute_metrics(predictions):
    logits = torch.tensor(predictions.predictions)
    labels = torch.tensor(predictions.label_ids)
    preds = torch.sigmoid(logits).round()
    intersection = (preds * labels).sum()
    union = preds.sum() + labels.sum()
    dice_coeff = (2 * intersection + 1e-7) / (union + 1e-7)
    return {'dice_coefficient': dice_coeff.item()}

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)

images_dir = '../images/'
train_json = '../train.json'
test_json = '../test.json'
dataset_train = EdgeDetectionDataset(train_json, images_dir, transform=transform)
dataset_test = EdgeDetectionDataset(test_json, images_dir, transform=transform)
dataloader_train = DataLoader(dataset_train, batch_size=8, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=8, shuffle=True)

training_args = TrainingArguments(
    output_dir='./result',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=20,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='dice_coefficient',
    greater_is_better=True,
    logging_dir='./logs',
    logging_steps=500,
    learning_rate=1e-4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_test,
    compute_metrics=compute_metrics,
)

trainer.train()