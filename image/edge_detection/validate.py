import torch
from torch.utils.data import DataLoader
from data import EdgeDetectionDataset
from model import UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)
model.load_state_dict(torch.load('unet_edge_detection.pth'))
model.eval()

images_dir = './images/'
json_file = './test.json'
dataset = EdgeDetectionDataset(json_file, images_dir)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

for images in dataloader:
    images = images['pixel_values'].to(device)
    with torch.no_grad():
        outputs = model(images)
        predictions = torch.sigmoid(outputs).cpu().numpy()
        print(predictions)
    break
