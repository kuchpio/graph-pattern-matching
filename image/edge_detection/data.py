import os
import json
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import torchvision.transforms as transforms

class EdgeDetectionDataset(Dataset):
    def __init__(self, json_file, images_dir, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        with open(json_file, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        filename = sample['filename']
        width = sample['width']
        height = sample['height']
        image_path = os.path.join(self.images_dir, filename)
        image = Image.open(image_path).convert('RGB')
        
        if width and height:
            image = image.resize((width, height))
            edge_mask = self.create_edge_mask(sample, (width, height))

        if self.transform:
            image = self.transform(image)
            edge_mask = self.transform(edge_mask)
        else:
            image = transforms.ToTensor()(image)
            edge_mask = transforms.ToTensor()(edge_mask)

        if edge_mask.dim() == 3 and edge_mask.size(0) == 1:
            edge_mask = edge_mask.squeeze(0)

        return {'pixel_values': image, 'labels': edge_mask}

    def create_edge_mask(self, sample, size):
        width, height = size
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)
        
        if 'junctions' in sample and 'edges_positive' in sample:
            junctions = sample['junctions']
            edges = sample['edges_positive']
            for edge in edges:
                idx1, idx2 = edge
                x1, y1 = junctions[idx1]
                x2, y2 = junctions[idx2]
                draw.line([x1, y1, x2, y2], fill=255, width=1)
        elif 'junc' in sample and 'lines' in sample:
            junctions = sample['junc']
            edges = sample['lines']
            for edge in edges:
                draw.line(edge, fill=255, width=1)
        else:
            raise ValueError("Sample does not contain required keys")

        return mask
