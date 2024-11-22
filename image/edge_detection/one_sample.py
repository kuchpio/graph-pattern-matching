import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from safetensors.torch import load_file
from model import UNet

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

model = UNet()
state_dict = load_file('result/checkpoint-3750/model.safetensors')
model.load_state_dict(state_dict)
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

image_path = '../beaglee.jpg'
image = Image.open(image_path).convert('RGB')
input_image = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(input_image)['logits']
    print(outputs)
    logits = outputs.squeeze(0).cpu().numpy()
    edge_mask = (logits > 0.5).astype('uint8') * 255

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(edge_mask, cmap='gray')
plt.title('Edge Mask')
plt.axis('off')
plt.show()
