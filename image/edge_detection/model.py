import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Definiujemy bloki enkodera
        self.enc1 = self.contracting_block(3, 64)
        self.enc2 = self.contracting_block(64, 128)
        self.enc3 = self.contracting_block(128, 256)
        self.enc4 = self.contracting_block(256, 512)
        # Definiujemy bloki dekodera
        self.uptrans1 = self.up_transpose_block(512, 256)
        self.dec1 = self.expansive_block(512, 256)
        self.uptrans2 = self.up_transpose_block(256, 128)
        self.dec2 = self.expansive_block(256, 128)
        self.uptrans3 = self.up_transpose_block(128, 64)
        self.dec3 = self.expansive_block(128, 64)
        # Warstwa wyjściowa
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
        
    def contracting_block(self, in_channels, out_channels, kernel_size=3):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=1),
            nn.ReLU(),
        )
        return block

    def expansive_block(self, in_channels, out_channels, kernel_size=3):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=1),
            nn.ReLU(),
        )
        return block

    def up_transpose_block(self, in_channels, out_channels, kernel_size=2, stride=2):
        block = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        return block
    
    def calculate_pos_weight(self, labels):
        total_pixels = labels.numel()
        positive_pixels = labels.sum()
        negative_pixels = total_pixels - positive_pixels
        pos_weight = negative_pixels / (positive_pixels + 1e-7)
        return pos_weight.item()
    
    def forward(self, pixel_values, labels=None): 
        # Enkoder
        enc1 = self.enc1(pixel_values)
        enc2 = self.enc2(nn.functional.max_pool2d(enc1, kernel_size=2))
        enc3 = self.enc3(nn.functional.max_pool2d(enc2, kernel_size=2))
        enc4 = self.enc4(nn.functional.max_pool2d(enc3, kernel_size=2))

        # Dekoder
        dec1 = self.uptrans1(enc4)
        dec1 = torch.cat((dec1, enc3), dim=1)
        dec1 = self.dec1(dec1)
        dec2 = self.uptrans2(dec1)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        dec3 = self.uptrans3(dec2)
        dec3 = torch.cat((dec3, enc1), dim=1)
        dec3 = self.dec3(dec3)

        out = self.final_conv(dec3)
        out = out.squeeze(1)

        if labels is not None:
            labels = labels.float()
            pos_weight_value = self.calculate_pos_weight(labels)
            pos_weight = torch.tensor([pos_weight_value]).to(labels.device)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss = loss_fn(out, labels)
            return {'loss': loss, 'logits': out}
        else:
            return {'logits': out}