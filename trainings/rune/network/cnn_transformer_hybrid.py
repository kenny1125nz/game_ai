import torch
import torch.nn as nn
import cv2
import os
from torchvision import transforms
# from image.processor import process_image
import numpy as np
from PIL import Image

# --- 1. Modular Components ---
HYBRID_CONFIGS = {
    'tiny':   {'hidden_dim': 64,  'layers': 2, 'nhead': 4, 'mlp_dim': 128, 'dropout': 0.1,'grid_size':(16,36)},
    'small':  {'hidden_dim': 128, 'layers': 4, 'nhead': 8, 'mlp_dim': 256, 'dropout': 0.1,'grid_size':(16,36)},
    'medium': {'hidden_dim': 256, 'layers': 6, 'nhead': 8, 'mlp_dim': 512, 'dropout': 0.2,'grid_size':(16,36)},
    'large':  {'hidden_dim': 512, 'layers': 8, 'nhead': 16, 'mlp_dim': 1024, 'dropout': 0.3,'grid_size':(16,36)}
}

class CNNStem(nn.Module):
    """Reduces 72x32 image to a 9x4 grid of feature vectors"""
    def __init__(self, hidden_dim):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, hidden_dim // 2, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x # Results in 36x16 feature map - more 'data' for the next layers

class TransformerProcessor(nn.Module):
    def __init__(self, config): # Default for your 72x32 stem
        super().__init__()
        grid_size=config['grid_size']
        self.hidden_dim = config['hidden_dim']
        self.num_patches = grid_size[0] * grid_size[1]
        
        # 1. POSITIONAL EMBEDDING (Must match grid_size)
        # We define this dynamically so we can change the CNN later without a rewrite
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.hidden_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=config['nhead'],
            dim_feedforward=config['mlp_dim'],
            dropout=config['dropout'],
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['layers'])

    def forward(self, x):
        # x input: (B, C, H, W) -> e.g., (Batch, 128, 8, 18)
        B, C, H, W = x.shape
        
        # 2. FLATTEN SPATIAL TO SEQUENCE
        # (B, 128, 8, 18) -> (B, 128, 144) -> (B, 144, 128)
        x = x.flatten(2).transpose(1, 2) 
        
        # 3. SAFETY CHECK (Crucial for debugging)
        if x.size(1) != self.pos_embed.size(1):
            raise RuntimeError(f"CNN output {x.size(1)} patches doesn't match Transformer {self.pos_embed.size(1)} patches. "
                               f"Check your CNN stride/padding!")

        x = x + self.pos_embed
        return self.encoder(x)

# --- 2. Main Model Container ---

class RuneHybridNet(nn.Module):
    def __init__(self, size, num_classes):
        super().__init__()
        config = HYBRID_CONFIGS[size]
        self.stem = CNNStem(config['hidden_dim'])
        self.transformer = TransformerProcessor(config)
        
        # Classification Head
        self.head = nn.Sequential(
            nn.LayerNorm(config['hidden_dim']),
            nn.Linear(config['hidden_dim'], num_classes)
        )

    def forward(self, x):
        features = self.stem(x)
        sequence = self.transformer(features)
        
        # Use Global Average Pooling (GAP) on the sequence dimension
        # This makes the model 'location agnostic'
        out = sequence.mean(dim=1) 
        return self.head(out)

# --- 3. Configuration Dictionaries ---


# 2. THE LOADING FUNCTION
def load_inference_model(model_path, device="cuda", size:str='small'):
    # 1. Initialize blank model
    model = RuneHybridNet(num_classes=35,size=size)
    
    # 2. Load the ENTIRE checkpoint file
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(state_dict)
    
    # model.to(device)
    # model.eval()
    return model


# def custom_transform_training(pil_img):
#     image = np.array(pil_img)  # to array for opencv process
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #color  conversion    
#     image = process_image(image) # opencv process
#     # cv2.imshow("post",image)
#     # cv2.waitKey(1)
#     return Image.fromarray(image)  #back to PIL


# def custom_transform_inference(pil_img):
#     image = np.array(pil_img)  # to array for opencv process
#     image = process_image(image) # opencv process
#     return Image.fromarray(image)  #back to PIL    

def train_transforms(size=(32,72)):
    return transforms.Compose([
        # transforms.Grayscale(),
        # transforms.Lambda(custom_transform_training),        # Your custom binary logic
        # transforms.Lambda(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB) if isinstance(x, np.ndarray) else x),
        transforms.Resize(size),
        transforms.ToTensor(),  
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomErasing(
            p=0.5,          # 50% chance to apply it to an image
            scale=(0.02, 0.1), # Erase between 2% and 10% of the image area
            ratio=(0.3, 3.3),  # Aspect ratio of the erased rectangle
            value=0,        # Fill with black (0) or 'random' for noise
            inplace=False
        )
        # transforms.Normalize((0.5,), (0.5,))
])


# 3. PRE-PROCESSING (Must match training normalization)
def inference_transforms(size=(32,72)):
    return  transforms.Compose([
        transforms.Lambda(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB) if isinstance(x, np.ndarray) else x),
        transforms.ToPILImage(),
        # transforms.Grayscale(),
        # transforms.Lambda(custom_transform_inference),        # Your custom binary logic
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # transforms.Normalize((0.5,), (0.5,))
    ])