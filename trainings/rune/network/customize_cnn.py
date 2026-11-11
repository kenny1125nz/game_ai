import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# 1. RE-DECLARE THE ARCHITECTURE (Must match training exactly)
class RuneNet(nn.Module):
    def __init__(self, num_classes=35, input_h=32, input_w=72):
        super(RuneNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # # This matches our 32x72 logic (4 * 9)
        # final_h, final_w = input_h // 8, input_w // 8
        # self.flatten_size = 256 * final_h * final_w

        # --- THE FIX ---
        # Height: 32 // 8 = 4
        # Width: 72 // 8 = 9
        # Channels: 64 (Must match the output of conv4)
        final_h, final_w = input_h // 8, input_w // 8
        self.flatten_size = 64 * final_h * final_w # Changed from 256 to 64
        
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.dropout = nn.Dropout(0.7)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        x = self.pool2(F.relu(self.conv3(x)))
        x = self.pool3(F.relu(self.conv4(x)))
        x = x.view(-1, self.flatten_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 2. THE LOADING FUNCTION
def load_inference_model(model_path, device="cuda"):
    # 1. Initialize blank model
    model = RuneNet(num_classes=35, input_h=32, input_w=72)
    
    # 2. Load the ENTIRE checkpoint file
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    
    # 3. EXTRACT the actual weights from the "model_state_dict" key
    # This fixes the "Missing key(s)" and "Unexpected key(s)" error
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint # Fallback if you save it differently later

    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    return model

# 3. PRE-PROCESSING (Must match training normalization)
inference_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((32, 72)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) # Same shift as training
])

train_transforms = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 72)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) # Same shift as training
])

# Example Usage:
# model = load_inference_model("rune_model_v2_72w.pth")