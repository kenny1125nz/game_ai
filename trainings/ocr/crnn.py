import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, num_chars, hidden_size=128):
        super(CRNN, self).__init__()

        # 1. Feature Extractor (Reduced from 512 to 128/256)
        # Input: [B, 1, 32, 256]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # -> [B, 32, 16, 128]
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # -> [B, 64, 8, 64]
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1)),  # -> [B, 128, 4, 64] (Keep width detail)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1)),  # -> [B, 256, 2, 64]
            nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(True),  # Final: [B, 256, 1, 63]
        )

        # 2. Sequence Learner (Single Bidirectional LSTM)
        # Input to LSTM: [63, B, 256]
        self.rnn = nn.LSTM(256, hidden_size, bidirectional=True, batch_first=True)

        # 3. Classifier
        # Input: [B, 63, hidden_size * 2]
        self.fc = nn.Linear(hidden_size * 2, num_chars + 1)

    def forward(self, x):
        # CNN Extraction
        conv = self.cnn(x)  # [B, 256, 1, 63]

        # Prepare for RNN
        b, c, h, w = conv.size()
        conv = conv.view(b, c * h, w)

        # Permute to [Batch, Time, Features]
        conv = conv.permute(0, 2, 1)  # [B, 63, 256]

        # RNN Step
        output, _ = self.rnn(conv)  # [B, 63, 256]

        # Classification
        logits = self.fc(output)  # [B, 63, num_chars + 1]

        # Permute for CTCLoss: [Time, Batch, Classes]
        return logits.permute(1, 0, 2)
