import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, num_chars, hidden_size=256):
        super(CRNN, self).__init__()

        # 1. CNN Backbone (Feature Extraction)
        # Input: [Batch, 3, 32, 256] -> Output: [Batch, 512, 1, 64]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [64, 16, 128]
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [128, 8, 64]
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # [256, 4, 64]
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # [512, 2, 64]
            nn.Conv2d(512, 512, 2),
            nn.ReLU(),  # [512, 1, 63]
        )

        # 2. RNN (Sequence Processing)
        # We treat the width (63-64 slices) as the "time steps"
        # Separate them so we can handle the tuple outputs
        self.rnn1 = nn.LSTM(512, hidden_size, bidirectional=True, batch_first=True)
        self.rnn_bridge = nn.Linear(hidden_size * 2, hidden_size)
        self.rnn2 = nn.LSTM(
            hidden_size, hidden_size, bidirectional=True, batch_first=True
        )

        # 3. Fully Connected (Classification)
        # Output must be num_chars + 1 (for the CTC blank token)
        self.fc = nn.Linear(hidden_size * 2, num_chars + 1)

    def forward(self, x):
        # 1. CNN extraction
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        conv = conv.view(b, c, w).permute(0, 2, 1)  # [B, W, C]

        # 2. First RNN Layer
        # LSTM returns (output, (h, c)), we only want the 'output' [B, W, H*2]
        out, _ = self.rnn1(conv)

        # 3. The Bridge (Reduce dimensions)
        out = self.rnn_bridge(out)

        # 4. Second RNN Layer
        out, _ = self.rnn2(out)

        # 5. Classifier
        out = self.fc(out)

        # [TimeSteps, Batch, Classes] for CTCLoss
        return out.permute(1, 0, 2)
