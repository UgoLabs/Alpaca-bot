import torch
import torch.nn as nn

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class VisionHead(nn.Module):
    """
    1D ResNet to treat time-series data as a 'visual' pattern.
    Detects shapes like flags, wedges, and head-and-shoulders in the raw signal.
    """
    def __init__(self, input_channels, output_dim=128):
        super(VisionHead, self).__init__()
        
        # Input: (Batch, Features, Time)
        # We treat 'Features' as Channels (OHLCV = 5 channels)
        
        self.initial_conv = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        self.layer1 = ResidualBlock1D(32, 64, stride=2)
        self.layer2 = ResidualBlock1D(64, 128, stride=2)
        self.layer3 = ResidualBlock1D(128, 256, stride=2)
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (Batch, Window_Size, Features)
        # Conv1d expects: (Batch, Channels, Length) -> (Batch, Features, Window_Size)
        x = x.permute(0, 2, 1)
        
        out = self.initial_conv(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = self.global_avg_pool(out)
        out = out.view(out.size(0), -1) # Flatten
        out = self.fc(out)
        out = self.relu(out)
        return out
