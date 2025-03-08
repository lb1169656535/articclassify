import torch
import torch.nn as nn
import torch.nn.functional as F  
class DNN(nn.Module):
    """DNN 模型"""
    def __init__(self, input_dim, hidden_units=[16, 32], num_classes=4):
        super(DNN, self).__init__()
        layers = []
        for units in hidden_units:
            layers.append(nn.Linear(input_dim, units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.25))
            input_dim = units
        layers.append(nn.Linear(input_dim, num_classes))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)
# models.py
class CNN(nn.Module):
    """CNN 模型"""
    def __init__(self, num_classes=4):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(2, 2), stride=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(2, 2), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(2, 2), stride=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.2)
        )
        # 动态计算全连接层输入维度
        self._to_linear = None  # 用于存储卷积层输出的特征维度
        
        # 前向传播一次以计算特征维度
        with torch.no_grad():
            x = torch.randn(1, 1, 13, 180)  # 输入形状 (batch, channel, height, width)
            x = self.conv_layers(x)
            self._to_linear = x.shape[1] * x.shape[2] * x.shape[3]
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self._to_linear, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # 展平
        return self.fc_layers(x)

class LSTM(nn.Module):
    """LSTM 模型"""
    def __init__(self, input_size=13, hidden_size=102, num_layers=3, num_classes=4):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        h_lstm = self.dropout(h_lstm[:, -1, :])
        return self.fc(h_lstm)


# 新增模型类
class ResNet(nn.Module):
    """ResNet 模型"""
    def __init__(self, num_classes=4):
        super(ResNet, self).__init__()
        
        # 初始卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 残差块
        self.resblock1 = ResidualBlock(16, 32)
        self.resblock2 = ResidualBlock(32, 64)
        self.resblock3 = ResidualBlock(64, 128)
        
        # 全连接层
        self._to_linear = None
        with torch.no_grad():
            x = torch.randn(1, 1, 13, 180)
            x = self.conv1(x)
            x = self.resblock1(x)
            x = self.resblock2(x)
            x = self.resblock3(x)
            self._to_linear = x.shape[1] * x.shape[2] * x.shape[3]
        
        self.fc = nn.Sequential(
            nn.Linear(self._to_linear, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class GRU(nn.Module):
    """GRU 模型"""
    def __init__(self, input_size=13, hidden_size=102, num_layers=3, num_classes=4):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h_gru, _ = self.gru(x)
        h_gru = self.dropout(h_gru[:, -1, :])
        return self.fc(h_gru)