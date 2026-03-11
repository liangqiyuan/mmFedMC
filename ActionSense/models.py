import torch.nn as nn
import torch.nn.functional as F

class Eye_LSTM(nn.Module):
    def __init__(self):
        super(Eye_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, 20)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return nn.LogSoftmax(dim=1)(x)

class EMG_LSTM(nn.Module):
    def __init__(self):
        super(EMG_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=8, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, 20)
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return nn.LogSoftmax(dim=1)(x)

class Tactile_LSTM(nn.Module):
    def __init__(self):
        super(Tactile_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=32*32, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, 20)

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return nn.LogSoftmax(dim=1)(x)

class IMU_LSTM(nn.Module):
    def __init__(self):
        super(IMU_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=22 * 3, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, 20)
        
    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return nn.LogSoftmax(dim=1)(x)
