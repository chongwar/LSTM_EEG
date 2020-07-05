import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings('ignore')
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class LSTM(nn.Module):  
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # self.bn = nn.BatchNorm2d(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        """
        shape of x: (N, T, C)  N: batch_size, T: seq_len(time), C: input_size(channel)
        """
        lstm_out, (h, c) = self.lstm(x, None)
        # x = self.bn(lstm_out)
        logits = self.fc(lstm_out[:, -1, :])
        probas = F.softmax(logits, dim=1)
        return probas


class LSTM_CNN(nn.Module):
    def __init__(self, num_classes, channels, input_size, hidden_size, num_layers, 
                 time_kernel_size=16, spatial_num=8, drop_out=0.5):
        super(LSTM_CNN, self).__init__()

        drop_out = 0.5
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        
        # self.block_1 = nn.Sequential(
        #     nn.ZeroPad2d((time_kernel_size // 2 - 1, time_kernel_size // 2, 0, 0)),
        #     nn.Conv2d(1, 8, (1, time_kernel_size), bias=False),
        #     nn.BatchNorm2d(8),
        #     nn.ELU(),
        #     nn.AvgPool2d((1, 4))
        # )
        
        self.block_2 = nn.Sequential(
            nn.Conv2d(1, spatial_num, (channels, 1), bias=False),
            nn.BatchNorm2d(spatial_num),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(drop_out)
        )
        
        self.fc = nn.Linear(spatial_num * self.hidden_size // 4, num_classes)
        
    def forward(self, x):
        # x: (N, C, T)  N: batch_size; C: channels; T: times
        N, C, T = x.shape
        x = x.reshape(N * C, T // self.input_size, self.input_size)
        lstm_out, _ = self.lstm(x, None)
        
        # x: (N, 1, C, H)  H: hidden_size
        x = lstm_out[:, -1, :].reshape(N, 1, C, self.hidden_size)
        
        # x = self.block_1(x)
        x = self.block_2(x)
        
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return probas
        
        
class LSTM_CNN_Half(nn.Module):
    def __init__(self, num_classes, batch_size, T, C, input_size, hidden_size,
                 num_layers, spatial_num=8, drop_out=0.5):
        super(LSTM_CNN_Half, self).__init__()

        self.N = batch_size
        self.T = T
        self.C = C
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pool = 4
        self.seq_len = self.T // self.input_size
        self.fc_in = spatial_num * self.seq_len // 2 * self.hidden_size // self.pool
        
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, 
                            self.num_layers, batch_first=True)
        
        self.block_1 = nn.Sequential(
            nn.Conv2d(1, spatial_num, (self.C, 1)),
            nn.BatchNorm2d(spatial_num),
            nn.ELU(),
            nn.AvgPool2d((1, self.pool)),
            nn.Dropout(drop_out)
        )
        
        self.fc = nn.Linear(self.fc_in , num_classes)
        
    def forward(self, x):
        # input shape of x: (N, 1, C, T)
        self.N = x.shape[0]
        x = x.reshape(self.N * self.C, self.seq_len, self.input_size)
        lstm_out, _ = self.lstm(x, None)
        
        # x: (N, 1, C, self.seq_len // 2 * H)  H: hidden_size
        x = lstm_out[:, -self.seq_len // 2:, :].reshape(self.N, 1, self.C, -1)
        x = self.block_1(x)
        
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return probas
    
    
class LSTM_CNN_Spatial(nn.Module):
    def __init__(self, num_classes, batch_size, T, C, input_size, hidden_size,
                 num_layers, spatial_num=8, drop_out=0.5):
        super(LSTM_CNN_Spatial, self).__init__()

        self.N = batch_size
        self.T = T
        self.C = C
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pool = 4
        self.seq_len = self.T // self.input_size
        self.fc_in = spatial_num * self.hidden_size // self.pool
        
        self._lstm = nn.LSTM(self.input_size, self.hidden_size, 
                            self.num_layers, batch_first=True)
        self.lstm = nn.ModuleList([self._lstm for i in range(self.C)])
        
        self.block_1 = nn.Sequential(
            nn.Conv2d(1, spatial_num, (self.C, 1)),
            nn.BatchNorm2d(spatial_num),
            nn.ELU(),
            nn.AvgPool2d((1, self.pool)),
            nn.Dropout(drop_out)
        )
        
        self.fc = nn.Linear(self.fc_in , num_classes)
        
    def forward(self, x):
        # input shape of x: (N, 1, C, T)
        self.N = x.shape[0]
        x = x.reshape(self.N, self.C, self.seq_len, self.input_size)
        _x = None
        for index, lstm in enumerate(self.lstm):
            lstm_out, _ = lstm(x[:, index, :, :], None)
            tmp = lstm_out[:, -1, :]
            tmp = tmp.unsqueeze(0)
            if _x is None:
                _x = tmp
            else:
                _x = torch.cat((_x, tmp), dim=0)
        
        # (C, N, H) ===> (N, 1, C, H)   H: hidden_size
        x = _x.permute(1, 0, 2).unsqueeze(1)
        x = self.block_1(x)
        
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return probas  


if __name__ == '__main__':
    # model test
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    x = torch.randn((4, 1, 64, 256))
    x = x.to(DEVICE)
    model = LSTM_CNN_Spatial(2, 4, 256, 64, 16, 16, 2)
    model = model.to(DEVICE)
    y = model(x)
    print(y.data)
