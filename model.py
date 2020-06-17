import warnings
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
    def __init__(self, num_classes, channels, input_size, hidden_size, num_layers):
        super(LSTM_CNN, self).__init__()

        drop_out = 0.25
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        
        self.block_1 = nn.Sequential(
            nn.ZeroPad2d((self.hidden_size // 2 - 1, self.hidden_size // 2, 0, 0)),
            nn.Conv2d(1, 8, (1, self.hidden_size), bias=False),
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.AvgPool2d((1, 4))
        )
        
        self.block_2 = nn.Sequential(
            nn.Conv2d(8, 8, (channels, 1), bias=False),
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.Dropout(drop_out)
        )
        
        self.fc = nn.Linear(8 * self.hidden_size // 4, num_classes)
        
    def forward(self, x):
        # x: (N, C, T)  N: batch_size; C: channels; T: times
        N, C, T = x.shape
        x = x.reshape(N * C, T // self.input_size, self.input_size)
        lstm_out, _ = self.lstm(x, None)
        
        # x: (N, 1, C, H)  H: hidden_size
        x = lstm_out[:, -1, :].reshape(N, 1, C, self.hidden_size)
        
        x = self.block_1(x)
        x = self.block_2(x)
        
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return probas
        

if __name__ == '__main__':
    model = LSTM(input_size=64, hidden_size=256, num_layers=2)
    # x = torch.randn((512, 4, 64))
    x = torch.randn((4, 512, 64))
    with torch.no_grad():
        y = model(x)
        print(y.numpy().reshape(-1))
