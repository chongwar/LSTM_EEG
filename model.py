import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings('ignore')
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class LSTM(nn.Module):  
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # self.bn = nn.BatchNorm2d(hidden_size)
        self.hidden2tag = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        """
        shape of x: (T, N, C) if 'batch_first=True' else (N, T, C)  
        T: seq_len, N: batch_size, C: input_size
        """
        lstm_out, _ = self.lstm(x, None)
        # x = self.bn(lstm_out)
        x = self.hidden2tag(lstm_out[:, -1, :])
        x = F.sigmoid(x)
        return x
        

if __name__ == '__main__':
    model = LSTM(input_size=64, hidden_size=256, num_layers=2, batch_size=4)
    # x = torch.randn((512, 4, 64))
    x = torch.randn((4, 512, 64))
    with torch.no_grad():
        y = model(x)
        print(y.numpy().reshape(-1))
