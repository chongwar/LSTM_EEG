import torch
import torch.nn as nn
import torch.optim as optim
from load_data import load_group_eeg_data, load_combined_eeg_data
from make_dataset import MyDataset
from model import LSTM, LSTM_CNN, LSTM_CNN_Half, LSTM_CNN_Spatial
from torch.utils.data import DataLoader
from train_test import train, test

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True


def main(epochs, batch_size, input_size, hidden_size, num_layers, spatial_num, drop_out, logged=False):
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('DEVICE: ', DEVICE)

    date = '06_03'
    group = 1
    sorted_ = True
    # sorted_ = False

    # load data from '.npy' file
    # x_train, x_test, y_train, y_test = load_group_eeg_data(date, group, sorted_=sorted_)
    x_train, x_test, y_train, y_test = load_combined_eeg_data(date, sorted_=sorted_)
    # x: (N, C, T)  N: trials  C: channels  T: times 
    train_num, test_num = x_train.shape[0], x_test.shape[0]
    
    # make dataset for train and test
    train_data = MyDataset(x_train, x_test, y_train, y_test)
    test_data = MyDataset(x_train, x_test, y_train, y_test, train=False)
    train_loader = DataLoader(train_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    # model initiation
    # model = LSTM(num_classes=2, input_size=64, hidden_size=256, num_layers=2)
    
    # model = LSTM_CNN(num_classes=2, channels=x_train.shape[1], input_size=input_size, hidden_size=hidden_size, 
    #                  num_layers=num_layers, spatial_num=spatial_num, drop_out=drop_out)
    
    # model = LSTM_CNN_Half(num_classes=2, batch_size=batch_size, T=x_train.shape[-1],
    #                       C=x_train.shape[-2], input_size=input_size, hidden_size=hidden_size,
    #                       num_layers=num_layers, spatial_num=spatial_num)
    
    model = LSTM_CNN_Spatial(num_classes=2, batch_size=batch_size, T=x_train.shape[-1],
                          C=x_train.shape[-2], input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, spatial_num=spatial_num)
    
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    log = []
    if logged:
        log.append(f'{epochs}\t{batch_size}\t{input_size}\t{hidden_size}\t'
                   f'{num_layers}\t{spatial_num}\t{drop_out}\t')
    train(model, criterion, optimizer, train_loader, DEVICE,train_num, epochs, logged)
    test(model, criterion, test_loader, DEVICE, test_num, log, logged)


if __name__ == '__main__':
    """
    Hyperparameter Search
    """    
    # _epochs = [30, 50, 70]
    # _batch_size = [8, 16]
    # _input_size = [16, 32]
    # _hidden_size = [16, 32]
    # _num_layers = [2, 4]
    # _spatial_num = [8, 16]
    # _drop_out = [0.25, 0.5]
    # iters = 5
    # logged = True
    # for epochs in _epochs:
    #     for batch_size in _batch_size:
    #         for input_size in _input_size:
    #             for hidden_size in _hidden_size:
    #                 for num_layers in _num_layers:
    #                     for spatial_num in _spatial_num:
    #                         for drop_out in _drop_out:
    #                             for i in range(iters):
    #                                 main(epochs, batch_size, input_size, hidden_size,
    #                                      num_layers, spatial_num, drop_out, logged)
    
    """
    good parameter:
    epochs  batch_size  input_size  hidden_size  num_layers  spatial_num  drop_out
      30       16           16          32            2          16         0.25
      30       16           16          32            2          16         0.5
      30       16           16          32            4          8          0.25
      30       16           16          32            4          8          0.5
      30       16           16          32            4          16         0.25
      30       16           16          32            4          16         0.5
    """                                
    epochs = 200
    batch_size = 16
    input_size = 16
    hidden_size = 16
    num_layers = 4
    spatial_num = 8
    drop_out = 0.5
    logged = False
    main(epochs, batch_size, input_size, hidden_size, num_layers, spatial_num, drop_out, logged)
