import torch
import torch.nn as nn
import torch.optim as optim
from load_data import load_group_eeg_data, load_combined_eeg_data
from make_dataset import MyDataset
from model import LSTM, LSTM_CNN
from torch.utils.data import DataLoader
from train_test import train, test

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True


def main():
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('DEVICE: ', DEVICE)

    epochs = 40
    batch_size = 64
    date = '06_03'
    group = 1
    sorted_ = True

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
    model = LSTM_CNN(num_classes=2, channels=x_train.shape[1], 
                     input_size=32, hidden_size=32, num_layers=2)
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    train(model, criterion, optimizer, train_loader, DEVICE, train_num=train_num, epochs=epochs)
    test(model, criterion, test_loader, DEVICE, test_num=test_num)


if __name__ == '__main__':
    main()
