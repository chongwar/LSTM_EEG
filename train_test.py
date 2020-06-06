import torch
import numpy as np
from tqdm import trange


def train(model, criterion, optimizer, data_loader, device, train_num, epochs=20):
    for epoch in trange(epochs):
        model.train()
        running_loss = 0.0
        correct_num = 0
        batch_size = None
        for index, data in enumerate(data_loader):
            x, y = data
            batch_size = x.shape[0] if index == 0 else batch_size
            x, y = x.to(device), y.to(device)
            
            y_pred = model(x)
            # correct_num += np.sum((y.numpy().reshape(-1) == np.round(y_pred.data.numpy().reshape(-1))))
            correct_num += np.sum((y.cpu().numpy().reshape(-1) == np.round(y_pred.data.cpu().numpy().reshape(-1))))  # on linux
            
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += float(loss.item())
            
        batch_num = train_num // batch_size
        _loss = running_loss / (batch_num + 1)
        acc = correct_num / train_num * 100
        print(f'Epoch {epoch+1}/{epochs}\tTrain loss: {_loss:.4f}\tTrain acc: {acc:.2f}%')
    print('Finish Training!')
         
         
def test(model, criterion, data_loader, device, test_num):
    running_loss = 0.0
    correct_num = 0
    model.eval()
    batch_size = None
    for index, data in enumerate(data_loader):
        x, y = data
        batch_size = x.shape[0] if index == 0 else batch_size
        x, y = x.to(device), y.to(device)
        
        y_pred = model(x)
        # correct_num += np.sum((y.numpy().reshape(-1) == np.round(y_pred.data.numpy().reshape(-1))))
        correct_num += np.sum((y.cpu().numpy().reshape(-1) == np.round(y_pred.data.cpu().numpy().reshape(-1))))  # on linux
        
        loss = criterion(y_pred, y)
        running_loss += float(loss.item())
    
    batch_num = test_num // batch_size
    _loss = running_loss / (batch_num + 1)
    acc = correct_num / test_num * 100
    print(f'Test loss: {_loss:.4f}\tTest acc: {acc:.2f}%')     
            