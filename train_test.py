import sys
import torch
import numpy as np
from tqdm import trange


def train(model, criterion, optimizer, data_loader, device, train_num, epochs, logged=False):
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
            _, pred = torch.max(y_pred, 1)
            if sys.platform == 'linux':
                correct_num += np.sum(pred.cpu().numpy() == y.cpu().numpy())
            else: 
                correct_num += np.sum(pred.numpy() == y.numpy())
            
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += float(loss.item())
            
        batch_num = train_num // batch_size
        _loss = running_loss / (batch_num + 1)
        acc = correct_num / train_num * 100
        if not logged:
            print(f'Epoch {epoch+1}/{epochs}\tTrain loss: {_loss:.4f}\tTrain acc: {acc:.2f}%')
    if not logged:
        print('Finish Training!')
         
         
def test(model, criterion, data_loader, device, test_num, log, logged):
    running_loss = 0.0
    correct_num = 0
    model.eval()
    batch_size = None
    for index, data in enumerate(data_loader):
        x, y = data
        batch_size = x.shape[0] if index == 0 else batch_size
        x, y = x.to(device), y.to(device)
        
        y_pred = model(x)
        _, pred = torch.max(y_pred, 1)
        if sys.platform == 'linux':
            correct_num += np.sum(pred.cpu().numpy() == y.cpu().numpy())
        else: 
            correct_num += np.sum(pred.numpy() == y.numpy())
        
        loss = criterion(y_pred, y)
        running_loss += float(loss.item())
    
    batch_num = test_num // batch_size
    _loss = running_loss / (batch_num + 1)
    acc = correct_num / test_num * 100
    print(f'Test loss: {_loss:.4f}\tTest acc: {acc:.2f}%')
    if logged:
        log.append(f'{acc:.2f}\t\n')
        with open('result.txt', 'a') as f:
            f.writelines(log)
            