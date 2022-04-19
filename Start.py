import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import time
import pickle
import torch
import data
from torchsummary import summary
import argparse

parser = argparse.ArgumentParser(description='Pytorch Cifar-10')
parser.add_argument('--batch-size', type=int, default = 32, metavar='', help = 'Input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default = 20, metavar='', help = 'Number of epochs to train (default:20)')
parser.add_argument('--lr', type=float, default = 0.01, metavar='', help = 'Learning rate (default:0.01)')
parser.add_argument('--model', type=str, default = 'LeNet', metavar='', help = 'Use the Model to train (defaule:LeNet), conclude: LeNet, AlexNet')
parser.add_argument('--model-save-path', type=str, default = './model.ckpt', metavar='', help = 'The Model save path (default:./model.ckpt)')
parser.add_argument('--cuda', default = False, action='store_true', help = 'Weather use cuda to train (default:False)')

def test(path, dataset, batch_size, device, con = 'test'):
    test_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size)
    model = torch.load(path).to(device)
    loss, acc = test_epoch(model, test_loader, device, con)
    return loss, acc

def test_epoch(model, data_loader, device, con):
    model.eval()
    test_loss = 0
    correct = 0
    losss = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data.float().to(device))
            # test_loss += torch.nn.functional.nll_loss(output, target.long().to(device), reduction = 'sum').item()
            test_loss += (losss(output, target.long().to(device)).item() * len(data))
            pred = output.max(1)[1]
            correct += pred.eq(target.long().to(device)).sum().item()
        test_loss /= len(data_loader.dataset)
        print('{} Loss:{}, Acc:{}/{} ({:.2f}%)'.format(con, test_loss, correct, len(data_loader.dataset), 100. * correct / len(data_loader.dataset)))
    return test_loss, correct / len(data_loader.dataset)

def train(model, device, dataset, batch_size, epochs, test_dataset, lr = 0.001, path = 'LeNet.ckpt'):
    train_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    losses = np.inf
    loss_train = []
    acc_train = []
    loss_test = []
    acc_test = []
    for epoch in range(epochs):
        losses = train_epoch(epoch, model, device, train_loader, optimizer, losses, path)
        lt, at = test(path, test_dataset, batch_size, device, 'test')
        loss_test.append(lt)
        acc_test.append(at)
        lt, at = test(path, dataset, batch_size, device, 'train')
        loss_train.append(lt)
        acc_train.append(at)
    return loss_train, acc_train, loss_test, acc_test

def train_epoch(epoch, model, device, data_loader, optimizer, losses, path):
    model.train()
    all_loss = 0
    losss = torch.nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(data.float().to(device))
        # loss = torch.nn.functional.nll_loss(output, target.long().to(device))
        loss = losss(output, target.long().to(device))
        loss.backward()
        optimizer.step()
        all_loss += (loss.item() * len(data))
    if all_loss/len(data_loader.dataset) < losses:
        losses = all_loss/len(data_loader.dataset)
        torch.save(model, path)
    print('Train Epoch:{}\tLoss:{:.6f}\tMin_Loss:{:.6f} with {}'.format(epoch, all_loss / len(data_loader.dataset), losses, device))
    return losses

def get_model(model_name, device):
    assert model_name == 'LeNet' or model_name == 'AlexNet'
    if model_name == 'LeNet':
        from LeNet import LeNet
        return LeNet().to(device)
    if model_name == 'AlexNet':
        from AlexNet import AlexNet
        return AlexNet().to(device)

def get_device(use_cuda):
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    return device

def get_loss_show(loss_train, loss_test):
    x = np.linspace(1, len(loss_train), len(loss_train))
    l1 = plt.plot(x, loss_train, color = 'green', label= 'train loss')
    l2 = plt.plot(x, loss_test, color = 'blue', label = 'test loss')
    plt.legend(loc = 'best')
    plt.title('Loss with train and test')
    plt.show()

def get_acc_show(acc_train, acc_test):
    x = np.linspace(1, len(acc_train), len(acc_train))
    l1 = plt.plot(x, acc_train, color = 'green', label = 'train acc')
    l2 = plt.plot(x, acc_test, color = 'blue', label = 'test acc')
    plt.legend(loc = 'best')
    plt.title('Acc with train and test')
    plt.show()

if __name__ == '__main__':
    args = parser.parse_args()
    train_dataset = data.train_dataset
    test_dataset = data.test_dataset
    batch_size = args.batch_size
    epochs =args.epochs
    lr = args.lr
    device = get_device(args.cuda)
    model = get_model(args.model, device)
    model_save_path = args.model_save_path
    summary(model, (3,32,32))
    p_time = time.time()
    loss_train, acc_train, loss_test, acc_test = train(model, device, train_dataset, batch_size, epochs, test_dataset, lr, model_save_path)
    print('Use {}s'.format(time.time() - p_time))
    get_loss_show(loss_train, loss_test)
    get_acc_show(acc_train, acc_test)
    
