import numpy as np
import os
import pickle
import torch

DATA_PATH = './cifar-10-batches-py'
FILENAME = ['data_batch_' + str(i) for i in range(1,6)]

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

test_data = unpickle(os.path.join(DATA_PATH, 'test_batch'))[b'data'].reshape(-1,3,32,32) / 255.0
test_label = unpickle(os.path.join(DATA_PATH, 'test_batch'))[b'labels']

datas = np.vstack([np.array(unpickle(os.path.join(DATA_PATH, FILENAME[i]))[b'data']) for i in range(5)])
labels = np.vstack([np.array(unpickle(os.path.join(DATA_PATH, FILENAME[i]))[b'labels']) for i in range(5)])
datas = datas.reshape(-1,3,32,32) / 255.0
labels = labels.reshape(labels.shape[0] * labels.shape[1])

class CifarDataset(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
    
    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]
        return torch.tensor(data), torch.tensor(label)
    
    def __len__(self):
        return self.data.shape[0]

train_dataset = CifarDataset(datas, labels)
test_dataset = CifarDataset(test_data, test_label)