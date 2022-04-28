import torch

class LeNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 3*32*32, conv1 output: 20*28*28
        self.conv1 = torch.nn.Conv2d(3, 20, kernel_size = (5,5), stride = 1)
        # 20*28*28->20*14*14
        self.pool1 = torch.nn.MaxPool2d(2)
        # 20*14*14->50*10*10
        self.conv2 = torch.nn.Conv2d(20, 50, kernel_size = (5,5), stride = 1)
        # 50*10*10->50*5*5
        self.pool2 = torch.nn.MaxPool2d(2)
        # 50*5*5->500
        self.fc1 = torch.nn.Linear(1250, 500)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(500, 10)
        self.relu2 = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)
    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.relu1(self.fc1(x.view(-1,1250)))
        x = self.fc2(x)
        return x
        