import torch

class AlexNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 48, kernel_size = 11, stride = 4, padding = 2)
        self.pool1 = torch.nn.MaxPool2d(kernel_size = 3, stride = 2)
        self.conv2 = torch.nn.Conv2d(48, 128, kernel_size = 5, padding = 2)
        self.pool2 = torch.nn.MaxPool2d(kernel_size = 3, stride = 2)
        self.conv3 = torch.nn.Conv2d(128, 192, kernel_size = 3, padding = 1)
        self.conv4 = torch.nn.Conv2d(192, 192, kernel_size = 3, padding = 1)
        self.conv5 = torch.nn.Conv2d(192, 128, kernel_size = 3, padding = 1)
        self.pool3 = torch.nn.MaxPool2d(kernel_size = 3, stride = 2)

        # self.Flatten = torch.nn.Flatten(start_dim = 1, end_dim = -1)
        # self.Linear1 = torch.nn.Linear(4608, 2048)
        self.Linear1 = torch.nn.LazyLinear(2048)
        self.drop1 = torch.nn.Dropout(0.5)
        self.Linear2 = torch.nn.Linear(2048, 2048)
        self.drop2 = torch.nn.Dropout(0.5)
        self.Linear3 = torch.nn.Linear(2048, 10)

    def forward(self, x):
    	x = self.pool1(torch.nn.functional.relu(self.conv1(x)))
    	x = self.pool2(torch.nn.functional.relu(self.conv2(x)))
    	x = torch.nn.functional.relu(self.conv3(x))
    	x = torch.nn.functional.relu(self.conv4(x))
    	x = torch.nn.functional.relu(self.conv5(x))
    	x = self.pool3(x)
    	x = x.view(x.size()[0], -1)
    	x = self.drop1(torch.nn.functional.relu(self.Linear1(x)))
    	x = self.drop2(torch.nn.functional.relu(self.Linear2(x)))
    	x = self.Linear3(x)
    	return x