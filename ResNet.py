import torch

class ResNet(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.pool1 = torch.nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.blocks = []
        self.blocks.append(basic_block(64, 64, 1, 1).to(device))
        self.blocks.append(basic_block(64, 64, 1, 1).to(device))
        self.blocks.append(basic_block(64, 128, 2, 1, 1).to(device))
        self.blocks.append(basic_block(128, 128, 1, 1).to(device))
        self.blocks.append(basic_block(128, 256, 2, 1, 1).to(device))
        self.blocks.append(basic_block(256, 256, 1, 1).to(device))
        self.blocks.append(basic_block(256, 512, 2, 1, 1).to(device))
        self.blocks.append(basic_block(512, 512, 1, 1).to(device))
        self.pool2 = torch.nn.AdaptiveAvgPool2d(output_size = (1,1))
        self.fc = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        for block in self.blocks:
        	x = block(x)
        x = self.pool2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x

class basic_block(torch.nn.Module):
	def __init__(self, in_channels, out_channels, f_stride, s_stride, downsample = 0):
		super(basic_block, self).__init__()
		self.downsample = downsample
		self.block1 = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = f_stride, padding = 1),
			torch.nn.BatchNorm2d(out_channels),
			torch.nn.ReLU(),
			torch.nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = s_stride, padding = 1),
			torch.nn.BatchNorm2d(out_channels),
			torch.nn.ReLU()
		)
		self.block2 = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = f_stride, padding = 0),
			torch.nn.BatchNorm2d(out_channels)
		)

	def forward(self,x):
		residual = x
		if self.downsample != 0:
			residual = self.block2(x)
		x = self.block1(x)
		x += residual
		return torch.nn.functional.relu(x)
