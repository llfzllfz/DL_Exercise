import torch

class VGG(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = self.conv_2(3, 64, 3, 1, 1, 2, 2, 0)
        self.conv2 = self.conv_2(64, 128, 3, 1, 1, 2, 2, 0)
        self.conv3 = self.conv_3(128, 256, 3, 1, 1, 2, 2, 0)
        self.conv4 = self.conv_3(256, 512, 3, 1, 1, 2, 2, 0)
        self.conv5 = self.conv_3(512, 512, 3, 1, 1, 2, 2, 0)
        self.fc = torch.nn.Sequential(
        	torch.nn.LazyLinear(4096),
        	torch.nn.ReLU(),
        	torch.nn.Dropout(0.5),
        	torch.nn.Linear(4096, 4096),
        	torch.nn.ReLU(),
        	torch.nn.Dropout(0.5),
        	torch.nn.Linear(4096, 10)
        )

    def forward(self, x):
    	x = self.conv1(x)
    	x = self.conv2(x)
    	x = self.conv3(x)
    	x = self.conv4(x)
    	x = self.conv5(x)
    	x = x.view(x.size()[0], -1)
    	x = self.fc(x)
    	return x

    def conv_2(self, input_channels, output_channels, c_kernel_size, c_stride, c_padding, p_kernel_size, p_stride, p_padding):
    	return torch.nn.Sequential(
    		torch.nn.Conv2d(input_channels, output_channels, kernel_size = c_kernel_size, stride = c_stride, padding = p_padding),
    		torch.nn.ReLU(),
    		torch.nn.Conv2d(output_channels, output_channels, kernel_size = c_kernel_size, stride = c_stride, padding = p_padding),
    		torch.nn.ReLU(),
    		torch.nn.MaxPool2d(kernel_size = p_kernel_size, stride = p_stride, padding = p_padding)
    	)

    def conv_3(self, input_channels, output_channels, c_kernel_size, c_stride, c_padding, p_kernel_size, p_stride, p_padding):
    	return torch.nn.Sequential(
    		torch.nn.Conv2d(input_channels, output_channels, kernel_size = c_kernel_size, stride = c_stride, padding = p_padding),
    		torch.nn.ReLU(),
    		torch.nn.Conv2d(output_channels, output_channels, kernel_size = c_kernel_size, stride = c_stride, padding = p_padding),
    		torch.nn.ReLU(),
    		torch.nn.Conv2d(output_channels, output_channels, kernel_size = c_kernel_size, stride = c_stride, padding = p_padding),
    		torch.nn.ReLU(),
    		torch.nn.MaxPool2d(kernel_size = p_kernel_size, stride = p_stride, padding = p_padding)
    	)