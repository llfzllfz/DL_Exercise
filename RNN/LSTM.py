import torch

class LSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(
            input_size = input_dim,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True
        )
        self.out = torch.nn.Linear(hidden_size, 10)

    def forward(self, x, h_c):
        out, h_c_ = self.lstm(x, h_c)
        return self.out(out[:, -1, :]), h_c_