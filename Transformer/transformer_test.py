import transformer
import transformer_data
import numpy as np
import torch
import torch.nn as nn

model = transformer.Transformer()
loss = nn.CrossEntropyLoss(ignore_index = 0)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)

def train(loader):
    for epoch in range(30):
        for enc_inputs, dec_inputs, dec_outputs in loader:
            enc_inputs, dec_inputs, dec_outputs = enc_inputs, dec_inputs, dec_outputs
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
            print(outputs, dec_outputs.view(-1))
            los = loss(outputs, dec_outputs.view(-1))
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(los))
            optimizer.zero_grad()
            los.backward()
            optimizer.step()

def test():
    pass

loader = transformer_data.loader
train(loader)