from transformers import BertTokenizer, BertModel, BertConfig
import torch
import torch.nn as nn
import os
import torch.utils.data as Data

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        config = BertConfig.from_pretrained('bert-base-uncased')
        self.emb = BertModel.from_pretrained('bert-base-uncased', config=config)
    
    def forward(self, txt, token, masked = None):
        bert = self.emb(txt, token, masked)
        embed, _ = bert
        cls_vector = bert[embed][:, 0, :]
        cls_vector = cls_vector.view(-1, 768)
        output = nn.Linear(768, 2)(cls_vector)
        return output

data_path = 'IMDB/aclImdb/train'
def get_data(datapath):
    max_len = 510
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    pos_files = os.listdir(os.path.join(datapath, 'pos'))[:100]
    neg_files = os.listdir(os.path.join(datapath, 'neg'))[:100]
    txt_all = []
    txt_token = []
    txt_mask = []
    labels = [0]*100 + [1] * 100
    for filename in pos_files:
        with open(os.path.join(os.path.join(datapath, 'pos', filename)), encoding='utf-8') as f:
            s = f.read()
            s = s[:max_len]
            s = tokenizer.encode(s)
            txt_token.append([1] * len(s) + [0] * (512 - len(s)))
            txt_mask.append([1] * len(s) + [0] * (512 - len(s)))
            txt_all.append(s + [0] * (512 - len(s)))
        f.close()
    for filename in neg_files:
        with open(os.path.join(os.path.join(datapath, 'neg', filename)), encoding='utf-8') as f:
            s = f.read()
            s = s[:max_len]
            s = tokenizer.encode(s)
            txt_token.append([1] * len(s) + [0] * (512 - len(s)))
            txt_mask.append([1] * len(s) + [0] * (512 - len(s)))
            txt_all.append(s + [0] * (512 - len(s)))
        f.close()
    return txt_all, txt_token, txt_mask, labels

txt_all, txt_token, txt_mask, labels = get_data(data_path)
# print(txt_all[1])
# print(txt_token[1])
# print(txt_mask[1])
# print(len(txt_all[0]), len(txt_token[0]), len(txt_mask[0]))
class MyData(Data.Dataset):
    def __init__(self, txt_all, txt_token, txt_mask, labels):
        super(MyData, self).__init__()
        self.txt_all = txt_all
        self.txt_token = txt_token
        self.txt_mask = txt_mask
        self.labels = labels
    
    def __len__(self):
        return len(self.txt_all)
    
    def __getitem__(self, idx):
        return torch.tensor(self.txt_all[idx]), torch.tensor(self.txt_token[idx]), torch.tensor(self.txt_mask[idx]), torch.tensor(self.labels[idx])

my_dataset = MyData(txt_all, txt_token, txt_mask, labels)
loader = torch.utils.data.DataLoader(my_dataset, batch_size = 2)

model = Model()
loss = nn.CrossEntropyLoss()
print(model.parameters())
optim = torch.optim.Adam(model.parameters(), 0.001)
for epoch in range(30):
    correct = 0
    for txt, token, mask, label in loader:
        output = model(txt, token, mask)
        pred = output.max(1)[1]
        correct += pred.eq(label.long()).sum().item()
        los = loss(output, label.long())
        optim.zero_grad()
        los.backward()
        optim.step()
        print('Correct:{}'.format(correct))
    print('Epoch:{}\tLoss:{}\tAcc:{}'.format(epoch, los, correct / len(labels)))
