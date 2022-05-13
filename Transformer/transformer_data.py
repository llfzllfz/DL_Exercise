import torch
import torch.utils.data as Data

sentences = [
    # enc_input
    ['Why do you want to drink coffee ? P', 'S 你 为什么 想要 喝 咖啡 ?', '你 为什么 想要 喝 咖啡 ? E']
]

src_vocab = {'P':0, 'Why':1, 'do':2, 'you':3, 'want':4, 'to':5, 'drink':6, 'coffee':7, '?':8}
src_vocab_size = len(src_vocab)
tgt_vocab = {'P':0, '你':1, '为什么':2, '想要':3, '喝':4, '咖啡':5, '?':6 ,'S':7, 'E':8}
tgt_vocab_size = len(tgt_vocab)
idx2word = {i:w for i, w in enumerate(tgt_vocab)}
print(idx2word)
def make_data(sentences):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
        enc_input = [[src_vocab[n] for n in sentences[i][0].split()]]
        dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]]
        dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]]
        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)
    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)
enc_inputs, dec_inputs, dec_outputs = make_data(sentences)

class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs
    def __len__(self):
        return self.enc_inputs.shape[0]
    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]

loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)