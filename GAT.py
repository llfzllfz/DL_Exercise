import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        
        self.W = nn.Parameter(torch.Tensor(in_features, out_features))
        self.a = nn.Parameter(torch.Tensor(2*out_features, 1))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # (N, out_features)
        Wh1 = torch.mm(Wh, self.a[:self.out_features, :])  # (N, 1)
        Wh2 = torch.mm(Wh, self.a[self.out_features:, :])  # (N, 1)
        e = self.leakyrelu(Wh1 + Wh2.T)                    # (N, N)
        padding = (-2 ** 31) * torch.ones_like(e)          # (N, N)
        attention = torch.where(adj > 0, e, padding)       # (N, N)
        attention = F.softmax(attention, dim=1)            # (N, N)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        h_prime = torch.matmul(attention, Wh)              # (N, out_features)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.MH = nn.ModuleList([
            GraphAttentionLayer(nfeat, nhid, dropout, alpha, concat=True)
            for _ in range(nheads)
        ])
        self.out_att = GraphAttentionLayer(nhid*nheads, nclass, dropout, alpha, concat=False)
        
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)    # (N, nfeat)
        x = torch.cat([head(x, adj) for head in self.MH], dim=1)  # (N, nheads*nhid)
        x = F.dropout(x, self.dropout, training=self.training)    # (N, nheads*nhid)
        x = F.elu(self.out_att(x, adj))
        return x

def train(model, A, X):
    opt = torch.optim.Adam(model.parameters())
    for epoch in range(1500):
        model.train()
        y_pred = F.softmax(model(A, X), dim=1)
        loss = (-y_pred.log().gather(1, label.view(-1, 1)))
        loss = loss.masked_select(label_mask).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        if epoch % 10 == 0:
            _, idx = y_pred.max(1)
            print('每个人的预测类别：', idx)
            print('ACC:{}\tLoss:{}' .format(float((idx == realY).sum()) / N, loss.item()))


if __name__ == '__main__':
    G = nx.karate_club_graph()
    A = torch.Tensor(nx.adjacency_matrix(G).todense())
    N = A.shape[0]
    feature = torch.eye(N)

    label = torch.zeros(N, 1).long()
    label[N - 1][0] = 1
    label_mask = torch.zeros(N, 1, dtype=torch.bool)
    label_mask[0][0] = 1
    label_mask[N - 1][0] = 1

    class0idx = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 16, 17, 19, 21]
    realY = torch.Tensor([0 if i in class0idx else 1 for i in range(N)])

    model = GAT(nclass = 2, nfeat = N, nhid = 100, dropout = 0.5, alpha = 0.1, nheads = 6)
    train(model, A, feature)