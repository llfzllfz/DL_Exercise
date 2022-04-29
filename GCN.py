import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

class GCN(nn.Module):
    def __init__(self, A, dim_in, dim_out):
        super(GCN, self).__init__()

        A = A + torch.eye(A.shape[0])
        D = torch.diag_embed(torch.pow(A.sum(dim=1), -0.5))
        self.A = D @ A @ D

        self.fc1 = nn.Linear(dim_in, dim_in, bias=False)
        self.fc2 = nn.Linear(dim_in, dim_in // 2, bias=False)
        self.fc3 = nn.Linear(dim_in // 2, dim_out, bias=False)

    def forward(self, X):
        X = F.relu(self.fc1(self.A @ X))
        X = F.relu(self.fc2(self.A @ X))
        return self.fc3(self.A @ X)

def train(model, X):
    opt = torch.optim.Adam(model.parameters())
    for epoch in range(150):
        model.train()
        y_pred = F.softmax(model(X), dim=1)
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

    model = GCN(A, dim_in=N, dim_out=2)
    train(model, feature)