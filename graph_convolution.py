import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义图结构
num_nodes = 4
num_features = 3
num_classes = 2

# 随机生成节点特征
features = torch.rand(num_nodes, num_features)

# 原始邻接矩阵（模拟数据）
adjacency = torch.tensor([
    [0, 1, 0, 0],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0]
], dtype=torch.float)

# 添加自环
I = torch.eye(num_nodes)
A_hat = adjacency + I

# 计算度矩阵D_hat
D_hat = torch.diag(torch.sum(A_hat, dim=1))

# 计算D_hat^(-1/2)
D_hat_inv_sqrt = torch.diag(torch.pow(D_hat.diagonal(), -0.5))

# 计算标准化邻接矩阵
norm_adj = torch.mm(torch.mm(D_hat_inv_sqrt, A_hat), D_hat_inv_sqrt)

# 定义图卷积层
class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.rand(input_dim, output_dim))

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.mm(adj, support)
        return output

# 定义GCN模型
class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GraphConvolution(num_features, 5)
        self.conv2 = GraphConvolution(5, num_classes)

    def forward(self, x, adj):
        x = self.conv1(x, adj)
        x = F.relu(x)
        x = self.conv2(x, adj)
        return F.log_softmax(x, dim=1)

# 初始化模型
model = GCN()

# 模型前向传播，使用标准化邻接矩阵
output = model(features, norm_adj)
print(output)
