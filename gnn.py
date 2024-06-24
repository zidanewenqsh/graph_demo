import torch
import torch.nn as nn
import torch.nn.functional as F

# 假设有一个简单的图结构，包括4个节点和它们之间的边
num_nodes = 4
num_features = 3
num_classes = 2

# 随机生成节点特征
features = torch.rand(num_nodes, num_features)

# 模拟的邻接矩阵（指示节点间的连接关系）
adjacency = torch.tensor([
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0]
], dtype=torch.float)

# GCN层的实现
class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.rand(input_dim, output_dim))

    def forward(self, x, adj):
        # 矩阵乘法：邻接矩阵与特征矩阵
        support = torch.mm(x, self.weight)
        output = torch.mm(adj, support)
        return output

# 定义模型
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

# 模型前向传播
output = model(features, adjacency)
print(output)
