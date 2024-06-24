import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 线性变换的权重
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # 注意力机制的参数
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj):
        # 应用线性变换
        Wh = torch.mm(h, self.W)  # (N, out_features)

        # 注意力机制的计算
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = torch.matmul(a_input, self.a).squeeze(-1)

        # 通过邻接矩阵与指数非线性化确保只计算实际存在的边的注意力系数
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, 0.6, training=self.training)

        # 用注意力权重加权节点特征
        h_prime = torch.matmul(attention, Wh)

        return F.elu(h_prime)

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh中的每一行是一个节点的特征向量
        N = Wh.size(0)  # 节点数量

        # 重复特征向量，准备按节点对应合并
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)

        # 每一行现在包含了两个节点特征的连接
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

# 定义GAT模型
class GAT(nn.Module):
    def __init__(self, num_features, num_classes):
        super(GAT, self).__init__()
        self.attention = GraphAttentionLayer(num_features, 8)
        self.out_att = GraphAttentionLayer(8, num_classes)

    def forward(self, x, adj):
        x = self.attention(x, adj)
        x = F.dropout(x, 0.6, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

# 初始化模型
model = GAT(num_features=3, num_classes=2)

# 假定的特征和邻接矩阵
features = torch.rand(5, 3)  # 5个节点，每个节点3个特征
adjacency = torch.FloatTensor([
    [0, 1, 0, 0, 0],
    [1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 1],
    [0, 0, 0, 1, 0]
])  # 一个简单的5节点图

# 模型前向传播
output = model(features, adjacency)
print("Output from GAT:\n", output)
