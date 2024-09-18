import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

# 读取节点特征数据
node_features_df = pd.read_csv('D:/xiangmu/multi_channel_PBR/dataset/1A34_A_centrality.txt', delimiter='\t', header=None)
node_features = torch.tensor(node_features_df.iloc[:, 1:].values, dtype=torch.float)

# 读取边列表数据
edges_df = pd.read_csv('D:/xiangmu/multi_channel_PBR/dataset/1A34_A_edgelist.txt', delimiter='\t', header=None)

# 建立节点到索引的映射
node_to_idx = {node: idx for idx, node in enumerate(node_features_df.iloc[:, 0])}

# 构建边的tensor
edge_list = [[node_to_idx[src], node_to_idx[dst]] for src, dst in edges_df.values]
edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

# 构建图数据
data = Data(x=node_features, edge_index=edge_index)

# 读取实际标签数据
labels_df = pd.read_csv('D:/xiangmu/multi_channel_PBR/dataset/1A34_A_label.txt', delimiter='\t', header=None)
actual_labels = torch.tensor(labels_df.iloc[:, 1].values, dtype=torch.float)

# 确保标签数据的大小与节点数一致
assert actual_labels.shape[0] == node_features.shape[0], "The number of labels must match the number of nodes."

# 定义图注意力网络模型
class GAT(torch.nn.Module):
    def __init__(self, num_features, embedding_dim):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, 8, heads=8, concat=True)       # 64 features output
        self.conv2 = GATConv(8 * 8, 8, heads=8, concat=True)              # 64 features output
        self.conv3 = GATConv(8 * 8, embedding_dim, heads=1, concat=False) # Output as embedding_dim

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)  # No activation function in the last layer (depends on the specific task)
        return x

# 初始化模型，embedding_dim 设置为 1024
model = GAT(num_features=node_features.shape[1], embedding_dim=1024)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# 训练模型
model.train()
for epoch in range(500):
    optimizer.zero_grad()
    out = model(data)
    # 通过降维（如线性层）将输出从1024维映射到标签维度
    out = torch.mean(out, dim=1)  # 将1024维的输出取平均，变为与标签匹配的形状

    # 使用实际标签数据进行训练
    loss = F.mse_loss(out, actual_labels)  # 确保有实际标签数据 actual_labels
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# 评估模型
model.eval()
with torch.no_grad():
    node_embeddings = model(data)
    df = pd.DataFrame(node_embeddings.numpy())

    # 保存 DataFrame 到 Excel 文件
    df.to_excel("D:/xiangmu/multi_channel_PBR/dataset/1A34_A_node_embeddings.xlsx", index=False)
