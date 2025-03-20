import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch.nn import LayerNorm, Linear
import edge_dict

class fineRGCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim=256, out_dim=128, num_relations=1, dropout=0.3):
        super(fineRGCN, self).__init__()
        self.num_layers = 3
        self.dropout = dropout
        self.num_relations = num_relations

        # 为每种边类型定义一个独立的 GraphSAGE 层
        self.convs = torch.nn.ModuleList([
            SAGEConv(num_features, hidden_dim) for _ in range(num_relations)
        ])

        # 定义 LayerNorm 层
        self.ln1 = LayerNorm(hidden_dim)
        self.ln2 = LayerNorm(hidden_dim)
        self.ln_out = LayerNorm(out_dim)

        # 定义用于残差连接的线性层
        self.lin = Linear(num_features, hidden_dim)

        # 定义最后一层 GraphSAGE
        self.conv_final = SAGEConv(hidden_dim, out_dim)

    def forward(self, data, edge_types):
        x = data['patient'].x  # [num_nodes, num_features]
        res_x = self.lin(x)     # [num_nodes, hidden_dim]

        # Dropout
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 针对每种边类型应用不同的 GraphSAGE 层
        all_conv_outputs = []
        for i, etype in enumerate(edge_types):
            edge_index = edge_dict.edge_dict[etype]
            conv = self.convs[i]
            conv_out = conv(x, edge_index)
            all_conv_outputs.append(conv_out)

        # 将所有边类型的输出进行聚合（例如求和）
        x = torch.stack(all_conv_outputs, dim=0).sum(dim=0)  # [num_nodes, hidden_dim]

        # 激活和归一化
        x = F.silu(x)
        x = self.ln1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 残差连接
        x = res_x + x
        x = F.silu(x)
        x = self.ln2(x)

        # 合并边类型的 edge_index
        combined_edge_index = self.combine_edge_indices(edge_types)

        # 最后一层卷积，传入整个 combined_edge_index
        x = self.conv_final(x, combined_edge_index)

        # 最后的 LayerNorm
        x = self.ln_out(x)

        return x

    def combine_edge_indices(self, edge_types):
        """
        合并多个边类型的 edge_index。

        参数:
        - edge_types (list of str): 边类型列表。

        返回:
        - combined_edge_index (torch.Tensor): 合并后的 edge_index, 形状为 [2, E_total]。
        """
        edge_index_list = []
        for etype in edge_types:
            edge_index = edge_dict.edge_dict[etype]  # [2, E]
            if edge_index.dim() != 2 or edge_index.size(0) != 2:
                raise ValueError(f"Edge index for '{etype}' must be of shape [2, E], but got {edge_index.shape}")
            edge_index_list.append(edge_index)

        combined_edge_index = torch.cat(edge_index_list, dim=1)  # [2, E1 + E2 + E3]
        return combined_edge_index

