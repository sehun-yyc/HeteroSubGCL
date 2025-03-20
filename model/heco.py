# 
# model/heco.py

import torch.nn as nn
import torch
import torch.nn.functional as F
from .Methy_model import MethyGCN
from .Mirna_model import MirnaGCN
from .Gene_model import GeneGCN
from .coarse_model import coarseRGCN
from .fine_model import fineRGCN
from .medium_model import mediumRGCN
from .transformer_module import GraphTransformer

class HeCo(nn.Module):
    def __init__(self, num_feature_coarse, num_feature_medium1, num_feature_medium2, num_feature_medium3, 
                num_feature_fine1, num_feature_fine2, num_feature_fine3):
        super(HeCo, self).__init__()
        self.coarse_encoder = coarseRGCN(num_feature_coarse)
        self.medium1_encoder = mediumRGCN(num_feature_medium1)
        self.medium2_encoder = mediumRGCN(num_feature_medium2)
        self.medium3_encoder = mediumRGCN(num_feature_medium3)
        self.fine1_encoder = fineRGCN(num_feature_fine1)
        self.fine2_encoder = fineRGCN(num_feature_fine2)
        self.fine3_encoder = fineRGCN(num_feature_fine3)

        # # 为每个视图分别定义Transformer
        # self.transformers = nn.ModuleDict({
        #     'coarse': GraphTransformer(dim_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1),
        #     'medium1': GraphTransformer(dim_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1),
        #     'medium2': GraphTransformer(dim_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1),
        #     'medium3': GraphTransformer(dim_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1),
        #     'fine1': GraphTransformer(dim_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1),
        #     'fine2': GraphTransformer(dim_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1),
        #     'fine3': GraphTransformer(dim_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1),
        # })

    def encode_view(self, encoder, data, edge_types):
        z = encoder(data, edge_types)
        z = F.silu(z)
        return z

    def forward(self, data_coarse, data_medium1, data_medium2, data_medium3, data_fine1, data_fine2, data_fine3):
        # 编码每个视图
        
        z_coarse = self.encode_view(self.coarse_encoder, data_coarse, ['gene_rel', 'methy_rel', 'mirna_rel'])
        z_medium1 = self.encode_view(self.medium1_encoder, data_medium1, ['gene_rel', 'methy_rel'])
        z_medium2 = self.encode_view(self.medium2_encoder, data_medium2, ['gene_rel', 'mirna_rel'])
        z_medium3 = self.encode_view(self.medium3_encoder, data_medium3, ['methy_rel', 'mirna_rel'])
        z_fine1 = self.encode_view(self.fine1_encoder, data_fine1, ['gene_rel'])
        z_fine2 = self.encode_view(self.fine2_encoder, data_fine2, ['methy_rel'])
        z_fine3 = self.encode_view(self.fine3_encoder, data_fine3, ['mirna_rel'])

        # # 分别通过各自的Transformer
        # z_coarse = self.transformers['coarse'](z_coarse)
        # z_medium1 = self.transformers['medium1'](z_medium1)
        # z_medium2 = self.transformers['medium2'](z_medium2)
        # z_medium3 = self.transformers['medium3'](z_medium3)
        # z_fine1 = self.transformers['fine1'](z_fine1)
        # z_fine2 = self.transformers['fine2'](z_fine2)
        # z_fine3 = self.transformers['fine3'](z_fine3)

        return {
            'coarse': z_coarse,
            'medium1': z_medium1,
            'medium2': z_medium2,
            'medium3': z_medium3,
            'fine1': z_fine1,
            'fine2': z_fine2,
            'fine3': z_fine3
        }

    def get_embeds(self, data_coarse, data_medium1, data_medium2, data_medium3, data_fine1, data_fine2, data_fine3):
        embeddings = self.forward(data_coarse, data_medium1, data_medium2, data_medium3, data_fine1, data_fine2, data_fine3)
        return embeddings

