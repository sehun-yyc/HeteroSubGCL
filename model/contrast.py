import torch
import torch.nn as nn
from method.clusters import  clusters

class Contrast(nn.Module):
    def __init__(self, hidden_dim, tau, lam):
        super(Contrast, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tau = tau
        self.lam = lam
        for layer in self.proj:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=1.414)
    
    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix
    
    def forward(self, coarse, medium1, medium2, medium3, fine1, fine2, fine3, pos):
        # embeds 是一个字典，包含不同粒度的嵌入
        z_coarse = self.proj(coarse)
        z_medium1 = self.proj(medium1)
        z_medium2 = self.proj(medium2)
        z_medium3 = self.proj(medium3)
        z_fine1 = self.proj(fine1)
        z_fine2 = self.proj(fine2)
        z_fine3 = self.proj(fine3)
        
        # 定义对比对，例如：
        # 粗粒度 vs 中等细粒度1, 2, 3
        # 中等细粒度1 vs 细粒度1, 2
        # 中等细粒度2 vs 细粒度1, 3
        # 中等细粒度3 vs 细粒度2, 3
        # 具体根据需求调整
        loss = 0.0
        pairs = [
            (z_coarse, z_medium1),
            (z_coarse, z_medium2),
            (z_coarse, z_medium3),
            (z_medium1, z_fine1),
            (z_medium1, z_fine2),
            (z_medium1, z_fine3),
            (z_medium2, z_fine1),
            (z_medium2, z_fine2),
            (z_medium2, z_fine3),
            (z_medium3, z_fine1),
            (z_medium3, z_fine2),
            (z_medium3, z_fine3),
        ]
        
        for (level1, level2) in pairs:
            z1 = level1
            z2 = level2
            sim_matrix1_2 = self.sim(z1, z2)
            sim_matrix2_1 = sim_matrix1_2.t()

            sim_matrix1_2 = sim_matrix1_2 / (torch.sum(sim_matrix1_2, dim=1).view(-1, 1) + 1e-8)
            lori1_2 = (-torch.log(sim_matrix1_2.mul(pos).sum(dim=-1)).mean())*self.lam

            sim_matrix2_1 = sim_matrix2_1 / (torch.sum(sim_matrix2_1, dim=1).view(-1, 1) + 1e-8)
            lori2_1 = (-torch.log(sim_matrix2_1.mul(pos).sum(dim=-1)).mean())*(1-self.lam)
            
            loss += (lori1_2 + lori2_1)
        
        return loss
