import torch
import torch.nn as nn
import torch.nn.functional as F
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        representations = torch.cat([z_i, z_j], dim=0)  # [2*B, D]
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = (torch.sum(torch.exp(similarity_matrix / self.temperature), dim=1) - torch.eye(2 * batch_size).to(z_i.device)).clamp(min=1e-8)

        loss = -torch.log(nominator / denominator).mean()
        return loss