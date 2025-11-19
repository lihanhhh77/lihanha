import random

import numpy as np
import torch
from torch.optim.optimizer import Optimizer, required

import math

episilon = 1e-8



class customAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        # orthogonal the weight of pointconvolution
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad)

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # if isinstance(p, StiefelParameter):
                if p.__class__.__name__ == 'StiefelParameter':
                    trans = orthogonal_projection(p.grad.data, p.data)
                    p.grad.data.copy_(trans)


        loss = super().step(closure) # W1=W0-\lambda*grad_rieman

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.__class__.__name__ == 'StiefelParameter':
                    Wt1=p.data.view(p.size()[0], -1)
                    # dir_tan = proj_tanX_stiefel(p.grad.data.view(p.size()[0], -1), p.data.view(p.size()[0], -1))
                    W_new = retraction(torch.zeros_like(Wt1), Wt1) # R_(W0) {-\lambda*grad_rieman} = qf(W0-\lambda*grad_rieman) =qf(W1)
                    p.data.copy_(W_new.view(p.size()))

        return loss


def symmetric(A):
    return 0.5 * (A + A.mT)

def retraction(A, ref):
    assert A.shape[0]>=A.shape[1]
    data = A + ref
    # Q, R = data.qr()
    Q,R = torch.linalg.qr(data)
    # To avoid (any possible) negative values in the output matrix, we multiply the negative values by -1
    sign = (R.diag().sign() + 0.5).sign().diag()
    out = Q.mm(sign)
    return out

def orthogonal_projection(A, B):
    assert A.shape[0] >= A.shape[1]
    out = A - B.mm(symmetric(B.transpose(0,1).mm(A)))
    return out
