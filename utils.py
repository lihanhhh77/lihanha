import torch.nn as nn
import torch
import scipy.signal as signal
import numpy as np
import copy
import math
import random
from torch.autograd import Function
from torch.nn import functional as F
from torch.functional import Tensor
from typing import Callable, Tuple
from typing import Any

dtype=torch.float32

EPS = {torch.float32: 1e-6, torch.float64: 1e-9}

def qr_algorithm(A, max_iter=5, tol=1e-3):
    n = A.shape[1]
    A_k = A.clone().to(A.device)

    Q_total = torch.eye(n, device=A.device)
    for i in range(max_iter):
        #  QR decomposition
        Q, R = torch.linalg.qr(A_k)
        A_k = R @ Q
        Q_total = Q_total @ Q

        # off_diag_sum = torch.sum(abs(A_k - torch.diag_embed(torch.diagonal(A_k,dim1=1,dim2=2))))
        # if off_diag_sum < tol:
        #     break
    # s=torch.diagonal(A_k,dim1=1,dim2=2)
    # U=Q_total
    # res=U @ torch.diag_embed(s) @ U.mT
    return torch.diagonal(A_k,dim1=1,dim2=2), Q_total

def ensure_sym(A: Tensor) -> Tensor:
    """Ensures that the last two dimensions of the tensor are symmetric.
    Parameters
    ----------
    A : torch.Tensor
        with the last two dimensions being identical
    -------
    Returns : torch.Tensor
    """
    return 0.5 * (A + A.transpose(-1,-2))


class StiefelParameter(nn.Parameter):
    """A kind of Variable that is to be considered a module parameter on the space of
        Stiefel manifold.
    """
    def __new__(cls, data=None, requires_grad=True):
        return super(StiefelParameter, cls).__new__(cls, data, requires_grad=requires_grad)

    def __repr__(self):
        return 'Parameter containing:' + self.data.__repr__()


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)



class SUMlayer(nn.Module):
    def forward(self, *x):

        return sum(*x)
    def __repr__(self): return f'{self.__class__.__name__}'


class Concat(nn.Module):
    def __init__(self, dim=1):
        super(Concat, self).__init__()
        self.dim = dim
    def forward(self, *x): return torch.cat(*x, dim=self.dim)
    def __repr__(self): return f'{self.__class__.__name__}(dim={self.dim})'

class Patch(nn.Module):
    def __init__(self,seq_len, patch_len, stride):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.num_patch = (max(seq_len, patch_len)-patch_len) // stride + 1
        tgt_len = patch_len  + stride*(self.num_patch-1)
        self.s_begin = seq_len - tgt_len

    def forward(self, x):
        """
        x: [bs x seq_len x n_vars]
        """
        x = x[:, self.s_begin:, :]
        x = x.unfold(dimension=1, size=self.patch_len, step=self.stride)                 # xb: [bs x num_patch x n_vars x patch_len]
        return x





class LayerNormalization(nn.Module):

    def __init__(self,
                 normal_shape,
                 gamma=True,
                 beta=True,
                 epsilon=1e-6):
        """Layer normalization layer

        See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)

        :param normal_shape: The shape of the input tensor or the last dimension of the input tensor.
        :param gamma: Add a scale parameter if it is True.
        :param beta: Add an offset parameter if it is True.
        :param epsilon: Epsilon for calculating variance.
        """
        super(LayerNormalization, self).__init__()
        if isinstance(normal_shape, int):
            normal_shape = (normal_shape,)
        else:
            normal_shape = (normal_shape[-1],)
        self.normal_shape = torch.Size(normal_shape)
        self.epsilon = epsilon
        if gamma:
            self.gamma = nn.Parameter(torch.ones(*normal_shape))
        else:
            self.register_parameter('gamma', None)
        if beta:
            self.beta = nn.Parameter(torch.zeros(*normal_shape))
        else:
            self.register_parameter('beta', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.gamma is not None:
            self.gamma.data.fill_(1)
        if self.beta is not None:
            self.beta.data.zero_()

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        x = (x - mean) / (torch.sqrt(var + 1e-6))
        if self.gamma is not None:
            x *= self.gamma.expand_as(x)
        if self.beta is not None:
            x += self.beta.expand_as(x)
        return x

    def extra_repr(self):
        return 'normal_shape={}, gamma={}, beta={}, epsilon={}'.format(
            self.normal_shape, self.gamma is not None, self.beta is not None, self.epsilon,
        )

class AffineInvariantLayer(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, momentum: float = 0.1):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum

        self.register_buffer("running_M_inv_sqrt", torch.eye(dim))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def inv_sqrtm(self, M):
        """计算矩阵的逆平方根"""
        n = M.size(-1)
        I = torch.eye(n, device=M.device, dtype=M.dtype)
        M_reg = M + self.eps * I  #

        # 特征分解
        s, U = torch.linalg.eigh(M_reg)
        s_abs = s.abs().clamp_min(self.eps)  #
        s_inv_sqrt = 1.0 / torch.sqrt(s_abs)

        # 重建逆平方根矩阵
        return U @ torch.diag_embed(s_inv_sqrt) @ U.transpose(-1, -2)

    def forward(self, cov_input: torch.Tensor) -> torch.Tensor:
        if self.training:
            M_batch = cov_input.mean(0)
            M_inv_sqrt_batch = self.inv_sqrtm(M_batch)

            with torch.no_grad():
                m = self.momentum
                self.running_M_inv_sqrt = (1 - m) * self.running_M_inv_sqrt + m * M_inv_sqrt_batch
                self.num_batches_tracked += 1

            M_inv_sqrt = M_inv_sqrt_batch
        else:
            M_inv_sqrt = self.running_M_inv_sqrt

        W = torch.einsum("ij,bjk,kl->bil", M_inv_sqrt, cov_input, M_inv_sqrt)

        gamma_diag = torch.diag_embed(self.gamma)  # (dim, dim)
        scaled_W = gamma_diag @ W @ gamma_diag

        return scaled_W



class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        B, C, H, W = x.shape
        return self.pe[:, :W].transpose(-1, -2).unsqueeze(2)
class PCOM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.alpha1 = nn.Parameter(torch.tensor(0.5))
        self.alpha2 = nn.Parameter(torch.tensor(0.5))
        self.alpha3 = nn.Parameter(torch.tensor(0.5))
        self.fuse_weights = nn.Parameter(torch.tensor([0.5, 0.25, 0.25]))  # Control the proportion of the fusion of the three branches


        self.PE = PositionalEmbedding(dim)
    def _compute_cov(self, x, alpha, regularization_type):
        B, C, _, T = x.shape
        pe = self.PE(x)
        x = x + pe
        x = x.squeeze(2)  # [B, C, T]
        x = x - x.mean(dim=2, keepdim=True)
        cov_sample = torch.bmm(x, x.transpose(1, 2)) / (T - 1)

        if regularization_type == 'identity':
            identity = torch.eye(C, device=x.device).expand(B, C, C)
            cov_matrix = alpha * cov_sample + (1 - alpha) * identity
        elif regularization_type == 'frobenius':
            cov_matrix = cov_sample.clone()
            frob_norm = torch.norm(cov_matrix, p='fro', dim=(1, 2), keepdim=True)
            cov_matrix = F.relu(cov_matrix / frob_norm - (1 - alpha))
        elif regularization_type == 'l2':
            cov_matrix = cov_sample.clone()
            l2_norm = torch.norm(cov_matrix, p=2, dim=(1, 2), keepdim=True)
            cov_matrix = cov_matrix / (l2_norm + 1e-6)
        else:
            raise ValueError("Unsupported regularization type")

        return cov_matrix

    def forward(self, x):
        branch1 = self._compute_cov(x, torch.sigmoid(self.alpha1), regularization_type='identity')
        branch2 = self._compute_cov(x, torch.sigmoid(self.alpha2), regularization_type='frobenius')
        branch3 = self._compute_cov(x, torch.sigmoid(self.alpha3), regularization_type='l2')

        # Use learnable parameters to fuse the three covariance matrices
        fuse_weights = F.softmax(self.fuse_weights, dim=0)
        fused_cov = fuse_weights[0] * branch1 + fuse_weights[1] * branch2 + fuse_weights[2] * branch3

        return fused_cov
class Stie_W(nn.Module):
    def __init__(self, input_dim,output_dim):
        super(Stie_W, self).__init__()
        assert  input_dim>=output_dim
        self.output_dim=output_dim
        geoopt_used=False
        # if geoopt_used:
        #     manifold = geoopt.Stiefel(canonical=False)
        #     self.weight = geoopt.ManifoldParameter(manifold.random(input_dim, output_dim), manifold=manifold,
        #                                            requires_grad=True)
        # else:
        self.weight = StiefelParameter(torch.FloatTensor(input_dim,output_dim), requires_grad=True)
        nn.init.orthogonal_(self.weight)

    def forward(self, x):
        B,C,H,W=x.shape
        input=x.reshape(B,C,H*W)
        output=torch.matmul(self.weight.t(),input)
        output=output.reshape(B,self.output_dim,H,W)
        return output


class PointwiseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PointwiseConv2d,self).__init__()
        self.weight = StiefelParameter(torch.FloatTensor(in_channels,out_channels), requires_grad=True)
        self.bias =  None
        self.stride = 1
        self.padding = 0
        self.in_chan=in_channels
        self.out_chan=out_channels
        nn.init.orthogonal_(self.weight)

    def forward(self, x):
        weight=self.weight.t().reshape(self.out_chan,self.in_chan,1,1)
        out= F.conv2d(x, weight, self.bias, stride=self.stride, padding=self.padding)
        return out

class PointwiseConv1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weight = StiefelParameter(torch.FloatTensor(in_channels,out_channels), requires_grad=True)
        self.bias =  None
        self.stride = 1
        self.padding = 0
        self.in_chan=in_channels
        self.out_chan=out_channels
        nn.init.orthogonal_(self.weight)

    def forward(self, x):
        weight=self.weight.t().reshape(self.out_chan,self.in_chan,1)
        out= F.conv1d(x, weight, self.bias, stride=self.stride, padding=self.padding)
        return out

class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.weight = StiefelParameter(torch.FloatTensor(in_features,out_features), requires_grad=True)
        self.bias = None
        nn.init.orthogonal_(self.weight)

    def forward(self, x):
        weight=self.weight.t()
        return F.linear(x, weight, self.bias)


class sym_modeig:
    """Basic class that modifies the eigenvalues with an arbitrary elementwise function
    """

    @staticmethod
    def forward(M : Tensor, fun : Callable[[Tensor], Tensor], fun_param : Tensor = None,
                ensure_symmetric : bool = False, ensure_psd : bool = False) -> Tensor:

        # if ensure_symmetric:
        #     M = ensure_sym(M)

        # compute the eigenvalues and vectors
        # U, s, vt = torch.linalg.svd(M)
        s,U = qr_algorithm(M)
        if ensure_psd:
            s = s.clamp(min=EPS[s.dtype])
        # modify the eigenvalues
        smod = fun(s, fun_param)
        X = U @ torch.diag_embed(smod) @ U.transpose(-1,-2)

        return X, s, smod, U

    @staticmethod
    def backward(dX : Tensor, s : Tensor, smod : Tensor, U : Tensor,
                    fun_der : Callable[[Tensor], Tensor], fun_der_param : Tensor = None) -> Tensor:
        """Backpropagates the derivatives



        Parameters
        ----------
        dX : torch.Tensor
            (batch) derivatives that should be backpropagated
        s : torch.Tensor
            eigenvalues of the original input
        smod : torch.Tensor
            modified eigenvalues
        U : torch.Tensor
            eigenvector of the input
        fun_der : Callable[[Tensor], Tensor]
            elementwise function derivative
        -------
        Returns : torch.Tensor containing the backpropagated derivatives
        """

        # compute Lowener matrix
        # denominator
        L_den = s[...,None] - s[...,None].transpose(-1,-2)
        # find cases (similar or different eigenvalues, via threshold)
        is_eq = L_den.abs() < EPS[s.dtype]
        L_den[is_eq] = 1.0
        # case: sigma_i != sigma_j
        L_num_ne = smod[...,None] - smod[...,None].transpose(-1,-2)
        L_num_ne[is_eq] = 0
        # case: sigma_i == sigma_j
        sder = fun_der(s, fun_der_param)
        L_num_eq = 0.5 * (sder[...,None] + sder[...,None].transpose(-1,-2))
        L_num_eq[~is_eq] = 0
        # compose Loewner matrix
        L = (L_num_ne + L_num_eq) / L_den
        dM = U @  (L * (U.transpose(-1,-2) @ ensure_sym(dX) @ U)) @ U.transpose(-1,-2)
        return dM


class sym_logm(Function):
    """
    Computes the matrix logarithm for a batch of SPD matrices.
    Ensures that the input matrices are SPD by clamping eigenvalues.
    During backprop, the update along the clamped eigenvalues is zeroed
    """
    @staticmethod
    def value(s : Tensor, param:Tensor = None) -> Tensor:
        # ensure that the eigenvalues are positive
        return s.clamp(min=EPS[s.dtype]).log()

    @staticmethod
    def derivative(s : Tensor, param:Tensor = None) -> Tensor:
        # compute derivative
        sder = s.reciprocal()
        # pick subgradient 0 for clamped eigenvalues
        sder[s<=EPS[s.dtype]] = 0
        return sder

    @staticmethod
    def forward(ctx: Any, M: Tensor, ensure_symmetric : bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_logm.value, ensure_symmetric=ensure_symmetric)
        ctx.save_for_backward(s, smod, U)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U = ctx.saved_tensors
        return sym_modeig.backward(dX, s, smod, U, sym_logm.derivative), None


class LogmLayer(nn.Module):

    def __init__(self, input_size, vectorize=False):
        super(LogmLayer, self).__init__()
        self.vectorize = vectorize

    def forward(self, input):
        output=sym_logm.apply(input)
        return output




class Vec(nn.Module):
    def __init__(self, input_size):
        super(Vec, self).__init__()
        mask = torch.triu(torch.ones([input_size,input_size], dtype=torch.bool), diagonal=0)
        self.register_buffer('mask', mask)

    def forward(self, input):
        output = input[..., self.mask]
        return output

