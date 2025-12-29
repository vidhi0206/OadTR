import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
import numpy as np
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import init
import math

class Linear_dr(nn.Module):

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dropout=0.0,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        self.register_buffer("P", self.compute_P(self.in_features, dropout))
        self.tr = 0
        self.count_infinite_tr = 0
        # self.l2_wq_T_wk_X_T = 0
        # self.l2_wq_T_wk = 0
        # self.l2_r_lambda = 0
        # self.l2_lambda = 0
        # self.l2_R = 0
        # self.X_T_X = 0
        # self.l2_P = 0

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        out= F.linear(input, self.weight, self.bias)
        self.tr = self.compute_dr_mlp(input)
        return out

    def compute_dr_mlp(self, X):
        batch_size, num_tokens, num_features = X.shape
        device = X.device

        X_T = X.transpose(1, 2)
        R = torch.bmm(X_T, X).to(device)
        # self.X_T_X = torch.norm(R, dim=(1, 2))[0]
        R.mul_(self.P)
        # self.l2_P = torch.norm(self.P)

        w = self.weight  # Shape: (num_features, num_features)
        # self.l2_wq_T_wk = torch.norm(w)
        Omega = w.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        Lambda = torch.bmm(Omega, Omega.transpose(1, 2))
        R_lambda = torch.bmm(R, Lambda)
        # self.l2_wq_T_wk_X_T = torch.norm(Omega, dim=(1, 2))[0]
        # self.l2_r_lambda = torch.norm(R_lambda, dim=(1, 2))[0]
        # self.l2_lambda = torch.norm(Lambda, dim=(1, 2))[0]
        # self.l2_R = torch.norm(R, dim=(1, 2))[0]
        count_infinite_tr = torch.isinf(R_lambda).any(dim=(1, 2)) | torch.isnan(
            R_lambda
        ).any(dim=(1, 2))
        self.count_infinite_tr += count_infinite_tr.sum().item()
        tr = torch.abs(torch.einsum("bii->b", R_lambda)) / num_tokens
        tr[count_infinite_tr] = 0  # Set trace to zero where infinite values are found
        return tr.sum()

    def compute_P(self, D, drop_ratio):
        P = np.full((D, D), drop_ratio**2)
        # p = np.array([drop_ratio] * D).reshape(-1, 1)
        # one = np.ones([D, 1])
        # I = np.eye(D)
        # P = torch.from_numpy(((p @ p.T) * (one @ one.T - I)) + ((p @ one.T) * I))
        return torch.from_numpy(P).float()

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
