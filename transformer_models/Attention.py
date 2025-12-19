import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Linear_dr import Linear_dr
import numpy as np


class SelfAttention(nn.Module):
    def __init__(
        self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0, drop_mha="drop_none", dr_mlp_mode=0,
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

        self.drop_mha = drop_mha
        self.dr_mlp_mode = dr_mlp_mode

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)
        if self.drop_mha=="drop_k":
            k = self.attn_drop(k)
        elif self.drop_mha=="drop_q":
            q = self.attn_drop(q) 
        elif self.drop_mha=="drop_v":
            v = self.attn_drop(v)
        elif self.drop_mha=="drop_k&q":
            q = self.attn_drop(q)
            k = self.attn_drop(k)
        # ---------------------------------------------
        #                DROPKEY
        # ---------------------------------------------
        if self.drop_mha=="dropkey":
            score = (q * self.scale) @ k.transpose(-2, -1)
            m_r = torch.ones_like(score) * self.drop_ratio
            score = score + torch.bernoulli(m_r) * -1e12
            score = score.softmax(dim=-1)
            attn = score @ v
            x = attn.transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x
        # ---------------------------------------------
        #                DROP ATTENTION
        # ---------------------------------------------
        if self.drop_mha=="drop_attn":
            score = (q * self.scale) @ k.transpose(-2, -1)
            score = score.softmax(dim=-1)
            m_r = torch.ones_like(score) * self.drop_ratio
            score = score * torch.bernoulli(m_r)
            row_sum = score.sum(dim=-1, keepdim=True)
            score = score / row_sum
            attn = score @ v
            x = attn.transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

        # ---------------------------------------------
        #                 DROP OUT
        # ---------------------------------------------
        if self.drop_mha=="drop_out":
            score = (q * self.scale) @ k.transpose(-2, -1)
            score = score.softmax(dim=-1)
            attn = score @ v
            x = attn.transpose(1, 2).reshape(B, N, C)
            x = self.proj_drop(self.proj(x))
            return x

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x
class SelfAttentionDr(nn.Module):
    def __init__(
        self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0, dr_mha="none", dr_mlp_mode=0,
    ):
        super().__init__()
        
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj_dr = Linear_dr(dim, dim,dropout_rate)
        #self.proj_drop = nn.Dropout(dropout_rate)
        
        self.dr_mha = dr_mha
        self.dr_mlp_mode = dr_mlp_mode

        self.register_buffer("P", self.compute_P(dim, dropout_rate))
        self.tr = torch.zeros(1)
        self.count_infinite_tr = torch.zeros(1)
        self.l2_wq_T_wk_X_T = torch.zeros(1)
        self.l2_wq_T_wk = torch.zeros(1)
        self.l2_r_lambda = torch.zeros(1)
        self.l2_lambda = torch.zeros(1)
        self.l2_R = torch.zeros(1)
        self.X_T_X = torch.zeros(1)
        self.l2_P = torch.zeros(1)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        if self.dr_mha == "Q":
            self.tr=self.compute_dr_Q(x)
        elif self.dr_mha == "K":
            self.tr=self.compute_dr_K(x)
        elif self.dr_mha == "V":
            self.tr=self.compute_dr_V(x)
        else:
            self.tr=torch.zeros(1)
        x = self.proj_dr(x)
        #x = self.proj_drop(x)
        return x
    
    def compute_dr_Q(self, X):
        batch_size, num_tokens, C = X.shape
        device = X.device

        X_T = X.transpose(1, 2)
        R = torch.bmm(X_T, X).to(device)
        self.X_T_X = torch.norm(R, dim=(1, 2))[0]
        R.mul_(self.P)
        self.l2_P = torch.norm(self.P)
        wq = self.qkv.weight[:C, :]
        wk = self.qkv.weight[C:2*C, :]  # Shape: (num_features, num_features)
        lambda1 = wq.T @ wk  # Shape: (num_features, num_features)
        self.l2_wq_T_wk = torch.norm(lambda1)
        Omega = torch.bmm(lambda1.unsqueeze(0).expand(batch_size, -1, -1), X_T).to(
            device
        )
        Lambda = torch.bmm(Omega, Omega.transpose(1, 2))
        R_lambda = torch.bmm(R, Lambda)
        self.l2_wq_T_wk_X_T = torch.norm(Omega, dim=(1, 2))[0]
        self.l2_r_lambda = torch.norm(R_lambda, dim=(1, 2))[0]
        self.l2_lambda = torch.norm(Lambda, dim=(1, 2))[0]
        self.l2_R = torch.norm(R, dim=(1, 2))[0]
        count_infinite_tr = torch.isinf(R_lambda).any(dim=(1, 2)) | torch.isnan(
            R_lambda
        ).any(dim=(1, 2))
        self.count_infinite_tr += count_infinite_tr.sum().item()
        tr = torch.abs(torch.einsum("bii->b", R_lambda)) / num_tokens
        tr[count_infinite_tr] = 0  # Set trace to zero where infinite values are found
        return tr.sum()
        

    def compute_dr_K(self, X):
        batch_size, num_tokens, C = X.shape
        device = X.device

        X_T = X.transpose(1, 2)
        R = torch.bmm(X_T, X).to(device)
        self.X_T_X = torch.norm(R, dim=(1, 2))[0]
        R.mul_(self.P)
        self.l2_P = torch.norm(self.P)
        wq = self.qkv.weight[:C, :]
        wk = self.qkv.weight[C:2*C, :]  # Shape: (num_features, num_features)
        lambda1 = wk.T @ wq  # Shape: (num_features, num_features)
        self.l2_wq_T_wk = torch.norm(lambda1)
        Omega = torch.bmm(lambda1.unsqueeze(0).expand(batch_size, -1, -1), X_T).to(
            device
        )
        Lambda = torch.bmm(Omega, Omega.transpose(1, 2))
        R_lambda = torch.bmm(R, Lambda)
        self.l2_wq_T_wk_X_T = torch.norm(Omega, dim=(1, 2))[0]
        self.l2_r_lambda = torch.norm(R_lambda, dim=(1, 2))[0]
        self.l2_lambda = torch.norm(Lambda, dim=(1, 2))[0]
        self.l2_R = torch.norm(R, dim=(1, 2))[0]
        count_infinite_tr = torch.isinf(R_lambda).any(dim=(1, 2)) | torch.isnan(
            R_lambda
        ).any(dim=(1, 2))
        self.count_infinite_tr += count_infinite_tr.sum().item()
        tr = torch.abs(torch.einsum("bii->b", R_lambda)) / num_tokens
        tr[count_infinite_tr] = 0  # Set trace to zero where infinite values are found
        return tr.sum()

    def compute_dr_V(self, X):
        batch_size, num_tokens, C = X.shape
        device = X.device

        X_T = X.transpose(1, 2)
        R = torch.bmm(X_T, X).to(device)
        self.X_T_X = torch.norm(R, dim=(1, 2))[0]
        R.mul_(self.P)
        self.l2_P = torch.norm(self.P)

        wv =self.qkv.weight[2*C:3*C, :] # Shape: (num_features, num_features)
        self.l2_wq_T_wk = torch.norm(wv)
        Omega = wv.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        Lambda = torch.bmm(Omega, Omega.transpose(1, 2))
        R_lambda = torch.bmm(R, Lambda)
        self.l2_wq_T_wk_X_T = torch.norm(Omega, dim=(1, 2))[0]
        self.l2_r_lambda = torch.norm(R_lambda, dim=(1, 2))[0]
        self.l2_lambda = torch.norm(Lambda, dim=(1, 2))[0]
        self.l2_R = torch.norm(R, dim=(1, 2))[0]
        count_infinite_tr = torch.isinf(R_lambda).any(dim=(1, 2)) | torch.isnan(
            R_lambda
        ).any(dim=(1, 2))
        self.count_infinite_tr += count_infinite_tr.sum().item()
        tr = torch.abs(torch.einsum("bii->b", R_lambda)) / num_tokens
        tr[count_infinite_tr] = 0  # Set trace to zero where infinite values are found
        return tr.sum()

    def compute_tr_R_lambda_single_image(self, img_tokens, should_log_norms=False):
        device = img_tokens.device

        N = img_tokens.shape[0]
        R = (img_tokens.T @ img_tokens).to(device)
        X_T_X = torch.norm(R)
        R = torch.mul(R, self.P)
        l2_P = torch.norm(self.P)
        wk = self.k.weight
        wq = self.q.weight
        lambda1 = wq.T @ wk
        Omega = (wq.T @ wk @ img_tokens.T).to(device)
        Lambda = Omega @ Omega.T
        R_lambda = R @ Lambda.double()
        l2_wq_T_wk_X_T = torch.norm(Omega)
        l2_wq_T_wk = torch.norm(lambda1)
        l2_r_lambda = torch.norm(R_lambda)
        l2_lambda = torch.norm(Lambda)
        l2_R = torch.norm(R)
        count_infinite_tr = 0
        if torch.isinf(R_lambda).any() or torch.isnan(R_lambda).any():
            count_infinite_tr = 1
            tr = torch.zeros([]).to(device)
            l2_r_lambda = -1
            return tr
        tr = torch.trace(R_lambda)
        tr = torch.abs(tr)
        tr = tr / N
        if should_log_norms == True:
            return (
                tr,
                X_T_X,
                l2_P,
                l2_wq_T_wk_X_T,
                l2_wq_T_wk,
                l2_r_lambda,
                l2_lambda,
                l2_R,
                count_infinite_tr,
            )
        else:
            return tr

    def compute_P(self, D, drop_ratio):
        P = np.full((D, D), drop_ratio**2)
        return torch.from_numpy(P).float()

class AxialAttention(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        groups=8,
        kernel_size=56,
        stride=1,
        bias=False,
        width=False,
    ):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.qkv_transform = nn.Conv1d(
            in_planes,
            out_planes * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)
        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        # Position embedding
        self.relative = nn.Parameter(
            torch.randn(self.group_planes * 2, kernel_size * 2 - 1),
            requires_grad=True,
        )
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # N, W, C, H
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(
            qkv.reshape(N * W, self.groups, self.group_planes * 2, H),
            [self.group_planes // 2, self.group_planes // 2, self.group_planes],
            dim=2,
        )

        # Calculate position embedding
        all_embeddings = torch.index_select(
            self.relative, 1, self.flatten_index
        ).view(self.group_planes * 2, self.kernel_size, self.kernel_size)
        q_embedding, k_embedding, v_embedding = torch.split(
            all_embeddings,
            [self.group_planes // 2, self.group_planes // 2, self.group_planes],
            dim=0,
        )
        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
        qk = torch.einsum('bgci, bgcj->bgij', q, k)
        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = (
            self.bn_similarity(stacked_similarity)
            .view(N * W, 3, self.groups, H, H)
            .sum(dim=1)
        )

        # (N, groups, H, H, W)
        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)
        stacked_output = torch.cat([sv, sve], dim=-1).view(
            N * W, self.out_planes * 2, H
        )
        output = (
            self.bn_output(stacked_output)
            .view(N, W, self.out_planes, 2, H)
            .sum(dim=-2)
        )

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(
            0, math.sqrt(1.0 / self.in_planes)
        )
        nn.init.normal_(self.relative, 0.0, math.sqrt(1.0 / self.group_planes))
