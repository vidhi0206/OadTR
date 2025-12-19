from torch import nn
from .Attention import SelfAttention,SelfAttentionDr
from .Linear_dr import Linear_dr


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn # PreNormDrop

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn #"self attention"

    def forward(self, x):
        return self.fn(self.norm(x))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate,dr_mlp_mode=0):
        super().__init__()
        self.dr_mlp_mode=dr_mlp_mode
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate) if self.dr_mlp_mode==0 else nn.Identity(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate) if self.dr_mlp_mode==0 else nn.Identity(),
        )

    def forward(self, x):
        return self.net(x)

class FeedForwardDr(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate, dr_mlp_mode=0):
        super().__init__()
        self.dr_mlp_mode=dr_mlp_mode
        self.net = nn.Sequential(
            Linear_dr(dim, hidden_dim,dropout_rate) if self.dr_mlp_mode==2 else nn.Linear(dim, hidden_dim),
            nn.GELU(),
            Linear_dr(hidden_dim, dim,dropout_rate) if self.dr_mlp_mode==2 else nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        mlp_dim,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        drop_mha="drop_none",
        dr_mha="none",
        dr_mlp_mode=0,
    ):
        self.drop_mha = drop_mha
        self.dr_mha = dr_mha
        self.dr_mlp_mode = dr_mlp_mode
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend(
                [
                    Residual(
                        PreNormDrop(
                            dim,
                            dropout_rate,
                            SelfAttention(
                                dim, heads=heads, dropout_rate=attn_dropout_rate, drop_mha=self.drop_mha, dr_mlp_mode=self.dr_mlp_mode
                            ) if self.dr_mha == "none" else SelfAttentionDr(dim, heads=heads, dropout_rate=attn_dropout_rate, dr_mha=self.dr_mha, dr_mlp_mode=self.dr_mlp_mode
                            ),
                        )
                    ),
                    Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout_ratedr_mlp_mode=self.dr_mlp_mode) if self.dr_mlp_mode==0 else FeedForwardDr(dim, mlp_dim, dropout_rate,dr_mlp_mode=self.dr_mlp_mode))),
                ]
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _register_ptflops():
    try:
        from ptflops import flops_counter as fc

        def self_attention_counter_hook(module, input, output):
            flops = 0

            q = input[0]
            k = input[0]
            v = input[0]
            batch_size = q.shape[1]

            num_heads = module.num_heads
            embed_dim = module.qkv.in_features
            kdim = embed_dim
            vdim = embed_dim

            # initial projections
            flops = (
                q.shape[0] * q.shape[2] * embed_dim
                + k.shape[0] * k.shape[2] * kdim
                + v.shape[0] * v.shape[2] * vdim
            )
            if module.qkv.bias is not None:
                flops += (q.shape[0] + k.shape[0] + v.shape[0]) * embed_dim

            # attention heads: scale, matmul, softmax, matmul
            head_dim = embed_dim // num_heads
            head_flops = (
                q.shape[0] * head_dim
                + head_dim * q.shape[0] * k.shape[0]
                + q.shape[0] * k.shape[0]
                + q.shape[0] * k.shape[0] * head_dim
            )

            flops += num_heads * head_flops

            # final projection, bias is always enabled
            flops += q.shape[0] * embed_dim * (embed_dim + 1)

            flops *= batch_size
            module.__flops__ += int(flops)

        fc.MODULES_MAPPING[SelfAttention] = self_attention_counter_hook

    except ModuleNotFoundError:  # pragma: no cover
        pass
    except Exception as e:  # pragma: no cover
        print(f"Failed to add flops_counter_hook: {e}")


_register_ptflops()
