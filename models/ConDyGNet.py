import math
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import PositionalEmbedding
from layers.StandardNorm import Normalize




@torch.no_grad()
def _moving_average_detrend(values: np.ndarray, window: int) -> np.ndarray:
    T, C = values.shape
    if window <= 1 or window >= T:
        mean = values.mean(axis=0, keepdims=True)
        return values - mean
    df = pd.DataFrame(values)
    trend = df.rolling(window=window, center=True, min_periods=1).mean().to_numpy(dtype=np.float32)
    return values - trend


@torch.no_grad()
def overall_Matrix_from_train_csv(root_path: str, data_path: str, data_name: str) -> torch.Tensor:
    """
    return: B_stat [C, C] on CPU
    NOTE: used only for E0 initialization (PCA). NOT used in graph fusion.
    """
    df_raw = pd.read_csv(os.path.join(root_path, data_path))

    if df_raw.shape[1] > 1:
        df_data = df_raw[df_raw.columns[1:]]
    else:
        df_data = df_raw

    if data_name in ["ETTh1", "ETTh2"]:
        train_df = df_data.iloc[0:12 * 30 * 24]
    elif data_name in ["ETTm1", "ETTm2"]:
        train_df = df_data.iloc[0:12 * 30 * 24 * 4]
    else:
        num_train = int(len(df_raw) * 0.7)
        train_df = df_data.iloc[0:num_train]

    values = train_df.to_numpy(dtype=np.float32)  # [T, C]
    T, C = values.shape

    if data_name in ["ETTh1", "ETTh2", "Weather"]:
        ma_window = 24
    elif data_name in ["ETTm1", "ETTm2"]:
        ma_window = 24 * 4
    elif data_name in ["Traffic"]:
        ma_window = 24
    else:
        ma_window = int(max(4, min(512, T // 100)))

    residual = _moving_average_detrend(values, ma_window)  # [T,C]
    x = torch.tensor(residual, dtype=torch.float32).transpose(0, 1)  # [C,T]

    x = x - x.mean(dim=1, keepdim=True)
    x = x / (x.norm(dim=1, keepdim=True) + 1e-8)
    cos = x @ x.transpose(0, 1)

    B_stat = torch.clamp(cos, min=0.0, max=1.0)
    return B_stat


@torch.no_grad()
def init_E_from_base_B(base_B: torch.Tensor, R: int) -> torch.Tensor:
    """
    base_B: [C,C] on CPU (statistics)
    return: E0_init [C,R] on CPU
    """
    if base_B is None or (not torch.is_tensor(base_B)) or base_B.dim() != 2:
        raise ValueError("base_B must be a 2D tensor [C,C].")
    C = int(base_B.shape[0])
    if base_B.shape[1] != C:
        raise ValueError("base_B must be square [C,C].")
    R = int(max(2, min(int(R), C)))

    B0 = base_B.detach().float().clone()
    eye = torch.eye(C, dtype=B0.dtype, device=B0.device)
    B0 = B0 * (1.0 - eye)
    B0 = 0.5 * (B0 + B0.t())

    q = int(min(R, C))
    U, _, _ = torch.pca_lowrank(B0, q=q, center=False)
    E0_init = U[:, :R].contiguous()

    if not torch.isfinite(E0_init).all():
        raise ValueError("E0_init contains non-finite values.")
    return E0_init


class PatchEmbed(nn.Module):
    def __init__(self, dim: int, patch_len: int, stride: Optional[int] = None, pos: bool = True):
        super().__init__()
        self.patch_len = int(patch_len)
        self.stride = int(self.patch_len if stride is None else stride)
        self.patch_proj = nn.Linear(self.patch_len, dim)
        self.pos = bool(pos)
        if self.pos:
            self.pe = PositionalEmbedding(dim, 10000)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [B, L, patch_len]
        x = self.patch_proj(x)  # [B, L, D]
        if self.pos:
            x = x + self.pe(x)
        return x


class PatchGraphLearner(nn.Module):
    def __init__(
        self,
        dim_head: int,
        n_vars: int,
        n_heads: int,
        R: int,
        base_B: torch.Tensor,               # kept for interface compatibility (unused)
        base_E: Optional[torch.Tensor],     # used as E0 init
        top_k: float,
        alpha_max: float,                   # kept for interface compatibility (unused)
        alpha_min: float,                   # kept for interface compatibility (unused)
        alpha_decay_steps: int,             # kept for interface compatibility (unused)
    ):
        super().__init__()
        self.n_vars = int(n_vars)
        self.n_heads = int(n_heads)
        self.dim_head = int(dim_head)

        R = int(R)
        if R < 2:
            R = 2
        self.R = R

        try:
            self.top_k = float(top_k)
        except Exception:
            self.top_k = float(int(top_k))

        self.proj_1 = nn.Linear(self.dim_head, self.dim_head)
        self.proj_2 = nn.Linear(self.dim_head, self.dim_head)

        if base_E is not None and torch.is_tensor(base_E) and base_E.dim() == 2:
            if base_E.shape[0] == self.n_vars and base_E.shape[1] == self.R:
                E0 = base_E.detach().float().clone()
            else:
                E0 = torch.randn(self.n_vars, self.R) * 0.02
        else:
            E0 = torch.randn(self.n_vars, self.R) * 0.02

        self.register_buffer("E0", E0)  # [C,R] fixed
        self.Q = nn.Parameter(torch.eye(self.R, dtype=torch.float32))  # [R,R] learnable

        hidden = max(16, self.dim_head)
        self.psi = nn.Sequential(
            nn.Linear(self.dim_head, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.R),
        )

        mu0 = torch.eye(self.R, dtype=torch.float32).unsqueeze(0).repeat(self.n_heads, 1, 1)  # [H,R,R]
        self.mu = nn.Parameter(mu0)
        self.log_tau = nn.Parameter(torch.tensor(math.log(2.0), dtype=torch.float32))  # scalar tau

    def forward(self, x_hld: torch.Tensor, is_training: bool = False) -> torch.Tensor:
        """
        x_hld: [B,H,L,dh], L = C*P
        return: adj_patch [B,H,P,C,C] (row-softmax on C)
        """
        eps = 1e-12
        B, H, L, dh = x_hld.shape
        C = self.n_vars
        assert H == self.n_heads
        assert dh == self.dim_head
        assert L % C == 0, f"L={L} must be divisible by C={C}"
        P = L // C

        x_cp = x_hld.contiguous().view(B, H, C, P, dh)
        x_pc = x_cp.permute(0, 1, 3, 2, 4).contiguous()  # [B,H,P,C,dh]

        q = self.proj_1(x_pc)
        k = self.proj_2(x_pc)
        S = torch.einsum("bhpcd,bhpmd->bhpcm", q, k) / math.sqrt(float(self.dim_head))  # [B,H,P,C,C]
        W = torch.sigmoid(S)  # [B,H,P,C,C]

        u = self.psi(x_pc)  # [B,H,P,C,R]
        tau = torch.exp(self.log_tau).clamp(0.2, 10.0)

        diff = u.unsqueeze(-2) - self.mu.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # [B,H,P,C,R,R]
        dist2 = (diff * diff).sum(dim=-1)  # [B,H,P,C,R]
        logits = -dist2 / tau
        Cmix = torch.softmax(logits, dim=-1)  # [B,H,P,C,R]

        E = (self.E0 @ self.Q).to(dtype=u.dtype, device=u.device)  # [C,R]
        G = torch.einsum("bhpcr,vr->bhpcv", Cmix, E)  # [B,H,P,C,C]

        Gp = F.relu(G)
        row_max = Gp.max(dim=-1, keepdim=True).values
        Gtilde = Gp / (row_max + eps)

        diagC = torch.eye(C, device=x_hld.device, dtype=torch.bool).view(1, 1, 1, C, C)
        Gtilde = Gtilde.masked_fill(diagC, 0.0)

        tk = float(self.top_k)
        if 0.0 < tk <= 1.0:
            k_keep = int(math.ceil(tk * float(max(C - 1, 1))))
        else:
            k_keep = int(round(tk))
        k_keep = max(1, min(k_keep, C - 1))

        G_rank = Gtilde.masked_fill(diagC, -1.0)
        topk_idx = torch.topk(G_rank, k=k_keep, dim=-1, largest=True, sorted=False).indices  # [B,H,P,C,k]

        M = torch.zeros_like(Gtilde)
        topk_val = torch.gather(Gtilde, dim=-1, index=topk_idx)  # [B,H,P,C,k]
        M.scatter_(dim=-1, index=topk_idx, src=topk_val)

        M = M.masked_fill(diagC, 1.0)

        A_patch = torch.clamp(M * W, 0.0, 1.0)

        A_logits = A_patch.masked_fill(M <= 0.0, -1e9)
        adj_patch = torch.softmax(A_logits, dim=-1)  # [B,H,P,C,C]
        return adj_patch


class PatchGCN(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_vars: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.n_heads = int(n_heads)
        self.n_vars = int(n_vars)
        assert dim % self.n_heads == 0
        self.d_head = dim // self.n_heads

    def forward(self, adj: torch.Tensor, x: torch.Tensor, mp_layers: int, mp_beta: float) -> torch.Tensor:
        """
        adj: [B,H,P,C,C]
        x  : [B,L,D], L=C*P
        """
        B, L, D = x.shape
        C = self.n_vars
        if C <= 0 or L % C != 0:
            return x
        P = L // C
        if P < 1:
            return x

        adj = F.normalize(adj, p=1, dim=-1)

        x_cp = x.contiguous().view(B, C, P, D)
        x_head = self.proj(x_cp).view(B, C, P, self.n_heads, self.d_head)

        device = x_head.device
        beta = float(mp_beta)
        Lh = max(1, int(mp_layers))

        h0 = x_head.permute(0, 3, 2, 1, 4).contiguous()  # [B,H,P,C,d]
        hs = [h0]
        h = h0
        for _ in range(Lh):
            msg = torch.einsum("bhpcm,bhpmd->bhpcd", adj, h)
            h = beta * h0 + (1.0 - beta) * msg
            hs.append(h)

        ws = torch.tensor([1.0 / (i + 1) for i in range(len(hs))], device=device, dtype=h0.dtype)
        ws = ws / ws.sum()

        out = 0.0
        for w, ht in zip(ws, hs):
            out = out + w * ht

        out = out.permute(0, 3, 2, 1, 4).contiguous()  # [B,C,P,H,d]
        return out.reshape(B, C * P, D).contiguous()


class GraphFilter(nn.Module):
    def __init__(
        self,
        dim: int,
        n_vars: int,
        n_heads: int,
        d_ff: int,
        R: int,
        base_B: torch.Tensor,
        base_E: Optional[torch.Tensor],
        top_k: float,
        dropout: float,
        alpha_max: float,
        alpha_min: float,
        alpha_decay_steps: int,
        mp_layers: int,
        mp_beta: float,
    ):
        super().__init__()
        self.dim = int(dim)
        self.n_heads = int(n_heads)
        self.n_vars = int(n_vars)

        assert self.dim % self.n_heads == 0
        dim_head = self.dim // self.n_heads

        self.graph_learner = PatchGraphLearner(
            dim_head=dim_head,
            n_vars=self.n_vars,
            n_heads=self.n_heads,
            R=R,
            base_B=base_B,
            base_E=base_E,
            top_k=top_k,
            alpha_max=alpha_max,
            alpha_min=alpha_min,
            alpha_decay_steps=alpha_decay_steps,
        )
        self.dropout = nn.Dropout(float(dropout))  # dropout after softmax, before conv
        self.graph_conv = PatchGCN(self.dim, self.n_heads, self.n_vars)

        self.mp_layers = int(mp_layers)
        self.mp_beta = float(mp_beta)

    def forward(self, x: torch.Tensor, is_training: bool = False) -> torch.Tensor:
        B, L, D = x.shape
        d_head = D // self.n_heads

        x_hld = x.view(B, L, self.n_heads, d_head).permute(0, 2, 1, 3).contiguous()  # [B,H,L,dh]
        adj_patch = self.graph_learner(x_hld, is_training=is_training)  # [B,H,P,C,C]

        adj_patch = self.dropout(adj_patch)
        out = self.graph_conv(adj_patch, x, mp_layers=self.mp_layers, mp_beta=self.mp_beta)
        return out


class GraphBlock(nn.Module):
    """
    Residual form:
        out = gnn(norm1(x))
        x = x + out
        x = x + ffn(norm2(x))
    """
    def __init__(self, dim: int, d_ff: int, gnn: GraphFilter, dropout: float):
        super().__init__()
        self.gnn = gnn
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, d_ff),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(d_ff, dim),
        )

    def forward(self, x: torch.Tensor, is_training: bool = False) -> torch.Tensor:
        out = self.gnn(self.norm1(x), is_training=is_training)
        x = x + out
        x = x + self.ffn(self.norm2(x))
        return x


class ConDyGNet_Backbone(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        n_vars: int,
        d_ff: int,
        n_heads: int,
        n_blocks: int,
        dropout: float,
        R: int,
        base_B: torch.Tensor,
        base_E: Optional[torch.Tensor],
        top_k: float,
        alpha_max: float,
        alpha_min: float,
        alpha_decay_steps: int,
        mp_layers: int,
        mp_beta: float,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(int(n_blocks)):
            gf = GraphFilter(
                dim=hidden_dim,
                n_vars=n_vars,
                n_heads=n_heads,
                d_ff=d_ff,
                R=R,
                base_B=base_B,
                base_E=base_E,
                top_k=top_k,
                dropout=dropout,
                alpha_max=alpha_max,
                alpha_min=alpha_min,
                alpha_decay_steps=alpha_decay_steps,
                mp_layers=mp_layers,
                mp_beta=mp_beta,
            )
            self.blocks.append(GraphBlock(hidden_dim, d_ff, gf, dropout))

    def forward(self, x: torch.Tensor, is_training: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        for blk in self.blocks:
            x = blk(x, is_training=is_training)
        aux_loss = x.new_zeros(())
        return x, aux_loss


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)

        self.n_vars = int(getattr(configs, "c_out", getattr(configs, "enc_in", 1)))
        self.dim = int(getattr(configs, "d_model", 256))
        self.d_ff = int(getattr(configs, "d_ff", 256))

        self.patch_len = int(getattr(configs, "patch_len", 16))
        self.stride = self.patch_len

        assert self.seq_len % self.patch_len == 0, f"seq_len={self.seq_len} must be divisible by patch_len={self.patch_len}"
        self.num_patches = self.seq_len // self.patch_len

        R = getattr(configs, "R", None)
        if R is None:
            R = getattr(configs, "dict_size", None)
        self.R = int(16 if R is None else R)

        self.mp_layers = int(getattr(configs, "mp_layers", 2))
        self.mp_beta = float(getattr(configs, "mp_beta", 0.2))

        self.alpha_max = float(getattr(configs, "alpha_max", 0.9))
        self.alpha_min = float(getattr(configs, "alpha_min", 0.1))
        self.alpha_decay_steps = int(getattr(configs, "alpha_decay_steps", 20000))

        top_k = getattr(configs, "top_k", None)
        if top_k is None:
            top_k = getattr(configs, "topk", None)
        if top_k is None:
            top_k = getattr(configs, "k_graph", None)
        if top_k is None:
            top_k = 0.3
        try:
            self.top_k = float(top_k)
        except Exception:
            self.top_k = float(int(top_k))

        n_heads = int(getattr(configs, "n_heads", 8))
        if n_heads <= 0:
            n_heads = 1
        if (self.dim % n_heads) != 0:
            n_heads = 1
        self.n_heads = n_heads

        drop = float(getattr(configs, "dropout", 0.0))
        pos = bool(getattr(configs, "pos", True))

        root_path = getattr(configs, "root_path", "./dataset")
        data_path = getattr(configs, "data_path", "ETTh1.csv")
        data_name = getattr(configs, "data", "ETTh1")

        base_B = overall_Matrix_from_train_csv(root_path, data_path, data_name)  # CPU
        if base_B.dim() != 2 or base_B.shape[0] != self.n_vars or base_B.shape[1] != self.n_vars:
            base_B = torch.eye(self.n_vars, dtype=torch.float32)

        try:
            base_E = init_E_from_base_B(base_B, self.R)  # [C,R] on CPU
        except Exception:
            base_E = torch.randn(self.n_vars, self.R) * 0.02

        self.patch_embed = PatchEmbed(self.dim, self.patch_len, self.stride, pos=pos)

        self.backbone = ConDyGNet_Backbone(
            hidden_dim=self.dim,
            n_vars=self.n_vars,
            d_ff=self.d_ff,
            n_heads=self.n_heads,
            n_blocks=int(getattr(configs, "e_layers", 2)),
            dropout=drop,
            R=self.R,
            base_B=base_B,
            base_E=base_E,
            top_k=self.top_k,
            alpha_max=self.alpha_max,
            alpha_min=self.alpha_min,
            alpha_decay_steps=self.alpha_decay_steps,
            mp_layers=self.mp_layers,
            mp_beta=self.mp_beta,
        )

        attn_heads = self.n_heads if (self.dim % self.n_heads == 0) else 1
        self.patch_attn_norm = nn.LayerNorm(self.dim)
        self.patch_attn = nn.MultiheadAttention(self.dim, attn_heads, dropout=drop, batch_first=True)
        self.patch_attn_drop = nn.Dropout(drop)

        self.head = nn.Linear(self.dim * self.num_patches, self.pred_len)

        self.use_RevIN = False
        self.norm = Normalize(getattr(configs, "enc_in", self.n_vars), affine=self.use_RevIN)

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: Optional[torch.Tensor] = None,
        x_dec: Optional[torch.Tensor] = None,
        x_mark_dec: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        is_training: bool = False,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x_enc  # [B,T,C]
        B, T, C = x.shape

        x = self.norm(x, "norm")

        x = x.permute(0, 2, 1).reshape(B, C * T)

        x = self.patch_embed(x)

        x, _ = self.backbone(x, is_training=is_training)

        x_cp = x.view(B, self.n_vars, self.num_patches, self.dim)
        x_bp = x_cp.reshape(B * self.n_vars, self.num_patches, self.dim)
        x_ln = self.patch_attn_norm(x_bp)
        attn_out, _ = self.patch_attn(x_ln, x_ln, x_ln, need_weights=False)
        x_bp = x_bp + self.patch_attn_drop(attn_out)
        x_cp = x_bp.view(B, self.n_vars, self.num_patches, self.dim)

        x = self.head(x_cp.flatten(start_dim=-2))  # [B,C,pred_len]
        x = x.permute(0, 2, 1)  # [B,pred_len,C]

        x = self.norm(x, "denorm")

        aux_loss = x.new_zeros(())
        return x, aux_loss
