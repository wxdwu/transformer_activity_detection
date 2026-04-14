from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenBatchNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        # Paper Eq. (20)-(23) uses mini-batch statistics.
        # Use batch stats in both train/eval (no running stats) for closer behavior.
        self.bn = nn.BatchNorm1d(dim, track_running_stats=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        b, t, d = x.shape
        y = self.bn(x.reshape(b * t, d))
        return y.reshape(b, t, d)


class HeterogeneousMHA(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.wq_b = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.wk_b = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.wv_b = nn.Linear(dim, num_heads * head_dim, bias=False)

        self.wq_y = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.wk_y = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.wv_y = nn.Linear(dim, num_heads * head_dim, bias=False)

        self.wo_b = nn.Parameter(torch.randn(num_heads, dim, head_dim) * 0.02)
        self.wo_y = nn.Parameter(torch.randn(num_heads, dim, head_dim) * 0.02)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.num_heads, self.head_dim)

    def forward(self, x_b: torch.Tensor, x_y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, n, _ = x_b.shape
        x_all = torch.cat([x_b, x_y], dim=1)  # [B, N+1, D]

        q = torch.cat([self._split_heads(self.wq_b(x_b)), self._split_heads(self.wq_y(x_y))], dim=1)
        k = torch.cat([self._split_heads(self.wk_b(x_b)), self._split_heads(self.wk_y(x_y))], dim=1)
        v = torch.cat([self._split_heads(self.wv_b(x_b)), self._split_heads(self.wv_y(x_y))], dim=1)

        qh = q.permute(0, 2, 1, 3)  # [B, H, N+1, Dh]
        kh = k.permute(0, 2, 1, 3)
        vh = v.permute(0, 2, 1, 3)

        score = (qh @ kh.transpose(-1, -2)) / math.sqrt(self.head_dim)
        weight = torch.softmax(score, dim=-1)
        ctx = weight @ vh  # [B, H, N+1, Dh]
        ctx = ctx.permute(0, 2, 1, 3)  # [B, N+1, H, Dh]

        ctx_b = ctx[:, :n]
        ctx_y = ctx[:, n:]

        out_b = torch.einsum("bnth,tdh->bnd", ctx_b, self.wo_b)
        out_y = torch.einsum("bnth,tdh->bnd", ctx_y, self.wo_y)
        return out_b, out_y


class HeterogeneousFFN(nn.Module):
    def __init__(self, dim: int, ff_dim: int) -> None:
        super().__init__()
        self.ff_b = nn.Sequential(nn.Linear(dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, dim))
        self.ff_y = nn.Sequential(nn.Linear(dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, dim))

    def forward(self, x_b: torch.Tensor, x_y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.ff_b(x_b), self.ff_y(x_y)


class HeterogeneousEncoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int, ff_dim: int) -> None:
        super().__init__()
        self.mha = HeterogeneousMHA(dim=dim, num_heads=num_heads, head_dim=head_dim)
        self.ffn = HeterogeneousFFN(dim=dim, ff_dim=ff_dim)
        self.bn1_b = TokenBatchNorm(dim)
        self.bn1_y = TokenBatchNorm(dim)
        self.bn2_b = TokenBatchNorm(dim)
        self.bn2_y = TokenBatchNorm(dim)

    def forward(self, x_b: torch.Tensor, x_y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mha_b, mha_y = self.mha(x_b, x_y)
        hat_b = self.bn1_b(x_b + mha_b)
        hat_y = self.bn1_y(x_y + mha_y)

        ff_b, ff_y = self.ffn(hat_b, hat_y)
        out_b = self.bn2_b(hat_b + ff_b)
        out_y = self.bn2_y(hat_y + ff_y)
        return out_b, out_y


class ContextDecoder(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int, score_scale: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.score_scale = score_scale

        self.wq_c = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.wk_cb = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.wv_cb = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.wk_cy = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.wv_cy = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.wo_c = nn.Parameter(torch.randn(num_heads, dim, head_dim) * 0.02)
        self.w_out = nn.Parameter(torch.randn(dim, dim) * 0.02)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.num_heads, self.head_dim)

    def forward(self, x_b: torch.Tensor, x_y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Context attention: query from Y token, keys/values from {B tokens + Y token}
        bsz, n, d = x_b.shape

        q = self._split_heads(self.wq_c(x_y))  # [B,1,H,Dh]
        k = torch.cat([self._split_heads(self.wk_cb(x_b)), self._split_heads(self.wk_cy(x_y))], dim=1)
        v = torch.cat([self._split_heads(self.wv_cb(x_b)), self._split_heads(self.wv_cy(x_y))], dim=1)

        qh = q.permute(0, 2, 1, 3)  # [B,H,1,Dh]
        kh = k.permute(0, 2, 1, 3)  # [B,H,N+1,Dh]
        vh = v.permute(0, 2, 1, 3)

        score = (qh @ kh.transpose(-1, -2)) / math.sqrt(self.head_dim)
        weight = torch.softmax(score, dim=-1)
        ctx = weight @ vh  # [B,H,1,Dh]
        ctx = ctx.permute(0, 2, 1, 3)  # [B,1,H,Dh]
        xc = torch.einsum("bnth,tdh->bnd", ctx, self.wo_c).squeeze(1)  # [B,D]

        # Output block: Eq. (25)-(26)
        xc_w = xc @ self.w_out  # [B,D]
        match = (xc_w.unsqueeze(1) * x_b).sum(dim=-1) / math.sqrt(d)
        logits = self.score_scale * torch.tanh(match)
        probs = torch.sigmoid(logits)
        return logits, probs


class HeterogeneousTransformer(nn.Module):
    def __init__(
        self,
        num_devices: int,
        pilot_len: int,
        dim: int = 128,
        num_layers: int = 5,
        num_heads: int = 8,
        head_dim: int = 32,
        ff_dim: int = 512,
        score_scale: float = 10.0,
    ) -> None:
        super().__init__()
        self.num_devices = num_devices
        self.pilot_len = pilot_len
        self.dim = dim

        self.embed_b = nn.Linear(2 * pilot_len, dim)
        self.embed_y = nn.Linear(2 * pilot_len * pilot_len, dim)

        self.layers = nn.ModuleList(
            [
                HeterogeneousEncoderLayer(
                    dim=dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    ff_dim=ff_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.decoder = ContextDecoder(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            score_scale=score_scale,
        )

    def forward(self, x_b: torch.Tensor, x_y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x_b: [B,N,2Lp], x_y: [B,2Lp^2]
        h_b = self.embed_b(x_b)
        h_y = self.embed_y(x_y).unsqueeze(1)

        for layer in self.layers:
            h_b, h_y = layer(h_b, h_y)

        return self.decoder(h_b, h_y)
