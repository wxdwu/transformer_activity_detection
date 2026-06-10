from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# 本文件实现论文：
# "Heterogeneous Transformer: A Scale Adaptable Neural Network
# Architecture for Device Activity Detection" 中的异构 Transformer。
#
# 下文常用符号：
# B  = batch size，一次训练/推理中的样本数
# N  = device/user 数量
# Lp = pilot sequence length，导频长度
# D  = 模型嵌入维度，对应论文中的 d
# H  = 注意力头数，对应论文中的 T
# Dh = 每个注意力头的维度，对应论文中的 d'
#
# 论文里有两类物理意义不同的 token：
# 1) 设备导频 token：x_1, ..., x_N，由缩放后的导频矩阵 B 的各列得到
# 2) 接收信号 token：x_{N+1}，由接收信号协方差 C = YY^H / M 得到
# 之所以叫 "heterogeneous"，是因为这两类 token 使用不同的投影、FFN 和归一化参数。


class TokenBatchNorm(nn.Module):
    """论文 Eq. (20)-(23) 使用的 BatchNorm 模块。

    输入 x 的形状是 [B, T, D]。
    T 表示当前要归一化的 token 数量：设备导频 token 时是 N，
    接收信号 token 时是 1。
    reshape 成 [B*T, D] 后，BatchNorm 会在所有样本、所有同类型 token
    上统计每个特征维度的均值和方差。
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        # 论文 Eq. (20)-(23) 使用 mini-batch 统计量。
        # 这里 train/eval 都使用当前 batch 统计量，不维护 running stats，更接近论文写法。
        self.bn = nn.BatchNorm1d(dim, track_running_stats=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        b, t, d = x.shape
        y = self.bn(x.reshape(b * t, d))
        return y.reshape(b, t, d)


class TokenLayerNorm(nn.Module):
    """可选的 LayerNorm 变体。

    这不是论文默认设置；论文 baseline 使用 BatchNorm。
    单独封装一层是为了在不改动其余异构 Transformer 结构的情况下切换归一化方式。
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x)


class HeterogeneousMHA(nn.Module):
    """异构多头注意力模块，对应论文 Eq. (10)-(17)。

    设备导频 token 和接收信号 token 参与同一个 attention 计算，
    但使用不同的 Q/K/V 投影参数：
    - x_1, ..., x_N 使用 Wq/Wk/Wv_B
    - x_{N+1} 使用 Wq/Wk/Wv_Y
    多头输出投影也分成 Wo_B 和 Wo_Y 两套参数。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # D  = dim，token 的嵌入维度。
        # H  = num_heads，多头注意力的 head 数。
        # Dh = head_dim，每个 head 内部的维度。
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Step 0-1，对应论文 Eq. (10)-(12)：B 类 token 使用单独的 Q/K/V 投影。
        # 输入 x_b: [B, N, D]
        # 输出 Q_B/K_B/V_B: [B, N, H*Dh]
        self.wq_b = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.wk_b = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.wv_b = nn.Linear(dim, num_heads * head_dim, bias=False)

        # Step 0-2，对应论文 Eq. (10)-(12)：Y 类 token 使用另一套 Q/K/V 投影。
        # 输入 x_y: [B, 1, D]
        # 输出 Q_Y/K_Y/V_Y: [B, 1, H*Dh]
        self.wq_y = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.wk_y = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.wv_y = nn.Linear(dim, num_heads * head_dim, bias=False)

        # 论文 Eq. (16)-(17)：每个 head 的输出投影也按 token 类型区分。
        # wo_b/wo_y: [H, D, Dh]，用于把每个 head 的 Dh 维输出投影回 D 维。
        self.wo_b = nn.Parameter(torch.randn(num_heads, dim, head_dim) * 0.02)
        self.wo_y = nn.Parameter(torch.randn(num_heads, dim, head_dim) * 0.02)
        self.attn_drop = nn.Dropout(attn_dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # 把最后一维 H*Dh 拆成两个维度 H 和 Dh。
        # 输入 : [B, T, H*Dh]
        # 输出 : [B, T, H, Dh]
        b, t, _ = x.shape
        return x.view(b, t, self.num_heads, self.head_dim)

    def forward(self, x_b: torch.Tensor, x_y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 输入：
        # x_b: [B, N, D]，N 个设备/用户导频 token。
        # x_y: [B, 1, D]，1 个接收信号 token。
        bsz, n, _ = x_b.shape
        x_all = torch.cat([x_b, x_y], dim=1)  # [B, N+1, D]，整体 token 序列，仅用于表达拼接形状。

        # Step 1，对应论文 Eq. (10)-(12)：为全部 N+1 个 token 构造 Q/K/V。
        # B token 路径：
        #   wq_b/wk_b/wv_b: [B, N, D] -> [B, N, H*Dh]
        #   _split_heads:   [B, N, H*Dh] -> [B, N, H, Dh]
        # Y token 路径：
        #   wq_y/wk_y/wv_y: [B, 1, D] -> [B, 1, H*Dh]
        #   _split_heads:   [B, 1, H*Dh] -> [B, 1, H, Dh]
        # 沿 token 维 dim=1 拼接：
        #   [B, N, H, Dh] + [B, 1, H, Dh] -> [B, N+1, H, Dh]
        q = torch.cat([self._split_heads(self.wq_b(x_b)), self._split_heads(self.wq_y(x_y))], dim=1)
        k = torch.cat([self._split_heads(self.wk_b(x_b)), self._split_heads(self.wk_y(x_y))], dim=1)
        v = torch.cat([self._split_heads(self.wv_b(x_b)), self._split_heads(self.wv_y(x_y))], dim=1)

        # Step 2：调整维度以调用 PyTorch 的 scaled_dot_product_attention。
        # q/k/v 当前是 [B, N+1, H, Dh]。
        # PyTorch attention 需要 [B, H, N+1, Dh]。
        qh = q.permute(0, 2, 1, 3)  # [B, H, N+1, Dh]
        kh = k.permute(0, 2, 1, 3)
        vh = v.permute(0, 2, 1, 3)

        # Step 3，对应论文 Eq. (13)-(15)：scaled dot-product attention。
        # 内部等价于：
        #   Eq. (13) alpha = Q K^H / sqrt(Dh)，计算 token 间 compatibility。
        #   Eq. (14) beta  = softmax(alpha)，得到 attention 权重。
        #   Eq. (15) ctx   = beta V，对 value 做加权求和。
        # 输入 qh/kh/vh: [B, H, N+1, Dh]
        # 输出 ctx:      [B, H, N+1, Dh]
        drop_p = self.attn_drop.p if self.training else 0.0
        ctx = F.scaled_dot_product_attention(
            qh,
            kh,
            vh,
            attn_mask=None,
            dropout_p=drop_p,
            is_causal=False,
        )  # [B, H, N+1, Dh]
        # Step 4：把 head 维放回 token 维后面，方便按 B/Y token 切分。
        # [B, H, N+1, Dh] -> [B, N+1, H, Dh]
        ctx = ctx.permute(0, 2, 1, 3)  # [B, N+1, H, Dh]

        # Step 5：按拼接顺序切回两类 token。
        # ctx_b: [B, N, H, Dh]，前 N 个设备 token。
        # ctx_y: [B, 1, H, Dh]，最后 1 个接收信号 token。
        ctx_b = ctx[:, :n]
        ctx_y = ctx[:, n:]

        # Step 6，对应论文 Eq. (16)-(17)：把多个 attention head 的结果合并回 D 维。
        # ctx_b: [B, N, H, Dh], wo_b: [H, D, Dh] -> out_b: [B, N, D]
        # ctx_y: [B, 1, H, Dh], wo_y: [H, D, Dh] -> out_y: [B, 1, D]
        # einsum "bnth,tdh->bnd" 表示对 head 维 t 和 head_dim 维 h 求和，
        # 保留 batch 维 b、token 维 n、输出特征维 d。
        out_b = torch.einsum("bnth,tdh->bnd", ctx_b, self.wo_b)
        out_y = torch.einsum("bnth,tdh->bnd", ctx_y, self.wo_y)
        return out_b, out_y


class GroupedHeterogeneousMHA(nn.Module):
    """MHA variant for group-correlated pilots.

    Device tokens still attend jointly with the received-signal token, but
    each pilot group uses its own Q/K/V projection matrices:
        group g: Wq_b[g], Wk_b[g], Wv_b[g]
        y token: Wq_y, Wk_y, Wv_y

    The input order is assumed to be group-contiguous, matching
    data_correlated.py: [group 0 users, group 1 users, ...].
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        num_groups: int = 4,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_groups <= 0:
            raise ValueError("num_groups must be positive.")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_groups = num_groups

        self.wq_b_groups = nn.ModuleList(
            [nn.Linear(dim, num_heads * head_dim, bias=False) for _ in range(num_groups)]
        )
        self.wk_b_groups = nn.ModuleList(
            [nn.Linear(dim, num_heads * head_dim, bias=False) for _ in range(num_groups)]
        )
        self.wv_b_groups = nn.ModuleList(
            [nn.Linear(dim, num_heads * head_dim, bias=False) for _ in range(num_groups)]
        )

        self.wq_y = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.wk_y = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.wv_y = nn.Linear(dim, num_heads * head_dim, bias=False)

        self.wo_b = nn.Parameter(torch.randn(num_heads, dim, head_dim) * 0.02)
        self.wo_y = nn.Parameter(torch.randn(num_heads, dim, head_dim) * 0.02)
        self.attn_drop = nn.Dropout(attn_dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.num_heads, self.head_dim)

    def _split_groups(self, x_b: torch.Tensor) -> tuple[torch.Tensor, ...]:
        n = x_b.shape[1]
        if n % self.num_groups != 0:
            raise ValueError(
                f"num_devices={n} must be divisible by num_groups={self.num_groups}."
            )
        return torch.chunk(x_b, self.num_groups, dim=1)

    def _project_grouped(
        self,
        x_b: torch.Tensor,
        projections: nn.ModuleList,
    ) -> torch.Tensor:
        chunks = self._split_groups(x_b)
        projected = [
            self._split_heads(proj(x_group))
            for proj, x_group in zip(projections, chunks, strict=True)
        ]
        return torch.cat(projected, dim=1)

    def forward(self, x_b: torch.Tensor, x_y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, n, _ = x_b.shape

        q_b = self._project_grouped(x_b, self.wq_b_groups)
        k_b = self._project_grouped(x_b, self.wk_b_groups)
        v_b = self._project_grouped(x_b, self.wv_b_groups)

        q = torch.cat([q_b, self._split_heads(self.wq_y(x_y))], dim=1)
        k = torch.cat([k_b, self._split_heads(self.wk_y(x_y))], dim=1)
        v = torch.cat([v_b, self._split_heads(self.wv_y(x_y))], dim=1)

        qh = q.permute(0, 2, 1, 3)
        kh = k.permute(0, 2, 1, 3)
        vh = v.permute(0, 2, 1, 3)

        drop_p = self.attn_drop.p if self.training else 0.0
        ctx = F.scaled_dot_product_attention(
            qh,
            kh,
            vh,
            attn_mask=None,
            dropout_p=drop_p,
            is_causal=False,
        )
        ctx = ctx.permute(0, 2, 1, 3)
        ctx_b = ctx[:, :n]
        ctx_y = ctx[:, n:]

        out_b = torch.einsum("bnth,tdh->bnd", ctx_b, self.wo_b)
        out_y = torch.einsum("bnth,tdh->bnd", ctx_y, self.wo_y)
        return out_b, out_y


class HeterogeneousFFN(nn.Module):
    """论文 Eq. (9) 中的 component-wise FF 模块。

    FF_B 被所有设备导频 token 共享，因此参数量不随 N 增长。
    FF_Y 是接收信号 token 使用的另一套前馈网络。
    """

    def __init__(self, dim: int, ff_dim: int, ffn_dropout: float = 0.0) -> None:
        super().__init__()
        self.ff_b = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(ff_dim, dim),
        )
        self.ff_y = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(ff_dim, dim),
        )

    def forward(self, x_b: torch.Tensor, x_y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.ff_b(x_b), self.ff_y(x_y)


class HeterogeneousEncoderLayer(nn.Module):
    """一个异构编码层，对应论文 Eq. (8)-(9)。

    该层是针对 activity detection 改造的 Transformer encoder block：
    先做 MHA + 残差连接 + 归一化，再做 FF + 残差连接 + 归一化。
    其中设备导频 token 和接收信号 token 使用不同参数。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        ff_dim: int,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        norm_type: str = "batch",
    ) -> None:
        super().__init__()
        self.mha = HeterogeneousMHA(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            attn_dropout=attn_dropout,
        )
        self.ffn = HeterogeneousFFN(dim=dim, ff_dim=ff_dim, ffn_dropout=ffn_dropout)
        if norm_type == "layer":
            norm = TokenLayerNorm
        else:
            norm = TokenBatchNorm
        self.bn1_b = norm(dim)
        self.bn1_y = norm(dim)
        self.bn2_b = norm(dim)
        self.bn2_y = norm(dim)

    def forward(self, x_b: torch.Tensor, x_y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Eq. (8)：异构 MHA 后接 skip connection 和 BN/LN。
        mha_b, mha_y = self.mha(x_b, x_y)
        hat_b = self.bn1_b(x_b + mha_b)
        hat_y = self.bn1_y(x_y + mha_y)

        # Eq. (9)：不同 token 类型各自经过 FF，再接 skip connection 和 BN/LN。
        ff_b, ff_y = self.ffn(hat_b, hat_y)
        out_b = self.bn2_b(hat_b + ff_b)
        out_y = self.bn2_y(hat_y + ff_y)
        return out_b, out_y


class GroupedHeterogeneousEncoderLayer(nn.Module):
    """Encoder layer whose pilot-token attention projections are group-specific."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        ff_dim: int,
        num_groups: int = 4,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        norm_type: str = "batch",
    ) -> None:
        super().__init__()
        self.mha = GroupedHeterogeneousMHA(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            num_groups=num_groups,
            attn_dropout=attn_dropout,
        )
        self.ffn = HeterogeneousFFN(dim=dim, ff_dim=ff_dim, ffn_dropout=ffn_dropout)
        if norm_type == "layer":
            norm = TokenLayerNorm
        else:
            norm = TokenBatchNorm
        self.bn1_b = norm(dim)
        self.bn1_y = norm(dim)
        self.bn2_b = norm(dim)
        self.bn2_y = norm(dim)

    def forward(self, x_b: torch.Tensor, x_y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mha_b, mha_y = self.mha(x_b, x_y)
        hat_b = self.bn1_b(x_b + mha_b)
        hat_y = self.bn1_y(x_y + mha_y)

        ff_b, ff_y = self.ffn(hat_b, hat_y)
        out_b = self.bn2_b(hat_b + ff_b)
        out_y = self.bn2_y(hat_y + ff_y)
        return out_b, out_y


class ContextDecoder(nn.Module):
    """解码层，对应论文 Eq. (24)-(26) 和 Appendix A。

    contextual block 使用最终的接收信号 token x_{N+1}^{[L]} 作为 query，
    使用 N 个设备 token 加 x_{N+1}^{[L]} 作为 keys/values，计算 context vector x_c。
    output block 再将每个设备 token 与 x_c 做匹配打分，得到活动概率 P_n。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        score_scale: float,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.score_scale = score_scale

        # Appendix A Eq. (A.1)：context query 来自最终的 Y token。
        self.wq_c = nn.Linear(dim, num_heads * head_dim, bias=False)
        # Appendix A Eq. (A.2)-(A.3)：keys/values 对 B/Y token 使用不同参数。
        self.wk_cb = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.wv_cb = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.wk_cy = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.wv_cy = nn.Linear(dim, num_heads * head_dim, bias=False)
        # Appendix A Eq. (A.7) 和论文 Eq. (25) 使用的输出投影参数。
        self.wo_c = nn.Parameter(torch.randn(num_heads, dim, head_dim) * 0.02)
        self.w_out = nn.Parameter(torch.randn(dim, dim) * 0.02)
        self.attn_drop = nn.Dropout(attn_dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.num_heads, self.head_dim)

    def forward(self, x_b: torch.Tensor, x_y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Eq. (24)：context attention。Query 来自 Y；
        # keys/values 来自 {设备导频 token + Y}，输出一个 context vector。
        bsz, n, d = x_b.shape

        q = self._split_heads(self.wq_c(x_y))  # [B,1,H,Dh]
        k = torch.cat([self._split_heads(self.wk_cb(x_b)), self._split_heads(self.wk_cy(x_y))], dim=1)
        v = torch.cat([self._split_heads(self.wv_cb(x_b)), self._split_heads(self.wv_cy(x_y))], dim=1)

        qh = q.permute(0, 2, 1, 3)  # [B,H,1,Dh]
        kh = k.permute(0, 2, 1, 3)  # [B,H,N+1,Dh]
        vh = v.permute(0, 2, 1, 3)

        drop_p = self.attn_drop.p if self.training else 0.0
        ctx = F.scaled_dot_product_attention(
            qh,
            kh,
            vh,
            attn_mask=None,
            dropout_p=drop_p,
            is_causal=False,
        )  # [B,H,1,Dh]
        ctx = ctx.permute(0, 2, 1, 3)  # [B,1,H,Dh]
        xc = torch.einsum("bnth,tdh->bnd", ctx, self.wo_c).squeeze(1)  # [B,D]

        # Eq. (25)：计算 context vector x_c 与每个 device token 的匹配分数。
        # Eq. (26)：通过 sigmoid 将分数转换成每个设备的活动概率 P_n。
        xc_w = xc @ self.w_out  # [B,D]
        match = (xc_w.unsqueeze(1) * x_b).sum(dim=-1) / math.sqrt(d)
        logits = self.score_scale * torch.tanh(match)
        probs = torch.sigmoid(logits)
        return logits, probs


class HeterogeneousTransformer(nn.Module):
    """论文 baseline 的 Heterogeneous Transformer。

    默认参数对应论文仿真设置：
    L=5 个编码层，嵌入维度 d=128，注意力头数 T=8，每个头维度 d'=32，
    FF 隐层维度 d_f=512，输出缩放系数 C=10。
    """

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
        pilot_feature_dim: int | None = None,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        ctx_attn_dropout: float = 0.0,
        norm_type: str = "batch",
    ) -> None:
        super().__init__()
        self.num_devices = num_devices
        self.pilot_len = pilot_len
        self.dim = dim

        # 论文 Eq. (5) 和 Eq. (7)：每个设备导频 b_n 表示为
        # [Re(b_n), Im(b_n)]，维度为 R^{2Lp}，再用共享的 W_B^in 投影。
        self.pilot_feature_dim = pilot_feature_dim or (2 * pilot_len)
        self.embed_b = nn.Linear(self.pilot_feature_dim, dim)
        # 论文 Eq. (6) 和 Eq. (7)：Y 通过 vec(C) 表示，其中 C = YY^H / M，
        # 因此输入维度与基站天线数 M 无关。
        self.embed_y = nn.Linear(2 * pilot_len * pilot_len, dim)

        # 论文 Section III-B：堆叠 L 个异构编码层。
        self.layers = nn.ModuleList(
            [
                HeterogeneousEncoderLayer(
                    dim=dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    ff_dim=ff_dim,
                    attn_dropout=attn_dropout,
                    ffn_dropout=ffn_dropout,
                    norm_type=norm_type,
                )
                for _ in range(num_layers)
            ]
        )
        self.decoder = ContextDecoder(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            score_scale=score_scale,
            attn_dropout=ctx_attn_dropout,
        )

    def forward(self, x_b: torch.Tensor, x_y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x_b: [B, N, 2Lp]，保存设备导频的实部/虚部特征。
        # x_y: [B, 2Lp^2]，保存向量化协方差的实部/虚部特征。
        h_b = self.embed_b(x_b)
        h_y = self.embed_y(x_y).unsqueeze(1)

        for layer in self.layers:
            h_b, h_y = layer(h_b, h_y)

        return self.decoder(h_b, h_y)


class GroupedHeterogeneousTransformer(nn.Module):
    """Transformer for correlated activity groups.

    This keeps the original heterogeneous structure:
    - one shared pilot embedding W_B^in
    - one received-signal embedding W_Y^in
    - received-signal token has its own Q/K/V projections
    - decoder is unchanged

    The difference is inside every encoder MHA layer: pilot tokens in
    different correlation groups use different Wq/Wk/Wv matrices.
    """

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
        pilot_feature_dim: int | None = None,
        num_groups: int = 4,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        ctx_attn_dropout: float = 0.0,
        norm_type: str = "batch",
    ) -> None:
        super().__init__()
        if num_devices % num_groups != 0:
            raise ValueError(
                f"num_devices={num_devices} must be divisible by num_groups={num_groups}."
            )
        self.num_devices = num_devices
        self.pilot_len = pilot_len
        self.dim = dim
        self.num_groups = num_groups

        self.pilot_feature_dim = pilot_feature_dim or (2 * pilot_len)
        self.embed_b = nn.Linear(self.pilot_feature_dim, dim)
        self.embed_y = nn.Linear(2 * pilot_len * pilot_len, dim)

        self.layers = nn.ModuleList(
            [
                GroupedHeterogeneousEncoderLayer(
                    dim=dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    ff_dim=ff_dim,
                    num_groups=num_groups,
                    attn_dropout=attn_dropout,
                    ffn_dropout=ffn_dropout,
                    norm_type=norm_type,
                )
                for _ in range(num_layers)
            ]
        )
        self.decoder = ContextDecoder(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            score_scale=score_scale,
            attn_dropout=ctx_attn_dropout,
        )

    def forward(self, x_b: torch.Tensor, x_y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h_b = self.embed_b(x_b)
        h_y = self.embed_y(x_y).unsqueeze(1)

        for layer in self.layers:
            h_b, h_y = layer(h_b, h_y)

        return self.decoder(h_b, h_y)


def build_model_from_config(cfg: dict) -> nn.Module:
    """根据 checkpoint 或训练配置构造模型。

    "base" 复现历史基础 Transformer 输入方式，只使用 2*pilot_len 特征。
    "base-dimension" 使用当前 base 结构和 2*pilot_len+2 特征。
    """

    def _get_num(name: str, default: float) -> float:
        v = cfg.get(name, default)
        return default if v is None else v

    def _get_str(name: str, default: str) -> str:
        v = cfg.get(name, default)
        return default if v is None else str(v)

    model_name = str(cfg.get("model_name", "base")).lower()
    common = {
        "num_devices": cfg["num_devices"],
        "pilot_len": cfg["pilot_len"],
    }

    if model_name in {"grouped", "correlated", "grouped_correlated"}:
        return GroupedHeterogeneousTransformer(
            **common,
            dim=int(_get_num("dim", 128)),
            num_layers=int(_get_num("num_layers", 5)),
            num_heads=int(_get_num("num_heads", 8)),
            head_dim=int(_get_num("head_dim", 32)),
            ff_dim=int(_get_num("ff_dim", 512)),
            score_scale=float(_get_num("score_scale", 10.0)),
            pilot_feature_dim=int(_get_num("pilot_feature_dim", 2 * int(common["pilot_len"]))),
            num_groups=int(_get_num("num_groups", 4)),
            attn_dropout=float(_get_num("attn_dropout", 0.0)),
            ffn_dropout=float(_get_num("ffn_dropout", 0.0)),
            ctx_attn_dropout=float(_get_num("ctx_attn_dropout", 0.0)),
            norm_type=_get_str("norm_type", "batch"),
        )

    return HeterogeneousTransformer(
        **common,
        dim=int(_get_num("dim", 128)),
        num_layers=int(_get_num("num_layers", 5)),
        num_heads=int(_get_num("num_heads", 8)),
        head_dim=int(_get_num("head_dim", 32)),
        ff_dim=int(_get_num("ff_dim", 512)),
        score_scale=float(_get_num("score_scale", 10.0)),
        pilot_feature_dim=int(_get_num("pilot_feature_dim", 2 * int(common["pilot_len"]))),
        attn_dropout=float(_get_num("attn_dropout", 0.0)),
        ffn_dropout=float(_get_num("ffn_dropout", 0.0)),
        ctx_attn_dropout=float(_get_num("ctx_attn_dropout", 0.0)),
        norm_type=_get_str("norm_type", "batch"),
    )
