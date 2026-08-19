# -*- coding: utf-8 -*-
"""模型组件：稀疏图算子、多模态编码与融合、关系感知 GNN 层、跨境自适应模块。"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 稀疏图基础算子（不依赖 torch_scatter / PyG，纯 PyTorch 实现）
# --------------------------------------------------------------------------- #
def scatter_sum(src, index, num_nodes):
    shape = (num_nodes,) + src.shape[1:]
    out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    return out.scatter_add_(0, idx, src)


_HAS_SCATTER_REDUCE = hasattr(torch.Tensor, "scatter_reduce")


def scatter_softmax(src, index, num_nodes):
    """
    按目标节点分组做 softmax。src: (E,)

    分组 softmax 对"减去任意常数"是不变的，所以数值稳定的平移量既可以用
    组内最大值，也可以用全局最大值。torch>=1.12 用 scatter_reduce 取组内
    最大值（更紧）；老版本没有该算子，回退到全局最大值——同样精确，只是
    平移量松一些。
    """
    if _HAS_SCATTER_REDUCE:
        m = torch.full((num_nodes,), float("-inf"), dtype=src.dtype, device=src.device)
        m = m.scatter_reduce(0, index, src, reduce="amax", include_self=True)
        shift = torch.nan_to_num(m, neginf=0.0)[index]
    else:
        shift = src.max().detach()

    ex = (src - shift).exp()
    denom = scatter_sum(ex, index, num_nodes)
    # 用 dtype 的最小正数兜底，而不是硬编码 1e-16：
    # 后者在组内分数远低于平移量时会引入千分之一量级的相对误差
    return ex / denom[index].clamp_min(torch.finfo(src.dtype).tiny)


# --------------------------------------------------------------------------- #
# 梯度反转层（语种对抗，用于学习"语言无关"的商品表示）
# --------------------------------------------------------------------------- #
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambd * grad, None


def grad_reverse(x, lambd=1.0):
    return _GradReverse.apply(x, lambd)


# --------------------------------------------------------------------------- #
# 多模态编码 + 融合
# --------------------------------------------------------------------------- #
class ModalityEncoder(nn.Module):
    """把冻结的预训练特征（LaBSE 文本 / CLIP 图像）投影到统一的 d 维空间。"""

    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.LayerNorm(out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class GatedFusion(nn.Module):
    """
    门控多模态融合。缺失模态（如无图商品）由 mask 置零，
    门控网络会自动把权重转移到可用模态上。
    """

    def __init__(self, dim, n_modalities, mode="gated"):
        super().__init__()
        self.mode = mode
        self.n = n_modalities
        if mode == "gated":
            self.gate = nn.Sequential(
                nn.Linear(dim * n_modalities, dim * n_modalities),
                nn.Sigmoid(),
            )
            self.proj = nn.Linear(dim * n_modalities, dim)
        elif mode == "concat":
            self.proj = nn.Linear(dim * n_modalities, dim)
        elif mode == "attention":
            self.q = nn.Parameter(torch.randn(dim))
            self.proj = nn.Linear(dim, dim)
        elif mode != "sum":
            raise ValueError(f"未知融合模式: {mode!r}")
        self.norm = nn.LayerNorm(dim)

    def forward(self, mods, masks=None):
        """mods: list of (N, d)；masks: list of (N, 1) 或 None"""
        if masks is not None:
            mods = [m * k for m, k in zip(mods, masks)]
        if self.mode == "sum":
            out = torch.stack(mods, 0).sum(0)
        elif self.mode == "attention":
            stack = torch.stack(mods, 1)                      # (N, M, d)
            score = (stack * self.q).sum(-1)                  # (N, M)
            if masks is not None:
                mk = torch.cat(masks, 1)                      # (N, M)
                score = score.masked_fill(mk < 0.5, -1e9)
            w = torch.softmax(score, -1).unsqueeze(-1)
            out = self.proj((stack * w).sum(1))
        else:
            cat = torch.cat(mods, -1)
            if self.mode == "gated":
                cat = cat * self.gate(cat)
            out = self.proj(cat)
        return self.norm(out)


# --------------------------------------------------------------------------- #
# 关系感知 GNN 层（KGAT 风格）
# --------------------------------------------------------------------------- #
class RelationalGNNLayer(nn.Module):
    """
    注意力：  pi(h, r, t) = (W_r e_t)^T · tanh(W_r e_h + e_r)
    聚合器：
        bi        : LeakyReLU(W1(e + n)) + LeakyReLU(W2(e * n))     [KGAT]
        gcn       : LeakyReLU(W(e + n))
        graphsage : LeakyReLU(W[e || n])
    """

    def __init__(self, in_dim, out_dim, aggregator="bi", dropout=0.1):
        super().__init__()
        self.aggregator = aggregator
        if aggregator == "bi":
            self.W1 = nn.Linear(in_dim, out_dim)
            self.W2 = nn.Linear(in_dim, out_dim)
        elif aggregator == "gcn":
            self.W1 = nn.Linear(in_dim, out_dim)
        elif aggregator == "graphsage":
            self.W1 = nn.Linear(in_dim * 2, out_dim)
        else:
            raise ValueError(f"未知聚合器: {aggregator!r}")
        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index, att):
        """x: (N, d)  edge_index: (2, E)  att: (E,) 已归一化的注意力权重"""
        src, dst = edge_index[0], edge_index[1]
        msg = x[src] * att.unsqueeze(-1)
        neigh = scatter_sum(msg, dst, x.size(0))

        if self.aggregator == "bi":
            out = self.act(self.W1(x + neigh)) + self.act(self.W2(x * neigh))
        elif self.aggregator == "gcn":
            out = self.act(self.W1(x + neigh))
        else:
            out = self.act(self.W1(torch.cat([x, neigh], dim=-1)))
        return self.dropout(out)


class RelationAttention(nn.Module):
    """仅为紧凑的正反 KG 关系计算注意力，不包含交互关系参数。"""

    def __init__(self, n_relations, embed_dim, relation_dim):
        super().__init__()
        self.rel_emb = nn.Embedding(n_relations, relation_dim)
        self.W_r = nn.Parameter(torch.empty(n_relations, embed_dim, relation_dim))
        nn.init.xavier_uniform_(self.W_r)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(self, x, edge_index, edge_type, num_nodes):
        """
        按关系类型分组计算。若直接用 W_r[edge_type] 会得到 (E, d, k) 的张量，
        E 上百万时显存直接爆掉；分组后每种关系只做一次 (E_r, d) @ (d, k) 矩阵乘。
        """
        src, dst = edge_index[0], edge_index[1]
        score = torch.zeros(edge_index.size(1), device=x.device, dtype=x.dtype)
        for r in torch.unique(edge_type):
            m = edge_type == r
            Wr = self.W_r[r]                                  # (d, k)
            h = x[src[m]] @ Wr                                # (E_r, k)
            t = x[dst[m]] @ Wr
            score[m] = (t * torch.tanh(h + self.rel_emb(r))).sum(-1)
        return scatter_softmax(score, dst, num_nodes)


def degree_norm_weight(edge_index, num_nodes):
    """对称度归一化 1/sqrt(d_src * d_dst)，用于消融掉 KG 时的 LightGCN 退化版本。"""
    src, dst = edge_index[0], edge_index[1]
    ones = torch.ones(edge_index.size(1), device=edge_index.device)
    deg = scatter_sum(ones, dst, num_nodes).clamp(min=1)
    return (deg[src].pow(-0.5) * deg[dst].pow(-0.5))


# --------------------------------------------------------------------------- #
# 跨境市场自适应门控
# --------------------------------------------------------------------------- #
class MarketGate(nn.Module):
    """
    用目标市场的国别嵌入调制表示：
    同一商品在不同目标市场应有不同的呈现权重，这是跨境推荐区别于单市场推荐的核心。

    可选国家关系图（country_adj，地理邻接/贸易强度）：国别嵌入先在图上做一阶
    对称归一化传播再参与门控，让相邻/贸易紧密的市场共享信息——小语种市场的
    国别表示不再只靠自身稀疏交互学习。
    """

    def __init__(self, n_countries, dim, country_adj=None):
        super().__init__()
        self.country_emb = nn.Embedding(n_countries, dim)
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        nn.init.xavier_uniform_(self.country_emb.weight)
        if country_adj is not None:
            A = country_adj + torch.eye(n_countries)          # 加自环
            d = A.sum(-1).clamp(min=1).pow(-0.5)
            # 邻接矩阵属于当前数据而非模型参数，不写入 checkpoint，避免加载旧
            # 权重时把另一份数据集的国家图一起覆盖进来。
            self.register_buffer(
                "adj", d.unsqueeze(-1) * A * d.unsqueeze(0), persistent=False
            )
        else:
            self.adj = None

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # 兼容旧 checkpoint（旧版把数据派生的 adj 持久化在 state_dict 中）。
        state_dict.pop(prefix + "adj", None)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def country_table(self):
        E = self.country_emb.weight
        if self.adj is not None:
            E = 0.5 * E + 0.5 * self.adj @ E                  # 保留自身 + 邻国混入
        return E

    def forward(self, x, country_ids):
        c = self.country_table()[country_ids]
        g = self.gate(torch.cat([x, c], dim=-1))
        return x * g + c * (1 - g)


# --------------------------------------------------------------------------- #
# 损失函数
# --------------------------------------------------------------------------- #
def bpr_loss(pos_score, neg_score):
    return -F.logsigmoid(pos_score - neg_score).mean()


def info_nce(z1, z2, temperature=0.2):
    """跨语言对齐：同一商品的不同语种视图互为正样本，batch 内其余为负样本。"""
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = z1 @ z2.t() / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels)
                  + F.cross_entropy(logits.t(), labels))
