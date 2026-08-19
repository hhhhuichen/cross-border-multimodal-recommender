# -*- coding: utf-8 -*-
"""独立推荐基线：BPR-MF、标准 LightGCN 与 VBPR。"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import bpr_loss, degree_norm_weight, scatter_sum


class _PairwiseBaseline(nn.Module):
    """与训练/评估入口共享的最小接口。"""

    def __init__(self, dataset, cfg):
        super().__init__()
        self.cfg = cfg
        self.mc = cfg.model
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        self.n_entities = dataset.n_entities
        self.n_nodes = dataset.n_nodes
        self.n_relations = dataset.n_relations
        dim = cfg.model.embed_dim
        self.user_emb = nn.Embedding(self.n_users, dim)
        self.item_emb = nn.Embedding(self.n_items, dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        self._att_cache = None

    def _item_keep(self, batch):
        keep = torch.ones(
            self.n_items, 1, device=self.item_emb.weight.device,
            dtype=self.item_emb.weight.dtype,
        )
        cold = batch.get("cold_items")
        if cold is not None and cold.numel():
            keep[cold] = 0.0
        zero_train = batch.get("zero_train_items")
        if zero_train is not None:
            if zero_train.numel():
                keep[zero_train] = 0.0
        else:
            degree = batch.get("item_train_degree")
            if degree is None:
                degree = batch.get("item_train_interaction_count")
            if degree is not None:
                keep[degree <= 0] = 0.0
        return keep

    def refresh_attention(self, batch):
        self._att_cache = None
        return None

    def get_embeddings(self, batch):
        return self.user_emb.weight, self.item_emb.weight * self._item_keep(batch)

    def score(self, batch, user_e, item_e, users, items):
        return (user_e[users] * item_e[items]).sum(dim=-1)

    def full_score(self, batch, user_e, item_e, users):
        return user_e[users] @ item_e.t()

    def precompute_item_tables(self, batch, item_e):
        return {"item_e": item_e}

    def full_score_cached(self, batch, user_e, item_e, users, tables):
        return user_e[users] @ tables["item_e"].t()

    def cf_loss(self, batch, users, pos, neg):
        user_e, item_e = self.get_embeddings(batch)
        loss = bpr_loss(
            self.score(batch, user_e, item_e, users, pos),
            self.score(batch, user_e, item_e, users, neg),
        )
        reg = self.mc.l2_reg * (
            self.user_emb(users).pow(2).sum()
            + self.item_emb(pos).pow(2).sum()
            + self.item_emb(neg).pow(2).sum()
        ) / max(len(users), 1)
        return loss + reg


class BPRMF(_PairwiseBaseline):
    """Rendle et al. 的隐式反馈矩阵分解基线。"""


class LightGCN(_PairwiseBaseline):
    """仅在训练 user-item 二部图上传播的标准 LightGCN。"""

    def refresh_attention(self, batch):
        edge_type = batch["edge_type"]
        reverse_interaction = getattr(
            self.cfg.model, "reverse_interaction_relation", None
        )
        if reverse_interaction is None:
            # CKG 的约定为 0=interact，n_relations+1=reverse interact。
            reverse_interaction = self.n_relations + 1
        interaction = (edge_type == 0) | (edge_type == reverse_interaction)
        edge_index = batch["edge_index"][:, interaction]
        self._interaction_edges = edge_index
        self._att_cache = degree_norm_weight(edge_index, self.n_nodes)
        return self._att_cache

    def get_embeddings(self, batch):
        if self._att_cache is None:
            self.refresh_attention(batch)
        dim = self.item_emb.embedding_dim
        entity_padding = torch.zeros(
            self.n_entities - self.n_items, dim,
            device=self.item_emb.weight.device, dtype=self.item_emb.weight.dtype,
        )
        x = torch.cat([
            self.item_emb.weight * self._item_keep(batch),
            entity_padding,
            self.user_emb.weight,
        ], dim=0)
        outputs = [x]
        src, dst = self._interaction_edges
        for _ in range(self.mc.n_gnn_layers):
            x = scatter_sum(
                x[src] * self._att_cache.unsqueeze(-1), dst, self.n_nodes
            )
            outputs.append(x)
        out = torch.stack(outputs, dim=0).mean(dim=0)
        return out[self.n_entities:], out[:self.n_items]


class VBPR(_PairwiseBaseline):
    """在 MF 分数上增加用户特定视觉偏好的 VBPR 基线。"""

    def __init__(self, dataset, cfg):
        super().__init__(dataset, cfg)
        dim = cfg.model.embed_dim
        self.visual_user_emb = nn.Embedding(self.n_users, dim)
        self.visual_item_proj = nn.Linear(cfg.data.image_dim, dim, bias=False)
        nn.init.xavier_uniform_(self.visual_user_emb.weight)
        nn.init.xavier_uniform_(self.visual_item_proj.weight)

    def _visual_items(self, batch, items=None):
        image = batch["item_image"] if items is None else batch["item_image"][items]
        mask = (batch["item_image_mask"] if items is None
                else batch["item_image_mask"][items]).unsqueeze(-1)
        image = F.normalize(image, dim=-1, eps=1e-12)
        return self.visual_item_proj(image) * mask

    def score(self, batch, user_e, item_e, users, items):
        collaborative = super().score(batch, user_e, item_e, users, items)
        visual = (self.visual_user_emb(users)
                  * self._visual_items(batch, items)).sum(dim=-1)
        return collaborative + visual

    def full_score(self, batch, user_e, item_e, users):
        collaborative = super().full_score(batch, user_e, item_e, users)
        return collaborative + self.visual_user_emb(users) @ self._visual_items(batch).t()

    def precompute_item_tables(self, batch, item_e):
        return {"item_e": item_e, "visual_items": self._visual_items(batch)}

    def full_score_cached(self, batch, user_e, item_e, users, tables):
        collaborative = user_e[users] @ tables["item_e"].t()
        return (collaborative
                + self.visual_user_emb(users) @ tables["visual_items"].t())

    def cf_loss(self, batch, users, pos, neg):
        loss = super().cf_loss(batch, users, pos, neg)
        visual_reg = self.mc.l2_reg * (
            self.visual_user_emb(users).pow(2).sum()
            + self.visual_item_proj.weight.pow(2).mean()
        ) / max(len(users), 1)
        return loss + visual_reg


def build_model(dataset, cfg=None, model_name=None):
    """构造预注册模型。

    首选 ``build_model(dataset, cfg, model_name)``，同时兼容训练入口早期使用的
    ``build_model(model_name, dataset, cfg)`` 调用顺序。
    """
    if isinstance(dataset, str):
        model_name, dataset, cfg = dataset, cfg, model_name
    if cfg is None:
        raise TypeError("build_model 缺少 cfg")
    if model_name is None:
        model_name = getattr(cfg.train, "model", "acmr")
    models = {
        "bpr_mf": BPRMF,
        "lightgcn": LightGCN,
        "vbpr": VBPR,
    }
    if model_name == "acmr":
        from model import ACMR
        return ACMR(dataset, cfg)
    try:
        model_class = models[model_name]
    except KeyError as exc:
        raise ValueError(f"未知 model: {model_name!r}") from exc
    return model_class(dataset, cfg)
