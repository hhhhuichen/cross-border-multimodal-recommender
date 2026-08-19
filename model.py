# -*- coding: utf-8 -*-
"""
ACMR: ASEAN Cross-border Multimodal Recommender

整体结构
--------
   多语种文本(LaBSE，多视图聚合) ─┐
                                  ├─► 门控融合 ─► 商品内容表示 ─┐
   商品图像(CLIP)   ─┘                              │(注入)
                                                    ▼
   ID Embedding ──────────────────────► CKG 节点初始表示
                                                    │
                        关系感知注意力 + L 层 GNN 传播（KGAT-inspired）
                                                    │
                         各层输出拼接 ─► 用户 / 商品最终表示
                                                    │
                             跨境市场门控 ─► 内积打分 ─► BPR

联合优化目标
   L = L_BPR + λ_kg·L_TransR-style + λ_align·L_跨语言对比 + λ_adv·L_语种对抗
       + λ_ccl·L_协同内容对比 + λ_L2
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import (
    ModalityEncoder, GatedFusion, RelationalGNNLayer, RelationAttention,
    MarketGate, grad_reverse, bpr_loss, info_nce, degree_norm_weight,
    scatter_sum,
)


class ACMR(nn.Module):
    def __init__(self, dataset, cfg):
        super().__init__()
        self.cfg = cfg
        m, d = cfg.model, cfg.data
        self.mc = m
        if m.use_mm_item_graph:
            if not m.use_multimodal:
                raise ValueError("冻结多模态商品图要求 use_multimodal=True")
            if not 1 <= m.mm_graph_layers <= m.n_gnn_layers:
                raise ValueError(
                    "mm_graph_layers 必须满足 1 <= mm_graph_layers "
                    "<= n_gnn_layers"
                )
            if not 0.0 <= m.mm_graph_beta <= 1.0:
                raise ValueError("mm_graph_beta 必须在 [0,1] 内")

        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        self.n_entities = dataset.n_entities
        self.n_nodes = dataset.n_nodes
        self.n_rel_total = dataset.n_relations_total
        self.n_relations = dataset.n_relations
        # interact(0) 与 reverse-interact(R+1) 只用对称度传播，不拥有
        # TransR/关系注意力参数。其余正反 KG 关系紧凑映射为 0..2R-1。
        self.n_kg_relations = 2 * self.n_relations
        self.n_countries = dataset.n_countries
        dim = m.embed_dim
        out_dim = dim * (m.n_gnn_layers + 1)
        self.residual_mode = getattr(m, "residual", getattr(m, "residual_mode", "none"))
        if self.residual_mode not in {
                "none", "fused", "decoupled", "market_reliable"}:
            raise ValueError(f"未知 residual 模式: {self.residual_mode!r}")
        if self.residual_mode != "none" and not m.use_multimodal:
            raise ValueError("内容残差要求 use_multimodal=True")

        # ---------- 1. ID Embedding ----------
        self.entity_emb = nn.Embedding(self.n_entities, dim)
        self.user_emb = nn.Embedding(self.n_users, dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.user_emb.weight)

        # ---------- 2. 多模态编码与融合 ----------
        if m.use_multimodal:
            self.text_enc = ModalityEncoder(d.text_dim, dim, m.layer_dropout)
            self.image_enc = ModalityEncoder(d.image_dim, dim, m.layer_dropout)
            self.fusion = GatedFusion(dim, 2, m.fusion)
            # 内容表示注入 ID 空间时的缩放系数（可学习）
            self.content_alpha = nn.Parameter(torch.tensor(0.5))
            # 跨模态补全：文本 -> 图像特征空间的预测网络，缺图商品用预测值顶上
            if m.use_modality_completion:
                self.img_from_text = nn.Sequential(
                    nn.Linear(d.text_dim, dim * 4), nn.GELU(),
                    nn.Linear(dim * 4, d.image_dim),
                )
                self.completion_confidence_head = nn.Sequential(
                    nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1), nn.Sigmoid()
                )

            if self.residual_mode != "none":
                # B1--B3 始终实例化相同参数，保证消融的参数量严格匹配。
                self.residual_query = nn.Linear(out_dim, dim * 2)
                self.residual_fused_query = nn.Linear(out_dim, dim)
                self.residual_fused_item = nn.Linear(dim, dim)
                self.residual_user_weight = nn.Linear(out_dim, 2)
                self.residual_market_weight = nn.Embedding(dataset.n_countries, 2)
                self.residual_reliability_weight = nn.Linear(6, 2)
                beta_init = float(getattr(m, "residual_beta_init", 0.01))
                # softplus 参数化保证训练中 beta 始终非负。
                beta_raw = torch.log(torch.expm1(torch.tensor(beta_init)))
                self.residual_beta_raw = nn.Parameter(beta_raw)
                nn.init.zeros_(self.residual_user_weight.bias)
                nn.init.zeros_(self.residual_reliability_weight.bias)
                nn.init.zeros_(self.residual_market_weight.weight)

        # ---------- 3. 语种对抗判别器 ----------
        if m.use_multimodal and m.use_lang_adversarial:
            self.lang_disc = nn.Sequential(
                nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dataset.n_languages)
            )

        # ---------- 4. 关系感知 GNN ----------
        self.rel_att = RelationAttention(self.n_kg_relations, dim, m.relation_dim)
        self.gnn_layers = nn.ModuleList([
            RelationalGNNLayer(dim, dim, m.aggregator, m.message_dropout)
            for _ in range(m.n_gnn_layers)
        ])

        # ---------- 5. 跨境市场门控（可选国家关系图传播）----------
        if m.use_market_gate:
            adj = None
            if m.use_country_graph and getattr(dataset, "country_adj", None) is not None:
                adj = torch.as_tensor(dataset.country_adj, dtype=torch.float32)
            self.user_gate = MarketGate(dataset.n_countries, out_dim, adj)
            self.item_gate = MarketGate(dataset.n_countries, out_dim, adj)

        self._att_cache = None      # 注意力缓存，训练中按 epoch 刷新
        self._eval_modality_cache = None
        self._eval_market_item_cache = {}
        self._eval_item_tables = None

    def compact_kg_relations(self, relations):
        """把 CKG 边类型映射到不含交互的紧凑 KG 参数索引。"""
        relations = relations.long()
        forward = (relations >= 1) & (relations <= self.n_relations)
        reverse = ((relations >= self.n_relations + 2)
                   & (relations < self.n_rel_total))
        valid = forward | reverse
        if not bool(valid.all()):
            bad = relations[~valid][:5].detach().cpu().tolist()
            raise ValueError(f"KG 关系中含 interaction 或越界编号: {bad}")
        return torch.where(forward, relations - 1, relations - 2)

    def upgrade_state_dict(self, state_dict):
        """兼容删除 interaction TransR 行之前的 checkpoint。"""
        keys = ("rel_att.W_r", "rel_att.rel_emb.weight")
        legacy = any(
            key in state_dict and state_dict[key].shape[0] == self.n_rel_total
            for key in keys
        )
        if not legacy:
            return state_dict
        upgraded = state_dict.copy()
        keep = torch.cat([
            torch.arange(1, self.n_relations + 1),
            torch.arange(self.n_relations + 2, self.n_rel_total),
        ])
        for key in keys:
            value = upgraded.get(key)
            if value is not None and value.shape[0] == self.n_rel_total:
                upgraded[key] = value.index_select(0, keep.to(value.device))
        return upgraded

    def train(self, mode=True):
        result = super().train(mode)
        if mode:
            self._eval_modality_cache = None
            self._eval_market_item_cache = {}
            self._eval_item_tables = None
        return result

    # ------------------------------------------------------------------ #
    # 内容表示
    # ------------------------------------------------------------------ #
    @staticmethod
    def _deduplicated_view_mask(text_feat, valid=None, content_hash=None,
                                deduplicated=None):
        """保留每个规范化内容哈希的首个有效视图。无哈希的旧批次不猜测。"""
        n, n_views = text_feat.shape[:2]
        if valid is None:
            keep = torch.ones(n, n_views, dtype=torch.bool, device=text_feat.device)
        else:
            keep = valid.bool().clone()
        if deduplicated is not None:
            return keep & deduplicated.bool()
        if content_hash is None:
            return keep
        known = content_hash >= 0
        for view in range(1, n_views):
            duplicate = torch.zeros(n, dtype=torch.bool, device=text_feat.device)
            for previous in range(view):
                duplicate |= (known[:, view] & known[:, previous]
                              & (content_hash[:, view] == content_hash[:, previous])
                              & keep[:, previous])
            keep[:, view] &= ~duplicate
        return keep

    @staticmethod
    def _batch_value(batch, names, item_ids=None):
        for name in names:
            value = batch.get(name)
            if value is not None:
                return value if item_ids is None else value[item_ids]
        return None

    def _text_metadata(self, batch, item_ids=None):
        return {
            "text_valid": self._batch_value(
                batch, ("item_text_valid", "item_text_view_valid"), item_ids),
            "text_fallback": self._batch_value(
                batch, ("item_text_is_fallback", "item_text_fallback"), item_ids),
            "content_hash": self._batch_value(
                batch, ("item_text_content_hash", "item_text_hash"), item_ids),
            "text_dedup": self._batch_value(
                batch, ("item_text_dedup_mask",), item_ids),
        }

    def _encode_modalities(self, text_feat, image_feat, image_mask, *,
                           text_valid=None, text_fallback=None,
                           content_hash=None, text_dedup=None, view=None):
        """返回融合前后的表示及可靠性；图像塔只接收 L2 归一化特征。"""
        n, n_views, in_dim = text_feat.shape
        view_mask = self._deduplicated_view_mask(
            text_feat, text_valid, content_hash, text_dedup
        )
        if view is not None:
            selected = torch.zeros_like(view_mask)
            selected[:, view] = view_mask[:, view]
            view_mask = selected

        encoded_views = self.text_enc(text_feat.reshape(n * n_views, in_dim))
        encoded_views = encoded_views.reshape(n, n_views, -1)
        weight = view_mask.to(text_feat.dtype).unsqueeze(-1)
        denom = weight.sum(dim=1).clamp_min(1.0)
        text = (encoded_views * weight).sum(dim=1) / denom
        pooled_raw = (text_feat * weight).sum(dim=1) / denom
        text_available = view_mask.any(dim=1, keepdim=True).to(text_feat.dtype)
        text = text * text_available

        observed = (image_mask > 0.5).to(text_feat.dtype).unsqueeze(-1)
        observed_image = F.normalize(image_feat, dim=-1, eps=1e-12)
        completion_confidence = torch.zeros_like(observed)
        if self.mc.use_modality_completion:
            predicted_image = F.normalize(
                self.img_from_text(pooled_raw), dim=-1, eps=1e-12
            )
            completion_confidence = (
                self.completion_confidence_head(text) * text_available
            )
            image_input = observed * observed_image + (1.0 - observed) * predicted_image
            image_available = torch.maximum(observed, text_available)
            image_reliability = observed + (1.0 - observed) * completion_confidence
        else:
            image_input = observed_image
            image_available = observed
            image_reliability = observed
        image = self.image_enc(image_input) * image_available
        fused = self.fusion(
            [text, image], masks=[text_available, image_reliability]
        )
        overall_available = torch.maximum(text_available, image_available)
        fused = fused * overall_available
        return {
            "text": text,
            "image": image,
            "fused": fused,
            "pooled_raw_text": pooled_raw,
            "view_mask": view_mask,
            "text_available": text_available,
            "image_available": image_available,
            "image_observed": observed,
            "completion_confidence": completion_confidence,
            "overall_available": overall_available,
        }

    def encode_content(self, text_feat, image_feat, image_mask, view=None, **metadata):
        """兼容旧接口；schema v2 元数据通过可选关键字传入。"""
        return self._encode_modalities(
            text_feat, image_feat, image_mask, view=view, **metadata
        )["fused"]

    def _batch_modalities(self, batch, item_ids=None):
        use_cache = (item_ids is None and not self.training
                     and not torch.is_grad_enabled())
        if use_cache and self._eval_modality_cache is not None:
            cache_key, modalities = self._eval_modality_cache
            if cache_key == id(batch):
                return modalities
        text = batch["item_text"] if item_ids is None else batch["item_text"][item_ids]
        image = batch["item_image"] if item_ids is None else batch["item_image"][item_ids]
        image_mask = (batch["item_image_mask"] if item_ids is None
                      else batch["item_image_mask"][item_ids])
        modalities = self._encode_modalities(
            text, image, image_mask, **self._text_metadata(batch, item_ids)
        )
        if use_cache:
            self._eval_modality_cache = (id(batch), modalities)
        return modalities

    def _item_id_keep(self, batch):
        """为 cold item 与训练期 ID dropout 生成一次共享掩码。"""
        keep = torch.ones(
            self.n_items, 1, device=self.entity_emb.weight.device
        )
        cold = batch.get("cold_items")
        if cold is not None and cold.numel():
            keep[cold] = 0.0
        zero_train = batch.get("zero_train_items")
        if zero_train is not None:
            if zero_train.numel():
                keep[zero_train] = 0.0
        else:
            train_degree = self._batch_value(
                batch, ("item_train_degree", "item_train_interaction_count")
            )
            if train_degree is not None:
                keep[train_degree <= 0] = 0.0
            has_train = batch.get("item_has_train_interaction")
            if has_train is not None:
                keep[~has_train.bool()] = 0.0
        if self.training and self.mc.cold_id_dropout > 0:
            keep = keep * (
                torch.rand(self.n_items, 1, device=keep.device)
                > self.mc.cold_id_dropout
            ).float()
        return keep

    def item_graph_id_features(self, item_id, batch):
        """沿冻结多模态图传播商品 ID，并对齐到主干的层拼接维度。"""
        if (not self.mc.use_mm_item_graph or self.mc.mm_graph_beta <= 0.0
                or self.mc.mm_graph_layers <= 0):
            return None
        edge_index = batch.get("mm_item_edge_index")
        edge_weight = batch.get("mm_item_edge_weight")
        if edge_index is None or edge_index.numel() == 0:
            return None

        src, dst = edge_index
        x = item_id
        zero = torch.zeros_like(item_id)
        outs = [zero]  # 主干第 0 层已经含自身 ID，避免在 side branch 重复相加。
        for layer in range(self.mc.n_gnn_layers):
            if layer < self.mc.mm_graph_layers:
                previous = x
                x = scatter_sum(
                    previous[src] * edge_weight.unsqueeze(-1), dst, self.n_items
                )
                active = scatter_sum(
                    torch.ones_like(edge_weight), dst, self.n_items
                ) > 0
                # 商品图是加到主干上的 side branch。孤立商品若沿用 previous，
                # 会把自身 ID 再加一次；零贡献才会保持主干表示不变。
                x = torch.where(active.unsqueeze(-1), x, zero)
                outs.append(F.normalize(x, dim=-1))
            else:
                outs.append(zero)
        return torch.cat(outs, dim=-1)

    def node_features(self, batch, item_keep=None):
        """构造 CKG 的第 0 层节点表示。"""
        ent = self.entity_emb.weight
        # DropoutNet-inspired 商品 ID dropout（不是原论文两阶段蒸馏）：
        #   训练期：随机抹掉部分商品的 ID 表示，逼内容 + KG 通道独立撑起排序；
        #   全程：已知新品（cold_items）的 ID 表示恒置零——它没被训练过，留着
        #   只是注入随机噪声；置零后恰好是 dropout 训练见过的分布内输入。
        if item_keep is None:
            item_keep = self._item_id_keep(batch)
        ent = torch.cat([
            ent[: self.n_items] * item_keep, ent[self.n_items:]
        ], dim=0)
        if self.mc.use_multimodal:
            content = self._batch_modalities(batch)["fused"]
            # LATTICE/FREEDOM 在侧信息图上传播 ID embeddings；本实现把它作为
            # ACMR side branch，多模态特征只用于构图，并非独立论文基线。
            pad = torch.zeros(
                self.n_entities - self.n_items, content.size(1), device=content.device
            )
            ent = ent + self.content_alpha * torch.cat([content, pad], dim=0)
        return torch.cat([ent, self.user_emb.weight], dim=0)

    # ------------------------------------------------------------------ #
    # 图传播
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def refresh_attention(self, batch):
        """
        重算并缓存 CKG 上的边权重，每个 epoch 刷新一次，不参与 CF 阶段的反传
            （否则每个 batch 都要在全图上重算）。这是 KGAT-inspired 缓存改造。

        混合传播设计：
          * interact 及其反向边：LightGCN 对称度归一化——CF 主干。KGAT 的统一
            softmax 会让每个节点入边权重和恒为 1，可能加重高阶节点过平滑；
          * KG 边：关系感知注意力（W_r、关系嵌入来自 KG 表示学习阶段），在 KG
            边内部按目标节点 softmax 归一化后乘以 γ（kg_att_scale）叠加，
            让 KG 作为有界的补充信号注入，而不是与 CF 邻居争抢归一化质量。
        """
        edge_index, edge_type = batch["edge_index"], batch["edge_type"]
        if not self.mc.use_kg:
            self._att_cache = degree_norm_weight(edge_index, self.n_nodes)
        else:
            is_int = (edge_type == 0) | (edge_type == self.n_relations + 1)
            att = torch.zeros(
                edge_index.size(1), device=edge_index.device, dtype=torch.float32
            )
            att[is_int] = degree_norm_weight(edge_index[:, is_int], self.n_nodes)

            was_training = self.training
            self.eval()
            x0 = self.node_features(batch)
            kg_att = self.rel_att(
                x0, edge_index[:, ~is_int],
                self.compact_kg_relations(edge_type[~is_int]), self.n_nodes
            )
            if was_training:
                self.train()
            att[~is_int] = self.mc.kg_att_scale * kg_att
            self._att_cache = att
        return self._att_cache

    def propagate(self, batch):
        item_keep = self._item_id_keep(batch)
        x0 = self.node_features(batch, item_keep=item_keep)
        edge_index = batch["edge_index"]

        if self._att_cache is None:
            self.refresh_attention(batch)
        att = self._att_cache

        x, outs = x0, [F.normalize(x0, dim=-1)]
        for layer in self.gnn_layers:
            if self.mc.use_kg:
                x = layer(x, edge_index, att)
            else:
                src, dst = edge_index
                x = scatter_sum(x[src] * att.unsqueeze(-1), dst, self.n_nodes)
            outs.append(F.normalize(x, dim=-1))
        out = torch.cat(outs, dim=-1)
        item_graph = self.item_graph_id_features(
            self.entity_emb.weight[: self.n_items] * item_keep, batch
        )
        if item_graph is not None:
            item_out = out[: self.n_items] + self.mc.mm_graph_beta * item_graph
            out = torch.cat([item_out, out[self.n_items:]], dim=0)
        return out

    def get_embeddings(self, batch):
        h = self.propagate(batch)
        item_e = h[: self.n_items]
        user_e = h[self.n_entities:]
        if self.mc.use_market_gate:
            user_e = self.user_gate(user_e, batch["user_country"])
        return user_e, item_e

    # ------------------------------------------------------------------ #
    # 打分
    # ------------------------------------------------------------------ #
    # 市场门控作用在「目标市场」维度：商品表示按用户所在国调制（同一商品面向
    # 不同市场呈现不同），而不是按商品原产国叠加静态国别嵌入——后者会在打分中
    # 形成 用户国别×商品原产国 的匹配偏置，可能把本国商品硬推到前面并压制
    # 跨境推荐；必须作为独立消融验证，不能当作既定因果结论。

    def _reliability_features(self, batch, item_ids, modalities, countries):
        """构造 [可用, genuine, local, fallback, observed, completion]。"""
        view_mask = modalities["view_mask"]
        denom = view_mask.sum(dim=1, keepdim=True).clamp_min(1)
        fallback = self._batch_value(
            batch, ("item_text_is_fallback", "item_text_fallback"), item_ids
        )
        if fallback is None:
            fallback = torch.zeros_like(view_mask)
        else:
            fallback = fallback.bool()
        genuine = self._batch_value(
            batch, ("item_text_genuine", "item_text_source_valid"), item_ids
        )
        if genuine is None:
            genuine = ~fallback
        else:
            genuine = genuine.bool() & ~fallback

        fallback_ratio = ((view_mask & fallback).sum(dim=1, keepdim=True)
                          / denom).to(modalities["text"].dtype)
        genuine_ratio = ((view_mask & genuine).sum(dim=1, keepdim=True)
                         / denom).to(modalities["text"].dtype)

        local = self._batch_value(
            batch, ("item_text_is_local",), item_ids
        )
        view_market = self._batch_value(
            batch, ("item_text_market", "item_text_country"), item_ids
        )
        if local is not None:
            local_mask = local.bool()
        elif view_market is not None:
            local_mask = view_market == countries.unsqueeze(-1)
        else:
            # 旧数据没有市场级来源；此时不给模型额外的不可验证信号。
            local_mask = torch.zeros_like(genuine)
        local_ratio = ((view_mask & genuine & local_mask).sum(dim=1, keepdim=True)
                       / denom).to(modalities["text"].dtype)

        completion = modalities["completion_confidence"]
        return torch.cat([
            modalities["text_available"], genuine_ratio, local_ratio,
            fallback_ratio, modalities["image_observed"], completion,
        ], dim=-1)

    @staticmethod
    def _masked_modality_weights(logits, availability):
        # 显式归一化避免两种模态都缺失时 softmax(-inf, -inf) 产生 NaN。
        shifted = logits - logits.max(dim=-1, keepdim=True).values
        weight = shifted.exp() * availability
        return weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def _residual_scale(self):
        return F.softplus(self.residual_beta_raw)

    def _residual_score(self, batch, user_e, users, items):
        if self.residual_mode == "none":
            return torch.zeros(len(users), device=user_e.device, dtype=user_e.dtype)
        modalities = self._batch_modalities(batch, items)
        user = user_e[users]
        if self.residual_mode == "fused":
            query = F.normalize(self.residual_fused_query(user), dim=-1)
            content = F.normalize(
                self.residual_fused_item(modalities["fused"]), dim=-1
            )
            available = torch.maximum(
                modalities["text_available"], modalities["image_available"]
            ).squeeze(-1)
            return self._residual_scale() * (query * content).sum(-1) * available

        query = self.residual_query(user).reshape(len(users), 2, -1)
        query = F.normalize(query, dim=-1)
        content = torch.stack([
            F.normalize(modalities["text"], dim=-1),
            F.normalize(modalities["image"], dim=-1),
        ], dim=1)
        modality_scores = (query * content).sum(dim=-1)
        availability = torch.cat([
            modalities["text_available"], modalities["image_available"]
        ], dim=-1)
        if self.residual_mode == "decoupled":
            weights = availability / availability.sum(-1, keepdim=True).clamp_min(1.0)
        else:
            countries = batch["user_country"][users]
            reliability = self._reliability_features(
                batch, items, modalities, countries
            )
            logits = (self.residual_user_weight(user)
                      + self.residual_market_weight(countries)
                      + self.residual_reliability_weight(reliability))
            weights = self._masked_modality_weights(logits, availability)
        return self._residual_scale() * (weights * modality_scores).sum(dim=-1)

    def _full_residual_score(self, batch, user_e, users):
        if self.residual_mode == "none":
            return torch.zeros(
                len(users), self.n_items, device=user_e.device, dtype=user_e.dtype
            )
        modalities = self._batch_modalities(batch)
        user = user_e[users]
        if self.residual_mode == "fused":
            query = F.normalize(self.residual_fused_query(user), dim=-1)
            content = F.normalize(
                self.residual_fused_item(modalities["fused"]), dim=-1
            )
            available = torch.maximum(
                modalities["text_available"], modalities["image_available"]
            ).squeeze(-1)
            return (self._residual_scale() * (query @ content.t())
                    * available.unsqueeze(0))

        query = self.residual_query(user).reshape(len(users), 2, -1)
        query = F.normalize(query, dim=-1)
        text_score = query[:, 0] @ F.normalize(modalities["text"], dim=-1).t()
        image_score = query[:, 1] @ F.normalize(modalities["image"], dim=-1).t()
        modality_scores = torch.stack([text_score, image_score], dim=-1)
        item_availability = torch.cat([
            modalities["text_available"], modalities["image_available"]
        ], dim=-1)
        availability = item_availability.unsqueeze(0).expand(len(users), -1, -1)
        if self.residual_mode == "decoupled":
            weights = availability / availability.sum(-1, keepdim=True).clamp_min(1.0)
        else:
            countries = batch["user_country"][users]
            logits = torch.empty_like(modality_scores)
            # 可靠性中的 local 是目标市场相关量，按市场分组避免 B x I x V 展开。
            for country in countries.unique():
                row = countries == country
                item_ids = torch.arange(self.n_items, device=user.device)
                country_ids = torch.full(
                    (self.n_items,), int(country), device=user.device, dtype=torch.long
                )
                reliability = self._reliability_features(
                    batch, item_ids, modalities, country_ids
                )
                item_logits = self.residual_reliability_weight(reliability)
                logits[row] = (self.residual_user_weight(user[row]).unsqueeze(1)
                               + self.residual_market_weight(country).view(1, 1, 2)
                               + item_logits.unsqueeze(0))
            weights = self._masked_modality_weights(logits, availability)
        return self._residual_scale() * (weights * modality_scores).sum(dim=-1)

    def precompute_item_tables(self, batch, item_e):
        """预计算一次评估所需的全目录商品表。

        返回值可以跨多个 user batch 传给 :meth:`full_score_cached`。调用方应在
        ``model.eval()`` 与 ``torch.no_grad()`` 下构造，训练模式切换会清空内部缓存。
        """
        tables = {"item_e": item_e, "market_items": {}, "residual": None}
        if self.mc.use_market_gate:
            for country in range(self.n_countries):
                cc = torch.full(
                    (self.n_items,), country, device=item_e.device, dtype=torch.long
                )
                gated = self.item_gate(item_e, cc)
                tables["market_items"][country] = gated
                if not self.training and not torch.is_grad_enabled():
                    self._eval_market_item_cache[(item_e.data_ptr(), country)] = gated

        if self.residual_mode != "none":
            modalities = self._batch_modalities(batch)
            residual = {
                "availability": torch.cat([
                    modalities["text_available"], modalities["image_available"]
                ], dim=-1),
                "overall_available": modalities["overall_available"].squeeze(-1),
            }
            if self.residual_mode == "fused":
                residual["fused"] = F.normalize(
                    self.residual_fused_item(modalities["fused"]), dim=-1
                )
            else:
                residual["text"] = F.normalize(modalities["text"], dim=-1)
                residual["image"] = F.normalize(modalities["image"], dim=-1)
                if self.residual_mode == "market_reliable":
                    item_ids = torch.arange(self.n_items, device=item_e.device)
                    reliability_logits = []
                    for country in range(self.n_countries):
                        countries = torch.full(
                            (self.n_items,), country, device=item_e.device,
                            dtype=torch.long,
                        )
                        reliability = self._reliability_features(
                            batch, item_ids, modalities, countries
                        )
                        reliability_logits.append(
                            self.residual_reliability_weight(reliability)
                            + self.residual_market_weight.weight[country]
                        )
                    residual["market_reliability_logits"] = torch.stack(
                        reliability_logits, dim=0
                    )
            tables["residual"] = residual
        return tables

    def full_score_cached(self, batch, user_e, item_e, users=None, tables=None):
        """使用 :meth:`precompute_item_tables` 的结果完成全目录评分。"""
        if tables is None:
            # 兼容早期内部接口 full_score_cached(batch, user_e, users, tables)。
            tables = users
            users = item_e
        item_e = tables["item_e"]
        if not self.mc.use_market_gate:
            out = user_e[users] @ item_e.t()
        else:
            countries = batch["user_country"][users]
            out = torch.empty(
                len(users), self.n_items, device=item_e.device, dtype=item_e.dtype
            )
            for country in countries.unique():
                row = countries == country
                out[row] = (user_e[users[row]]
                            @ tables["market_items"][int(country)].t())

        residual = tables.get("residual")
        if residual is None:
            return out
        user = user_e[users]
        if self.residual_mode == "fused":
            query = F.normalize(self.residual_fused_query(user), dim=-1)
            extra = query @ residual["fused"].t()
            extra = extra * residual["overall_available"].unsqueeze(0)
            return out + self._residual_scale() * extra

        query = F.normalize(
            self.residual_query(user).reshape(len(users), 2, -1), dim=-1
        )
        modality_scores = torch.stack([
            query[:, 0] @ residual["text"].t(),
            query[:, 1] @ residual["image"].t(),
        ], dim=-1)
        availability = residual["availability"].unsqueeze(0).expand(
            len(users), -1, -1
        )
        if self.residual_mode == "decoupled":
            weights = availability / availability.sum(-1, keepdim=True).clamp_min(1.0)
        else:
            countries = batch["user_country"][users]
            item_logits = residual["market_reliability_logits"][countries]
            logits = self.residual_user_weight(user).unsqueeze(1) + item_logits
            weights = self._masked_modality_weights(logits, availability)
        extra = (weights * modality_scores).sum(dim=-1)
        return out + self._residual_scale() * extra

    def score(self, batch, user_e, item_e, users, items):
        ie = item_e[items]
        if self.mc.use_market_gate:
            ie = self.item_gate(ie, batch["user_country"][users])
        base = (user_e[users] * ie).sum(-1)
        return base + self._residual_score(batch, user_e, users, items)

    def full_score(self, batch, user_e, item_e, users):
        """users: (B,) -> (B, n_items)。按目标市场逐国分组，共 n_countries 次门控。"""
        if not self.training and not torch.is_grad_enabled():
            cache_key = (id(batch), item_e.data_ptr())
            if (self._eval_item_tables is None
                    or self._eval_item_tables[0] != cache_key):
                self._eval_item_tables = (
                    cache_key, self.precompute_item_tables(batch, item_e)
                )
            return self.full_score_cached(
                batch, user_e, item_e, users, self._eval_item_tables[1]
            )
        if not self.mc.use_market_gate:
            out = user_e[users] @ item_e.t()
        else:
            tgt = batch["user_country"][users]
            out = torch.empty(len(users), self.n_items,
                              device=item_e.device, dtype=item_e.dtype)
            for c in tgt.unique():
                m = tgt == c
                cache_key = (item_e.data_ptr(), int(c))
                gated_items = self._eval_market_item_cache.get(cache_key)
                if gated_items is None:
                    cc = torch.full((self.n_items,), int(c), device=item_e.device,
                                    dtype=torch.long)
                    gated_items = self.item_gate(item_e, cc)
                    if not self.training and not torch.is_grad_enabled():
                        self._eval_market_item_cache[cache_key] = gated_items
                out[m] = user_e[users[m]] @ gated_items.t()
        return out + self._full_residual_score(batch, user_e, users)

    # ------------------------------------------------------------------ #
    # 各项损失
    # ------------------------------------------------------------------ #
    def cf_loss(self, batch, users, pos, neg):
        user_e, item_e = self.get_embeddings(batch)
        pos_s = self.score(batch, user_e, item_e, users, pos)
        neg_s = self.score(batch, user_e, item_e, users, neg)
        loss = bpr_loss(pos_s, neg_s)
        # BPR 正则作用在 sampled ego embeddings。最终表示的每个 block 已做
        # L2 normalize，在那里正则几乎是常数，无法约束真正的 ID 参数。
        reg = self.mc.l2_reg * (
            self.user_emb(users).pow(2).sum()
            + self.entity_emb(pos).pow(2).sum()
            + self.entity_emb(neg).pow(2).sum()
        ) / len(users)
        return loss + reg

    def kg_loss(self, h, r, t, t_neg):
        """纯 KG 三元组的分关系 TransR 损失；采样器应排除交互关系。

        按关系分组避免构造 (batch, embed_dim, relation_dim) 投影矩阵；同时在
        模型边界防御性过滤 interact 及其反向关系，避免旧 sampler 污染 KG 目标。
        """
        emb = torch.cat([self.entity_emb.weight, self.user_emb.weight], dim=0)
        kg_only = (r != 0) & (r != self.n_relations + 1)
        if not kg_only.any():
            return (emb.sum() + self.rel_att.W_r.sum()
                    + self.rel_att.rel_emb.weight.sum()) * 0.0
        h, r, t, t_neg = h[kg_only], r[kg_only], t[kg_only], t_neg[kg_only]
        r = self.compact_kg_relations(r)
        pos_d = torch.empty(len(r), device=emb.device, dtype=emb.dtype)
        neg_d = torch.empty_like(pos_d)
        projected_reg = torch.zeros((), device=emb.device, dtype=emb.dtype)
        for relation in r.unique():
            mask = r == relation
            projection = self.rel_att.W_r[relation]
            relation_emb = self.rel_att.rel_emb(relation)
            hp = emb[h[mask]] @ projection
            tp = emb[t[mask]] @ projection
            tn = emb[t_neg[mask]] @ projection
            pos_d[mask] = (hp + relation_emb - tp).pow(2).sum(-1)
            neg_d[mask] = (hp + relation_emb - tn).pow(2).sum(-1)
            fraction = mask.to(emb.dtype).mean()
            projected_reg = projected_reg + fraction * (
                hp.pow(2).mean() + tp.pow(2).mean() + relation_emb.pow(2).mean()
            )
        loss = F.softplus(pos_d - neg_d).mean()
        return loss + self.mc.l2_reg * projected_reg

    def alignment_loss(self, batch, item_ids, kg_pos=None):
        """真实、异语言、非重复文本视图之间的对称 text-only InfoNCE。

        共享图像永不进入此损失。品牌/品类邻居仍是 KG 边，但不再被视作语义等价
        商品。``kg_pos`` 仅为旧训练循环保留，刻意不消费。
        """
        if self.mc.align_source != "parallel":
            raise ValueError(
                "alignment_loss 只接受 align_source='parallel'"
            )
        item_ids = item_ids.unique()
        text = batch["item_text"][item_ids]
        metadata = self._text_metadata(batch, item_ids)
        valid = self._deduplicated_view_mask(
            text, metadata["text_valid"], metadata["content_hash"],
            metadata["text_dedup"],
        )
        fallback = metadata["text_fallback"]
        if fallback is not None:
            valid &= ~fallback.bool()
        genuine = self._batch_value(
            batch, ("item_text_genuine", "item_text_source_valid"), item_ids
        )
        if genuine is not None:
            valid &= genuine.bool()
        confidence = self._batch_value(
            batch, ("item_text_language_confidence", "item_text_lang_confidence"),
            item_ids,
        )
        threshold = float(getattr(self.mc, "alignment_language_confidence", 0.8))
        if confidence is not None:
            valid &= confidence >= threshold
        languages = batch["item_lang"][item_ids]
        hashes = metadata["content_hash"]
        pair_valid = self._batch_value(
            batch, ("item_text_pair_valid",), item_ids
        )

        left, right = [], []
        chosen = torch.zeros(len(item_ids), dtype=torch.bool, device=text.device)
        for a in range(text.size(1)):
            for b in range(a + 1, text.size(1)):
                if pair_valid is not None:
                    # schema v2 数据契约已同时校验 role、hash、语言与置信度。
                    pair = pair_valid[:, a, b].bool()
                else:
                    pair = (valid[:, a] & valid[:, b]
                            & (languages[:, a] != languages[:, b]))
                    if hashes is not None:
                        known = (hashes[:, a] >= 0) & (hashes[:, b] >= 0)
                        distinct = (hashes[:, a] != hashes[:, b])
                        exact_distinct = (text[:, a] != text[:, b]).any(dim=-1)
                        pair &= torch.where(known, distinct, exact_distinct)
                    else:
                        pair &= (text[:, a] != text[:, b]).any(dim=-1)
                pair &= ~chosen
                if pair.any():
                    left.append(text[pair, a])
                    right.append(text[pair, b])
                    chosen |= pair
        if not left or sum(len(x) for x in left) < 2:
            return torch.zeros((), device=text.device)
        z1 = self.text_enc(torch.cat(left, dim=0))
        z2 = self.text_enc(torch.cat(right, dim=0))
        return info_nce(z1, z2, self.mc.temperature)

    def collaborative_content_loss(self, batch, item_ids):
        """CLCRec-inspired 单项对齐：warm 内容对齐 detached raw ID。"""
        item_ids = item_ids.unique()
        cold = batch.get("cold_items")
        if cold is not None and cold.numel() and item_ids.numel():
            item_ids = item_ids[~torch.isin(item_ids, cold)]
        if item_ids.numel() < 2:
            return torch.zeros((), device=batch["item_text"].device)

        content = self._batch_modalities(batch, item_ids)["fused"]
        collaborative = self.entity_emb(item_ids).detach()
        return info_nce(content, collaborative, self.mc.collab_temperature)

    def modality_completion_loss(self, batch, item_ids):
        """人工遮蔽已观测图像，联合训练归一化补全及其质量置信度。"""
        modalities = self._batch_modalities(batch, item_ids)
        mk = ((batch["item_image_mask"][item_ids] > 0.5)
              & modalities["text_available"].squeeze(-1).bool())
        if int(mk.sum()) == 0:
            return torch.zeros((), device=batch["item_text"].device)
        pred = F.normalize(
            self.img_from_text(modalities["pooled_raw_text"][mk]),
            dim=-1, eps=1e-12,
        )
        target = F.normalize(
            batch["item_image"][item_ids][mk], dim=-1, eps=1e-12
        )
        cosine = F.cosine_similarity(pred, target, dim=-1)
        predicted_quality = self.completion_confidence_head(
            modalities["text"][mk]
        ).squeeze(-1)
        quality_target = ((cosine.detach() + 1.0) * 0.5).clamp(0.0, 1.0)
        confidence_loss = F.binary_cross_entropy(predicted_quality, quality_target)
        confidence_weight = float(getattr(
            self.mc, "completion_confidence_weight",
            getattr(self.mc, "lambda_completion_confidence", 0.1),
        ))
        return (1.0 - cosine).mean() + confidence_weight * confidence_loss

    def adversarial_loss(self, batch, item_ids, lambd=1.0):
        """只在有效且非 fallback 的文本视图上学习语言不变性。"""
        text = batch["item_text"][item_ids]
        metadata = self._text_metadata(batch, item_ids)
        valid = self._deduplicated_view_mask(
            text, metadata["text_valid"], metadata["content_hash"],
            metadata["text_dedup"],
        )
        if metadata["text_fallback"] is not None:
            valid &= ~metadata["text_fallback"].bool()
        if not valid.any():
            return torch.zeros((), device=text.device)
        z = self.text_enc(text[valid])
        logits = self.lang_disc(grad_reverse(z, lambd))
        labels = batch["item_lang"][item_ids][valid]
        return F.cross_entropy(logits, labels)
