# -*- coding: utf-8 -*-
"""Focused regressions for schema-v2 model innovations and fair baselines."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from baselines import build_model
from config import Config
from data_utils import build_dataset
from model import ACMR
from train import make_batch


def tiny_setup(*, residual="none", use_multimodal=True, use_market_gate=False):
    cfg = Config()
    cfg.data.n_users = 24
    cfg.data.n_items = 30
    cfg.data.n_entities = 80
    cfg.data.n_interactions = 600
    cfg.data.n_triples = 240
    cfg.data.text_dim = 16
    cfg.data.image_dim = 12
    cfg.data.val_ratio = 0.2
    cfg.data.test_ratio = 0.2
    cfg.data.cold_item_ratio = 0.1
    cfg.data.cold_val_item_ratio = 0.5
    cfg.data.split_seed = 17
    cfg.model.embed_dim = 8
    cfg.model.relation_dim = 6
    cfg.model.n_gnn_layers = 1
    cfg.model.layer_dropout = 0.0
    cfg.model.message_dropout = 0.0
    cfg.model.cold_id_dropout = 0.0
    cfg.model.use_mm_item_graph = False
    cfg.model.use_multimodal = use_multimodal
    cfg.model.use_market_gate = use_market_gate
    cfg.model.residual = residual
    dataset = build_dataset(cfg)
    batch = make_batch(dataset, torch.device("cpu"))
    return cfg, dataset, batch


def test_pair_valid_drives_text_only_alignment():
    cfg, dataset, batch = tiny_setup()
    model = ACMR(dataset, cfg).eval()
    item_ids = torch.arange(8)

    model.zero_grad(set_to_none=True)
    loss = model.alignment_loss(batch, item_ids)
    assert torch.isfinite(loss) and float(loss.detach()) > 0.0
    loss.backward()
    text_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.text_enc.parameters()
        if parameter.grad is not None
    )
    image_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.image_enc.parameters()
        if parameter.grad is not None
    )
    assert text_grad > 0.0
    assert image_grad == 0.0

    no_pairs = dict(batch)
    no_pairs["item_text_pair_valid"] = torch.zeros_like(
        batch["item_text_pair_valid"]
    )
    assert float(model.alignment_loss(no_pairs, item_ids)) == 0.0
    fake_kg_pairs = torch.roll(item_ids, 1)
    assert torch.allclose(
        model.alignment_loss(batch, item_ids),
        model.alignment_loss(batch, item_ids, fake_kg_pairs),
    )


def test_deduplicated_pooling_and_hard_missing_masks():
    cfg, dataset, batch = tiny_setup()
    model = ACMR(dataset, cfg).eval()
    item_ids = torch.arange(4)
    text = batch["item_text"][item_ids].clone()
    changed_duplicate = text.clone()
    changed_duplicate[:, 1] += 1000.0
    metadata = {
        name: value.clone() if value is not None else None
        for name, value in model._text_metadata(batch, item_ids).items()
    }
    metadata["text_dedup"][:, 1] = False
    image = batch["item_image"][item_ids]
    image_mask = batch["item_image_mask"][item_ids]

    first = model._encode_modalities(text, image, image_mask, **metadata)
    second = model._encode_modalities(
        changed_duplicate, image, image_mask, **metadata
    )
    assert torch.allclose(first["text"], second["text"], atol=1e-7)
    assert torch.allclose(first["fused"], second["fused"], atol=1e-7)

    n_items, n_views = 3, text.size(1)
    missing = model._encode_modalities(
        torch.randn(n_items, n_views, cfg.data.text_dim),
        torch.randn(n_items, cfg.data.image_dim),
        torch.zeros(n_items),
        text_valid=torch.zeros(n_items, n_views, dtype=torch.bool),
        text_dedup=torch.zeros(n_items, n_views, dtype=torch.bool),
    )
    for key in ("text", "image", "fused", "completion_confidence"):
        assert torch.count_nonzero(missing[key]) == 0, key


def test_normalized_completion_and_confidence_loss():
    cfg, dataset, batch = tiny_setup()
    model = ACMR(dataset, cfg).eval()
    item_ids = torch.arange(dataset.n_items)
    modalities = model._batch_modalities(batch, item_ids)
    predicted = F.normalize(
        model.img_from_text(modalities["pooled_raw_text"]), dim=-1, eps=1e-12
    )
    assert torch.allclose(
        predicted.norm(dim=-1), torch.ones(dataset.n_items), atol=1e-5
    )

    model.zero_grad(set_to_none=True)
    loss = model.modality_completion_loss(batch, item_ids)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in model.img_from_text.parameters()
    )
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in model.completion_confidence_head.parameters()
    )


def test_residual_modes_match_point_and_cached_full_scores():
    torch.manual_seed(31)
    for mode in ("none", "fused", "decoupled", "market_reliable"):
        cfg, dataset, batch = tiny_setup(
            residual=mode, use_market_gate=True
        )
        model = ACMR(dataset, cfg).eval()
        out_dim = cfg.model.embed_dim * (cfg.model.n_gnn_layers + 1)
        user_e = torch.randn(dataset.n_users, out_dim)
        item_e = torch.randn(dataset.n_items, out_dim)
        users = torch.tensor([0, 1, 2, 3])
        items = torch.tensor([3, 5, 7, 9])

        with torch.no_grad():
            point = model.score(batch, user_e, item_e, users, items)
            tables = model.precompute_item_tables(batch, item_e)
            full = model.full_score_cached(
                batch, user_e, item_e, users, tables
            )
        assert torch.allclose(
            point, full[torch.arange(len(users)), items], atol=3e-6
        ), mode
        if mode != "none":
            assert torch.allclose(
                model._residual_scale().detach(), torch.tensor(0.01), atol=1e-7
            )


def test_zero_degree_id_mask_and_grouped_kg_loss():
    cfg, dataset, batch = tiny_setup(use_multimodal=False)
    model = ACMR(dataset, cfg).eval()
    degree_batch = dict(batch)
    degree_batch.pop("zero_train_items")
    degree_batch["cold_items"] = torch.empty(0, dtype=torch.long)
    degree_batch["item_train_degree"] = torch.ones(dataset.n_items, dtype=torch.long)
    degree_batch["item_train_degree"][0] = 0
    before = model.node_features(degree_batch)[0].detach().clone()
    with torch.no_grad():
        saved = model.entity_emb.weight[0].clone()
        model.entity_emb.weight[0].add_(100.0)
        after = model.node_features(degree_batch)[0].detach().clone()
        model.entity_emb.weight[0].copy_(saved)
    assert torch.allclose(before, after, atol=1e-7)

    h = torch.tensor([2, 3, 4, 5])
    r = torch.tensor([1, 2, 1, dataset.n_relations + 2])
    t = torch.tensor([20, 21, 22, 23])
    negative = torch.tensor([21, 22, 23, 24])
    assert model.rel_att.W_r.shape[0] == 2 * dataset.n_relations
    assert model.rel_att.W_r.shape[0] == dataset.n_relations_total - 2
    compact_r = model.compact_kg_relations(r)
    emb = torch.cat([model.entity_emb.weight, model.user_emb.weight], dim=0)
    projection = model.rel_att.W_r[compact_r]
    hp = torch.bmm(emb[h].unsqueeze(1), projection).squeeze(1)
    tp = torch.bmm(emb[t].unsqueeze(1), projection).squeeze(1)
    tn = torch.bmm(emb[negative].unsqueeze(1), projection).squeeze(1)
    relation = model.rel_att.rel_emb(compact_r)
    pos_distance = (hp + relation - tp).pow(2).sum(-1)
    neg_distance = (hp + relation - tn).pow(2).sum(-1)
    reference = F.softplus(pos_distance - neg_distance).mean()
    reference = reference + cfg.model.l2_reg * (
        hp.pow(2).mean() + tp.pow(2).mean() + relation.pow(2).mean()
    )
    assert torch.allclose(model.kg_loss(h, r, t, negative), reference, atol=1e-6)

    mixed_h = torch.cat([torch.tensor([dataset.n_entities, 0]), h])
    mixed_r = torch.cat([torch.tensor([0, dataset.n_relations + 1]), r])
    mixed_t = torch.cat([torch.tensor([0, dataset.n_entities]), t])
    mixed_negative = torch.cat([
        torch.tensor([1, dataset.n_entities + 1]), negative
    ])
    assert torch.allclose(
        model.kg_loss(mixed_h, mixed_r, mixed_t, mixed_negative), reference,
        atol=1e-6,
    )
    interaction_only = model.kg_loss(
        mixed_h[:2], mixed_r[:2], mixed_t[:2], mixed_negative[:2]
    )
    assert float(interaction_only.detach()) == 0.0
    model.zero_grad(set_to_none=True)
    interaction_only.backward()
    assert torch.count_nonzero(model.rel_att.W_r.grad) == 0


def test_independent_baseline_interfaces_and_backward():
    cfg, dataset, batch = tiny_setup()
    warm = torch.nonzero(batch["item_train_degree"] > 0).flatten()
    assert len(warm) >= 6
    users = torch.tensor([0, 1, 2])
    positive, negative = warm[:3], warm[3:6]

    for name in ("bpr_mf", "lightgcn", "vbpr"):
        model = build_model(name, dataset, cfg)
        model.refresh_attention(batch)
        user_e, item_e = model.get_embeddings(batch)
        assert user_e.shape == (dataset.n_users, cfg.model.embed_dim)
        assert item_e.shape == (dataset.n_items, cfg.model.embed_dim)
        point = model.score(batch, user_e, item_e, users, positive)
        tables = model.precompute_item_tables(batch, item_e)
        full = model.full_score_cached(batch, user_e, item_e, users, tables)
        assert full.shape == (len(users), dataset.n_items)
        assert torch.allclose(
            point, full[torch.arange(len(users)), positive], atol=1e-6
        )

        model.zero_grad(set_to_none=True)
        loss = model.cf_loss(batch, users, positive, negative)
        assert torch.isfinite(loss)
        loss.backward()
        assert any(
            parameter.grad is not None and torch.count_nonzero(parameter.grad)
            for parameter in model.parameters()
        ), name


if __name__ == "__main__":
    torch.manual_seed(0)
    test_pair_valid_drives_text_only_alignment()
    test_deduplicated_pooling_and_hard_missing_masks()
    test_normalized_completion_and_confidence_loss()
    test_residual_modes_match_point_and_cached_full_scores()
    test_zero_degree_id_mask_and_grouped_kg_loss()
    test_independent_baseline_interfaces_and_backward()
    print("PASS: model innovations and independent baselines")
