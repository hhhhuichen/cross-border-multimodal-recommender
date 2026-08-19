# -*- coding: utf-8 -*-
"""
自检脚本：用极小规模数据跑通 前向 -> 各项损失 -> 反向 -> 评测。
本地装好 PyTorch 后先跑这个，全部 PASS 再跑 train.py。

    python test_smoke.py
"""
import json
from pathlib import Path
import sys
import tempfile
import warnings
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

# Windows 下控制台/管道常是 GBK 编码，中文输出会乱码，统一成 UTF-8
if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
    sys.stdout.reconfigure(encoding="utf-8")

from checkpoint_utils import load_checkpoint, save_checkpoint
from config import Config, validate_config
from data_contract import (
    IMAGE_AVAILABLE_FILE,
    SCHEMA_VERSION,
    TEXT_META_FILE,
    schema_descriptor,
    text_content_hash,
)
from data_utils import (
    build_dataset, CKGDataset, BPRSampler, KGSampler, build_frozen_mm_item_graph,
    sample_degree_sensitive_interactions,
)
from model import ACMR
from off_data import OFFRealData
from train import (
    make_batch, make_training_graph_batch, evaluate, print_report,
    select_validation_metrics,
)


def tiny_cfg():
    cfg = Config()
    cfg.data.n_users = 200
    cfg.data.n_items = 220
    cfg.data.n_entities = 440
    cfg.data.n_interactions = 3000
    cfg.data.n_triples = 2000
    cfg.data.cold_item_ratio = 0.25
    cfg.data.cold_val_item_ratio = 0.4
    cfg.model.n_gnn_layers = 2
    cfg.train.batch_size = 256
    cfg.train.kg_batch_size = 512
    return cfg


def run(name, cfg):
    dev = torch.device("cpu")
    ds = build_dataset(cfg)
    if not cfg.model.use_kg:
        expected = {0, ds.n_relations + 1}
        assert set(np.unique(ds.edge_type).tolist()) <= expected, (
            "build_dataset(use_kg=False) 仍保留 KG 边"
        )
    assert not np.intersect1d(ds.cold_val_items, ds.cold_test_items).size, (
        "冷验证与冷测试商品必须不相交"
    )
    batch = make_batch(ds, dev)
    train_batch = make_training_graph_batch(
        batch, ds, cfg, np.random.default_rng(cfg.data.seed + 11)
    )
    model = ACMR(ds, cfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    T = lambda x: torch.as_tensor(x, dtype=torch.long, device=dev)

    if name == "完整模型" and cfg.model.use_market_gate:
        state = model.state_dict()
        assert "user_gate.adj" not in state and "item_gate.adj" not in state
        # 模拟旧 checkpoint 中遗留的邻接键，严格加载仍应兼容。
        state["user_gate.adj"] = model.user_gate.adj.clone()
        state["item_gate.adj"] = model.item_gate.adj.clone()
        ACMR(ds, cfg).load_state_dict(state, strict=True)

    # 冷商品的随机 ID 在 cold_id_dropout=0 的消融中也必须永久屏蔽。
    if len(ds.cold_items):
        model.eval()
        with torch.no_grad():
            before = model.node_features(batch)[batch["cold_items"]].clone()
            saved = model.entity_emb.weight[batch["cold_items"]].clone()
            model.entity_emb.weight[batch["cold_items"]].add_(100.0)
            after = model.node_features(batch)[batch["cold_items"]]
            model.entity_emb.weight[batch["cold_items"]].copy_(saved)
        assert torch.allclose(before, after, atol=1e-6), (
            "冷商品表示仍受未训练的随机 ID 影响"
        )

    model.train()
    model.refresh_attention(train_batch)

    # --- 冻结多模态商品图 ---
    mm_edge = batch["mm_item_edge_index"]
    mm_weight = batch["mm_item_edge_weight"]
    if cfg.model.use_multimodal and cfg.model.use_mm_item_graph:
        assert mm_edge.shape[0] == 2 and mm_edge.shape[1] == len(mm_weight)
        assert mm_edge.shape[1] <= 4 * ds.n_items * cfg.model.mm_graph_k
        assert torch.isfinite(mm_weight).all() and (mm_weight > 0).all()
        directed = set(map(tuple, mm_edge.t().cpu().tolist()))
        assert all((b, a) in directed for a, b in directed), "item 图未对称化"

    # --- 边权重是否符合混合传播设计 ---
    att = model._att_cache
    assert att.shape[0] == train_batch["edge_index"].shape[1], "注意力长度应等于边数"
    assert torch.isfinite(att).all(), "注意力出现 nan/inf"
    assert (att >= 0).all(), "边权重出现负值"
    dst = train_batch["edge_index"][1]
    if cfg.model.use_kg:
        # KG 边：组内 softmax 后乘 γ，按目标节点求和应等于 γ
        is_int = ((train_batch["edge_type"] == 0)
                  | (train_batch["edge_type"] == ds.n_relations + 1))
        kg_dst = dst[~is_int]
        s = torch.zeros(ds.n_nodes).index_add_(0, kg_dst, att[~is_int])
        active = torch.bincount(kg_dst, minlength=ds.n_nodes) > 0
        gamma = cfg.model.kg_att_scale
        assert torch.allclose(s[active], torch.full((int(active.sum()),), gamma),
                              atol=1e-4), "KG 边注意力未按目标节点归一化到 γ"

    # --- 表示维度 ---
    ue, ie = model.get_embeddings(train_batch)
    exp = cfg.model.embed_dim * (cfg.model.n_gnn_layers + 1)
    assert ue.shape == (ds.n_users, exp), f"用户表示维度错误 {ue.shape}"
    assert ie.shape == (ds.n_items, exp), f"商品表示维度错误 {ie.shape}"

    # --- 各项损失可反传 ---
    u, p, n = next(iter(BPRSampler(
        ds, cfg.train.batch_size,
        market_alpha=cfg.model.market_sampling_alpha,
    )))
    for uu, nn in zip(u, n):
        assert int(nn) not in set(ds.train_user_pos.get(int(uu), [])), (
            "BPR 负采样命中了训练正交互"
        )
    loss = model.cf_loss(train_batch, T(u), T(p), T(n))
    if cfg.model.use_multimodal and cfg.model.use_cross_lingual_align:
        iu = T(p).unique()
        # schema-v2 对齐只读取同商品的可信异语文本视图。
        loss = loss + model.alignment_loss(train_batch, iu)
    if cfg.model.use_multimodal and cfg.model.use_collab_content_cl:
        ids = T(p).unique()
        singleton = model.collaborative_content_loss(train_batch, ids[:1])
        assert float(singleton) == 0.0, "单商品对比 batch 应安全跳过"
        opt.zero_grad(set_to_none=True)
        l_ccl_probe = model.collaborative_content_loss(train_batch, ids)
        assert torch.isfinite(l_ccl_probe), "协同-内容对比损失出现 nan/inf"
        l_ccl_probe.backward()
        content_grad = sum(
            float(q.grad.abs().sum()) for q in model.text_enc.parameters()
            if q.grad is not None
        )
        id_grad = model.entity_emb.weight.grad
        assert content_grad > 0, "协同-内容对比损失没有训练内容编码器"
        assert id_grad is None or float(id_grad.abs().sum()) == 0.0, (
            "协同 teacher 未 detach，辅助损失正在拖动 ID 空间"
        )
        opt.zero_grad(set_to_none=True)
        l_ccl = model.collaborative_content_loss(train_batch, ids)
        loss = loss + cfg.model.lambda_collab_cl * l_ccl
    if cfg.model.use_multimodal and cfg.model.use_lang_adversarial:
        loss = loss + model.adversarial_loss(train_batch, T(p).unique(), 1.0)
    if cfg.model.use_multimodal and cfg.model.use_modality_completion:
        l_mmc = model.modality_completion_loss(train_batch, T(p).unique())
        assert torch.isfinite(l_mmc), "补全损失为 nan"
        loss = loss + l_mmc
    assert torch.isfinite(loss), "CF 损失为 nan"
    opt.zero_grad(); loss.backward(); opt.step()

    grads = [n_ for n_, q in model.named_parameters()
             if q.requires_grad and q.grad is not None and q.grad.abs().sum() > 0]
    assert len(grads) > 0, "没有任何参数收到梯度"
    if not cfg.model.use_kg:
        assert not any(name.startswith("gnn_layers.") for name in grads), (
            "no-KG 分支仍经过带参数的 GNN 层，不是 LightGCN 传播"
        )

    if cfg.model.use_kg:
        h, r, t, tn = next(iter(KGSampler(ds, cfg.train.kg_batch_size)))
        interaction_relations = {0, ds.n_relations + 1}
        assert not np.isin(ds.ckg_triples[:, 1], list(interaction_relations)).any(), (
            "TransR 训练集仍含 interaction 正反关系"
        )
        assert not np.isin(r, list(interaction_relations)).any(), (
            "KGSampler 仍采样 interaction 正反关系"
        )
        assert min(h.min(), t.min(), tn.min()) >= 0
        assert max(h.max(), t.max(), tn.max()) < ds.n_entities, (
            "TransR 不应读取用户节点"
        )
        true_triples = {tuple(map(int, x)) for x in ds.ckg_triples}
        for x in zip(h, r, tn):
            assert tuple(map(int, x)) not in true_triples, "KG 负采样命中了真实三元组"
        lkg = model.kg_loss(T(h), T(r), T(t), T(tn))
        assert torch.isfinite(lkg), "KG 损失为 nan"
        opt.zero_grad(); lkg.backward(); opt.step()
        assert model.rel_att.W_r.grad is not None, "W_r 未收到 KG 阶段的梯度"

    # --- 评测跑得通且指标在 [0,1]（整体 / 跨境 / 冷启动三组）---
    model.refresh_attention(batch)  # 推理必须恢复完整训练图
    m = evaluate(model, batch, ds, [10, 20], ds.test_user_pos, max_users=100)
    assert all(0.0 <= v <= 1.0 for v in m.values()), f"指标越界 {m}"
    mc = evaluate(model, batch, ds, [10, 20], ds.test_user_pos_cross, max_users=100)
    cold10 = float("nan")
    if ds.test_user_pos_cold:
        cold_report = evaluate(
            model, batch, ds, [10, 20], ds.test_user_pos_cold,
            max_users=100, return_report=True,
        )
        assert all(0.0 <= v <= 1.0 for v in cold_report.metrics.values()), (
            f"冷启动指标越界 {cold_report.metrics}"
        )
        for k in (10, 20):
            coverage = cold_report.coverage["eligible"][str(k)]
            assert 0 <= coverage["users"] <= cold_report.coverage["total_users"]
            assert 0.0 <= coverage["user_fraction"] <= 1.0
        cold10 = cold_report.metrics["recall@10"]

    print(f"  PASS [{name}] 收到梯度参数 {len(grads)} 组 | "
          f"recall@10={m['recall@10']:.4f} 跨境={mc['recall@10']:.4f} "
          f"冷启动={cold10:.4f}")


class FixedScoreModel(torch.nn.Module):
    def __init__(self, scores):
        super().__init__()
        self.scores = torch.as_tensor(scores, dtype=torch.float)

    def get_embeddings(self, batch):
        return torch.empty(0), torch.empty(0)

    def full_score(self, batch, user_e, item_e, users):
        return self.scores[users.cpu()]


def test_evaluate_known_scores():
    ds = SimpleNamespace(
        n_items=5,
        train_user_pos={0: np.array([0]), 1: np.array([3])},
    )
    user_pos = {
        0: np.array([2]),
        1: np.array([1, 4]),
    }
    scores = [
        [9.0, 0.2, 0.9, 0.1, 0.0],
        [0.1, 0.5, 0.2, 9.0, 0.4],
    ]
    training_model = FixedScoreModel(scores)
    assert training_model.training
    out = evaluate(training_model, {}, ds, [1, 2], user_pos)
    assert training_model.training, "evaluate 未恢复调用前的 train 状态"
    assert np.isclose(out["recall@1"], 0.75), out
    assert np.isclose(out["recall@2"], 1.0), out
    assert np.isclose(out["hit@1"], 1.0), out
    assert np.isclose(out["hit@2"], 1.0), out

    empty = evaluate(
        training_model, {}, ds, [1, 2], {}, return_report=True
    )
    print_report("空正例回归", empty)
    assert empty.coverage["sampled_users"] == 0
    for k in ("1", "2"):
        eligibility = empty.coverage["eligible"][k]
        assert eligibility == {
            "markets": [], "excluded_markets": [],
            "users": 0, "excluded_users": 0,
            "positives": 0, "excluded_positives": 0,
            "user_fraction": 0.0, "positive_fraction": 0.0,
            "minimum_candidates": 0,
        }

    limited_model = FixedScoreModel(scores)
    limited_model.eval()
    limited_truth = {0: np.array([2]), 1: np.array([1])}
    limited = evaluate(
        limited_model, {}, ds, [3], limited_truth,
        candidate_items=np.array([1, 2]), return_report=True,
    )
    assert limited.coverage["eligible"]["3"]["markets"] == []
    assert limited.metrics["recall@3"] == 0.0
    assert not limited_model.training, "evaluate 未恢复调用前的 eval 状态"

    market_ds = SimpleNamespace(
        n_items=5,
        train_user_pos={},
        user_country=np.array([0]),
        item_market_mask=np.array([[True, True, False, False, False]]),
    )
    market_scores = [[0.0, 1.0, 100.0, 99.0, 98.0]]
    market_out = evaluate(
        FixedScoreModel(market_scores), {}, market_ds, [1],
        {0: np.array([1])},
    )
    assert market_out["recall@1"] == 1.0, "不可售商品进入了 OFF 排名候选集"

    market_ds.train_user_pos = {0: np.array([0])}
    ineligible = evaluate(
        FixedScoreModel(market_scores), {}, market_ds, [2],
        {0: np.array([1])}, return_report=True,
    )
    coverage = ineligible.coverage["eligible"]["2"]
    assert coverage["markets"] == [] and coverage["excluded_markets"] == ["0"]
    assert coverage["users"] == 0, "候选不足 K 的市场仍进入了聚合"

    try:
        select_validation_metrics(
            "cross", {"recall@10": 0.1}, {"recall@10": 0.0}, None,
            overall_available=True, cross_available=False,
        )
    except ValueError as exc:
        assert "跨境验证集为空" in str(exc)
    else:
        raise AssertionError("cross 验证为空时静默降级为 overall")

    try:
        select_validation_metrics(
            "overall", {"recall@10": 0.0}, {"recall@10": 0.0}, None,
            overall_available=False, cross_available=False,
        )
    except ValueError as exc:
        if "整体验证集为空" not in str(exc):
            raise
    else:
        raise AssertionError("overall 验证为空时仍选择了零指标 checkpoint")


def test_frozen_mm_graph_contract():
    text = np.array([
        [[1.0, 0.0], [1.0, 0.0]],
        [[0.9, 0.1], [0.9, 0.1]],
        [[0.0, 1.0], [0.0, 1.0]],
        [[0.1, 0.9], [0.1, 0.9]],
    ], dtype=np.float32)
    image = np.array([
        [1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]
    ], dtype=np.float32)
    image_mask = np.array([1, 1, 0, 1], dtype=np.float32)

    edge, weight = build_frozen_mm_item_graph(
        text, image, image_mask, k=1, text_weight=0.0, chunk_size=2
    )
    assert edge.shape[0] == 2 and edge.shape[1] == len(weight)
    assert 2 not in edge, "缺图商品不应参与 image-only kNN"
    assert np.isfinite(weight).all() and (weight > 0).all()
    pairs = set(map(tuple, edge.T.tolist()))
    assert all((b, a) in pairs for a, b in pairs)


def test_graph_does_not_depend_on_split():
    a = tiny_cfg()
    b = tiny_cfg()
    a.model.use_mm_item_graph = True
    b.model.use_mm_item_graph = True
    b.data.val_ratio = 0.2
    b.data.test_ratio = 0.1
    da, db = build_dataset(a), build_dataset(b)
    assert np.array_equal(da.mm_item_edge_index, db.mm_item_edge_index)
    assert np.allclose(da.mm_item_edge_weight, db.mm_item_edge_weight)


def test_mm_graph_preserves_isolated_items():
    cfg = tiny_cfg()
    cfg.model.use_mm_item_graph = True
    ds = build_dataset(cfg)
    batch = make_batch(ds, torch.device("cpu"))
    batch["mm_item_edge_index"] = torch.tensor([[0, 1], [1, 0]])
    batch["mm_item_edge_weight"] = torch.ones(2)
    model = ACMR(ds, cfg)
    item_id = torch.randn(ds.n_items, cfg.model.embed_dim)
    side = model.item_graph_id_features(item_id, batch)
    assert side is not None
    first_graph_block = side[:, cfg.model.embed_dim:2 * cfg.model.embed_dim]
    assert torch.allclose(
        first_graph_block[0], F.normalize(item_id[1], dim=-1)
    )
    assert torch.allclose(
        first_graph_block[1], F.normalize(item_id[0], dim=-1)
    )
    assert torch.count_nonzero(first_graph_block[2:]) == 0, (
        "冻结 item 图不应给无邻居商品增加 side-branch 表示"
    )


def test_training_samplers_ignore_heldout_labels():
    cfg = tiny_cfg()
    ds = build_dataset(cfg)
    train_items = set(ds.train_pairs[:, 1].tolist())
    heldout_items = set(ds.val_pairs[:, 1].tolist()) | set(ds.test_pairs[:, 1].tolist())
    assert heldout_items <= train_items, "warm held-out 商品没有训练交互"
    bpr = BPRSampler(ds, cfg.train.batch_size, seed=7)
    kg = KGSampler(ds, cfg.train.kg_batch_size, seed=7)
    interaction_relations = {0, ds.n_relations + 1}
    assert not np.isin(
        ds.ckg_triples[:, 1], list(interaction_relations)
    ).any(), "TransR 数据不得包含 interaction 关系"
    assert not interaction_relations.intersection(
        relation for _, relation in kg.forbidden_tails
    ), "KGSampler 的禁止集仍包含 interaction 关系"

    heldout = np.concatenate([ds.val_pairs, ds.test_pairs], axis=0)
    assert len(heldout) > 0
    checked = False
    for u, i in heldout:
        u, i = int(u), int(i)
        if i in set(ds.train_user_pos.get(u, [])):
            continue
        assert i not in bpr.pos_set.get(u, set())
        checked = True
        break
    assert checked, "未找到可用于 held-out 泄漏回归测试的交互"


def test_bpr_sampler_excludes_users_without_market_negatives():
    dataset = SimpleNamespace(
        n_items=3,
        train_pairs=np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64),
        train_user_pos={
            0: np.array([0, 1], dtype=np.int64),
            1: np.array([0], dtype=np.int64),
        },
        user_country=np.array([0, 0], dtype=np.int64),
        item_market_mask=np.array([[True, True, True]]),
    )
    sampler = BPRSampler(dataset, batch_size=8, seed=13)
    assert sampler.excluded_users.tolist() == [0]
    assert sampler.excluded_pair_count == 2
    assert sampler.pairs.tolist() == [[1, 0]]
    users, positives, negatives = next(iter(sampler))
    assert users.tolist() == [1]
    assert positives.tolist() == [0]
    assert negatives.tolist() == [1]


def test_degree_sensitive_pruning_contract():
    cfg = tiny_cfg()
    ds = build_dataset(cfg)
    ratio = 0.6
    keep = sample_degree_sensitive_interactions(
        ds.train_pairs, ratio, np.random.default_rng(123)
    )
    assert len(keep) == int(np.ceil(len(ds.train_pairs) * (1 - ratio)))
    assert len(np.unique(keep)) == len(keep)

    cfg.model.use_degree_sensitive_pruning = True
    cfg.model.interaction_prune_ratio = ratio
    full = make_batch(ds, torch.device("cpu"))
    train = make_training_graph_batch(
        full, ds, cfg, np.random.default_rng(123)
    )
    n = len(keep)
    forward = train["edge_index"][:, :n]
    reverse = train["edge_index"][:, n:2 * n]
    assert torch.equal(forward.flip(0), reverse), "交互边没有成对保留/删除"
    assert full["edge_index"].shape[1] == ds.edge_index.shape[1]


def test_market_smoothed_sampler():
    cfg = tiny_cfg()
    ds = build_dataset(cfg)
    sampler = BPRSampler(ds, cfg.train.batch_size, seed=9, market_alpha=0.0)
    idx = sampler._epoch_indices()
    markets = ds.user_country[ds.train_pairs[idx, 0]]
    counts = np.bincount(markets)
    active = counts[counts > 0]
    assert active.max() - active.min() <= 1, "alpha=0 未实现等市场采样"


def test_checkpoint_semantic_fingerprint():
    cfg = tiny_cfg()
    ds = build_dataset(cfg)
    model = ACMR(ds, cfg)
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/fingerprinted.pt"
        save_checkpoint(path, model, ds, cfg, extra={"epoch": 1})
        loaded = ACMR(ds, cfg)
        loaded._att_cache = torch.ones(1)
        metadata = load_checkpoint(path, loaded, ds, cfg)
        assert metadata["extra"]["epoch"] == 1
        assert loaded._att_cache is None, "加载 checkpoint 后未清空注意力缓存"

        changed_cfg = tiny_cfg()
        changed_cfg.data.seed += 1
        changed_ds = build_dataset(changed_cfg)
        try:
            load_checkpoint(
                path, ACMR(changed_ds, changed_cfg), changed_ds, changed_cfg
            )
        except ValueError as exc:
            assert "不一致" in str(exc)
        else:
            raise AssertionError("checkpoint 未拒绝同形状但语义不同的数据")

        legacy_path = f"{tmp}/legacy.pt"
        torch.save(model.state_dict(), legacy_path)
        try:
            load_checkpoint(legacy_path, ACMR(ds, cfg), ds, cfg)
        except ValueError as exc:
            assert "allow_legacy=True" in str(exc)
        else:
            raise AssertionError("旧 checkpoint 未经显式授权就被加载")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            assert load_checkpoint(
                legacy_path, ACMR(ds, cfg), ds, cfg, allow_legacy=True
            ) is None

        # 早期权重为 interact/reverse-interact 也分配了 TransR 行。
        # 加载器应自动丢弃这两行，同时保留所有 KG 权重。
        compact_state = model.state_dict()
        old_relation_state = compact_state.copy()
        relation_rows = torch.cat([
            torch.arange(1, ds.n_relations + 1),
            torch.arange(ds.n_relations + 2, ds.n_relations_total),
        ])
        for key in ("rel_att.W_r", "rel_att.rel_emb.weight"):
            compact = compact_state[key]
            expanded = torch.zeros(
                (ds.n_relations_total, *compact.shape[1:]), dtype=compact.dtype
            )
            expanded[relation_rows] = compact
            old_relation_state[key] = expanded
        old_relation_path = f"{tmp}/legacy-relation-layout.pt"
        torch.save(old_relation_state, old_relation_path)
        migrated = ACMR(ds, cfg)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            load_checkpoint(
                old_relation_path, migrated, ds, cfg, allow_legacy=True
            )
        assert any("interaction" in str(item.message) for item in caught)
        assert torch.equal(
            migrated.rel_att.W_r, compact_state["rel_att.W_r"]
        )
        assert torch.equal(
            migrated.rel_att.rel_emb.weight,
            compact_state["rel_att.rel_emb.weight"],
        )

        unknown_path = f"{tmp}/unknown-version.pt"
        torch.save({
            "format_version": 999,
            "model_state": model.state_dict(),
        }, unknown_path)
        try:
            load_checkpoint(unknown_path, ACMR(ds, cfg), ds, cfg)
        except ValueError as exc:
            assert "不支持" in str(exc)
        else:
            raise AssertionError("未知 checkpoint 格式版本未被拒绝")


def test_mm_graph_depth_validation():
    cfg = tiny_cfg()
    cfg.model.use_mm_item_graph = True
    cfg.model.mm_graph_layers = cfg.model.n_gnn_layers + 1
    ds = build_dataset(cfg)
    try:
        ACMR(ds, cfg)
    except ValueError as exc:
        assert "mm_graph_layers" in str(exc)
    else:
        raise AssertionError("越界 mm_graph_layers 被静默截断")


def test_data_contract_validation():
    bad = tiny_cfg()
    bad.model.aggregator = "unknown"
    try:
        validate_config(bad)
    except ValueError as exc:
        assert "aggregator" in str(exc)
    else:
        raise AssertionError("未知 aggregator 未被拒绝")

    bad = tiny_cfg()
    bad.model.aggregator = "unknown"
    dataset = build_dataset(bad)
    try:
        ACMR(dataset, bad)
    except ValueError as exc:
        assert "聚合器" in str(exc)
    else:
        raise AssertionError("直接构造模型时未知 aggregator 未被拒绝")

    bad = tiny_cfg()
    bad.train.model = "unknown"
    try:
        validate_config(bad)
    except ValueError as exc:
        assert "model" in str(exc)
    else:
        raise AssertionError("未知 model 未被拒绝")

    bad = tiny_cfg()
    bad.model.align_source = "kg"
    try:
        validate_config(bad)
    except ValueError as exc:
        assert "align_source" in str(exc)
    else:
        raise AssertionError("KG 邻居伪平行正样本配置未被拒绝")

    bad = tiny_cfg()
    bad.data.val_ratio = -0.1
    try:
        build_dataset(bad)
    except ValueError as exc:
        assert "val_ratio" in str(exc)
    else:
        raise AssertionError("负 val_ratio 未被拒绝")

    bad = tiny_cfg()
    bad.data.val_ratio = 0.6
    bad.data.test_ratio = 0.4
    try:
        build_dataset(bad)
    except ValueError as exc:
        assert "必须小于 1" in str(exc)
    else:
        raise AssertionError("val_ratio + test_ratio >= 1 未被拒绝")

    bad = tiny_cfg()
    bad.data.cold_item_ratio = 1.0
    try:
        build_dataset(bad)
    except ValueError as exc:
        assert "cold_item_ratio" in str(exc)
    else:
        raise AssertionError("cold_item_ratio=1 未被拒绝")

    # 冷商品比例必须按实际有交互的商品计算，而不是按稀疏的完整目录计算。
    n_interacted = 33
    sparse_raw = SimpleNamespace(
        n_users=n_interacted,
        n_items=1000,
        n_entities=1001,
        n_relations=5,
        n_countries=1,
        n_languages=1,
        user_country=np.zeros(n_interacted, dtype=np.int64),
        item_country=np.zeros(1000, dtype=np.int64),
        item_text_feat=np.zeros((1000, 1, 2), dtype=np.float32),
        item_text_lang=np.zeros((1000, 1), dtype=np.int64),
        item_image_feat=np.zeros((1000, 2), dtype=np.float32),
        item_image_mask=np.ones(1000, dtype=np.float32),
        interactions=np.column_stack([
            np.arange(n_interacted), np.arange(n_interacted)
        ]).astype(np.int64),
        triples=np.array([[0, 1, 1000]], dtype=np.int64),
    )
    sparse_cfg = tiny_cfg()
    sparse_cfg.data.cold_item_ratio = 0.05
    sparse_cfg.data.val_ratio = 0.0
    sparse_cfg.data.test_ratio = 0.0
    sparse_ds = CKGDataset(sparse_raw, sparse_cfg)
    if len(sparse_ds.cold_items) != 1 or len(sparse_ds.train_pairs) != 32:
        raise AssertionError(
            "稀疏目录的冷商品划分未按有交互商品计算或吃掉了 warm 训练集"
        )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        meta = {
            "schema_version": SCHEMA_VERSION,
            "data_schema": schema_descriptor(),
            "n_items": 2,
            "n_entities": 4,
            "n_relations": 5,
            "countries": ["M0"],
            "languages": ["x"],
            "n_markets": 1,
            "n_views": 1,
            "entity_layout": {"countries": [2, 3]},
        }
        (root / "meta.json").write_text(
            json.dumps(meta), encoding="utf-8"
        )
        np.save(root / "triples.npy", np.array([
            [-1, 5, 2], [0, 1, 3]
        ], dtype=np.int64))
        np.save(root / "item_country.npy", np.zeros(2, dtype=np.int64))
        language = np.zeros((2, 1), dtype=np.int64)
        np.save(root / "item_text_lang.npy", language)
        np.savez_compressed(
            root / TEXT_META_FILE,
            language=language,
            source=np.full((2, 1), "fixture", dtype="<U16"),
            role=np.full((2, 1), "product_name", dtype="<U16"),
            valid=np.ones((2, 1), dtype=bool),
            is_fallback=np.zeros((2, 1), dtype=bool),
            content_hash=np.asarray([
                [text_content_hash("fixture item 0")],
                [text_content_hash("fixture item 1")],
            ]),
            dedup_mask=np.ones((2, 1), dtype=bool),
            language_confidence=np.ones((2, 1), dtype=np.float32),
        )
        np.save(root / "item_text_feat.npy", np.zeros((2, 1, 2), np.float32))
        np.save(root / "item_image_feat.npy", np.zeros((2, 2), np.float32))
        np.save(root / "item_image_mask.npy", np.ones(2, np.float32))
        np.save(root / IMAGE_AVAILABLE_FILE, np.ones(2, dtype=bool))
        np.savez(
            root / "users.npz",
            user_country=np.zeros(1, dtype=np.int64),
            interactions=np.array([[0, 0]], dtype=np.int64),
        )
        cfg = tiny_cfg()
        cfg.data.off_dir = str(root)
        cfg.data.text_dim = 2
        cfg.data.image_dim = 2
        try:
            OFFRealData(cfg)
        except ValueError as exc:
            assert "头/尾实体编号" in str(exc)
        else:
            raise AssertionError("OFF 负实体编号未被拒绝")

        np.save(root / "triples.npy", np.array([
            [0, 5, 2], [0, 1, 3]
        ], dtype=np.int64))
        np.savez(
            root / "users.npz",
            user_country=np.zeros(1, dtype=np.int64),
            interactions=np.array([[0, 0], [0, 0]], dtype=np.int64),
        )
        try:
            OFFRealData(cfg)
        except ValueError as exc:
            if "重复用户-商品对" not in str(exc):
                raise
        else:
            raise AssertionError("OFF 重复用户-商品对未被拒绝")


if __name__ == "__main__":
    torch.manual_seed(0); np.random.seed(0)
    print("开始自检 ...")
    test_evaluate_known_scores()
    test_frozen_mm_graph_contract()
    test_graph_does_not_depend_on_split()
    test_mm_graph_preserves_isolated_items()
    test_training_samplers_ignore_heldout_labels()
    test_bpr_sampler_excludes_users_without_market_negatives()
    test_degree_sensitive_pruning_contract()
    test_market_smoothed_sampler()
    test_checkpoint_semantic_fingerprint()
    test_mm_graph_depth_validation()
    test_data_contract_validation()

    cases = {
        "完整模型": {},
        "去掉KG": {"use_kg": False},
        "去掉多模态": {"use_multimodal": False},
        "去掉跨语言对齐": {"use_cross_lingual_align": False},
        "去掉语种对抗": {"use_lang_adversarial": False},
        "去掉市场门控": {"use_market_gate": False},
        "B3市场可靠性残差": {"residual": "market_reliable"},
        "文本平行对齐配置": {"align_source": "parallel"},
        "去国家关系图": {"use_country_graph": False},
        "去模态补全": {"use_modality_completion": False},
        "启用冻结多模态图": {"use_mm_item_graph": True},
        "启用协同内容对比": {"use_collab_content_cl": True},
        "启用度敏感边剪枝": {"use_degree_sensitive_pruning": True},
        "启用等市场采样": {"market_sampling_alpha": 0.0},
        "去冷启动Dropout": {"cold_id_dropout": 0.0},
        "concat融合": {"fusion": "concat"},
        "attention融合": {"fusion": "attention"},
        "gcn聚合器": {"aggregator": "gcn"},
        "graphsage聚合器": {"aggregator": "graphsage"},
    }
    for name, overrides in cases.items():
        cfg = tiny_cfg()
        for k, v in overrides.items():
            setattr(cfg.model, k, v)
        run(name, cfg)

    print("全部自检通过。")
