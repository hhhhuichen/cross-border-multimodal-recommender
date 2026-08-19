# -*- coding: utf-8 -*-
"""
ACMR (ASEAN Cross-border Multimodal Recommender) 全局配置。

论文对应：《融合多语种知识图谱与GNN的东盟跨境多模态推荐系统研究》
所有超参数集中在此，做消融实验时只需改这里的开关（use_* 字段）。
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    # ---- 数据源：synthetic / OFF / XMarket ----
    source: str = "synthetic"
    # 各真实数据加载器均按项目根解析相对路径，不依赖启动命令所在目录。
    off_dir: str = "./data_off/processed"
    # XMarket 原始数据不提交仓库。预处理器会把可训练数组写入此目录。
    xmarket_dir: str = "./data_xmarket/processed"
    schema_version: int = 2

    # ---- 合成数据规模（换成真实数据时这几项由数据集自动推断）----
    n_users: int = 3000
    n_items: int = 2000
    n_entities: int = 4000          # KG 实体数（含 item 对齐实体）
    n_relations: int = 12           # KG 关系类型数（不含反向关系）
    n_interactions: int = 60000
    n_triples: int = 40000

    # ---- 东盟跨境场景：国家 / 语言 ----
    countries: List[str] = field(default_factory=lambda: [
        "TH", "SG", "PH", "ID", "MY", "TL", "VN", "KH", "BN", "MM", "LA"
    ])
    languages: List[str] = field(default_factory=lambda: [
        "zh", "th", "vi", "id", "ms", "en", "tl", "my", "km", "lo",
        "tet", "pt",
    ])

    # ---- 预训练多模态特征维度 ----
    text_dim: int = 768             # LaBSE / XLM-R 输出维度
    image_dim: int = 512            # CLIP ViT-B/32 图像塔输出维度
    n_text_views: int = 2           # 每个 item 保留几个语种的文本视图（用于跨语言对齐）

    # ---- 划分 ----
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    # 冷启动保留集：从有交互商品中随机抽这一比例，其全部交互从训练集移除并评测
    # （完整目录中的零交互商品没有 ground truth；至少保留一个 warm 商品）
    cold_item_ratio: float = 0.05
    # 冷商品必须按 item 拆成 validation/test 两个不相交集合，才能在不查看冷测试
    # 结果的前提下选择 cold_id_dropout、图传播强度和对比损失权重。
    cold_val_item_ratio: float = 0.25
    # seed 保留为旧脚本兼容入口；split_seed 非 None 时优先控制数据生成与划分。
    seed: int = 2026
    split_seed: Optional[int] = None


@dataclass
class ModelConfig:
    embed_dim: int = 64             # 统一表示维度 d
    relation_dim: int = 64          # KG 关系空间维度
    n_gnn_layers: int = 3           # GNN 传播层数 L
    layer_dropout: float = 0.1
    message_dropout: float = 0.1
    aggregator: str = "bi"          # bi | gcn | graphsage

    # ---- 模块开关（消融实验入口）----
    use_kg: bool = True             # 关闭 -> LightGCN-style 参数无关传播，非标准 LightGCN
    use_multimodal: bool = True     # 关闭 -> 仅 ID embedding
    use_cross_lingual_align: bool = True   # 跨语言对比对齐
    use_lang_adversarial: bool = True      # 语种对抗（语言无关表示）
    use_market_gate: bool = True    # 跨境市场自适应门控

    # ---- 四个创新点模块（对应 README 第五节原局限）----
    # 对齐只允许 schema-v2 验证过的同商品异语真实文本。
    # 旧版 kg/both 会把品牌或品类邻居冒充平行商品，已禁用。
    align_source: str = "parallel"
    use_country_graph: bool = True  # 国别嵌入先在国家关系图（地理邻接/贸易）上传播
    use_modality_completion: bool = True   # 文本->图像特征预测，补全缺图商品
    cold_id_dropout: float = 0.2    # DropoutNet-inspired 商品 ID dropout；非论文的
                                    # 两阶段蒸馏复现，0 = 关闭训练期 dropout

    # ---- 文献驱动的图去噪与协同-内容对齐 ----
    # FREEDOM/LATTICE-inspired：只用冻结侧信息构建 positive symmetric kNN 图，
    # 不使用交互或测试标签；图结构在训练中固定，避免动态图学习放大噪声。
    # 合成数据三种子消融未显示稳定增益，默认关闭；真实数据上验证后再启用。
    use_mm_item_graph: bool = False
    mm_graph_k: int = 10
    mm_graph_layers: int = 1
    mm_graph_text_weight: float = 0.5
    mm_graph_beta: float = 0.1       # 冻结 ID 图 side branch 强度；0 = 禁用该分支
    mm_graph_chunk_size: int = 512   # 分块精确 top-k，避免创建完整 N x N 相似度矩阵

    # FREEDOM-inspired：训练图按端点度数重采样 user-item 边；推理恢复完整
    # 训练图。论文候选模块，需配对消融后再决定是否默认启用。
    use_degree_sensitive_pruning: bool = False
    interaction_prune_ratio: float = 0.6

    # 跨市场采样平滑：市场 m 的总采样质量正比于 n_m^alpha。
    # alpha=1 为原始交互分布；alpha=0 是受 FOREC 跨市场动机启发的启发式，
    # 不是 FOREC 的 MAML/fork 实现。
    market_sampling_alpha: float = 1.0

    # CLCRec-inspired 单项内容-ID InfoNCE；不是完整 CLCRec。
    # 协同 ID 在该损失中 detach，避免内容分支反向拖动 BPR 的锚点。
    use_collab_content_cl: bool = False
    lambda_collab_cl: float = 0.01
    collab_temperature: float = 0.2

    fusion: str = "gated"           # gated | concat | sum | attention
    kg_att_scale: float = 0.5       # KG 边注意力的叠加系数 γ（interact 边走度归一化主干）

    # ---- 损失权重 ----
    lambda_kg: float = 1.0          # TransR-style 关系空间损失
    lambda_align: float = 0.2       # 跨语言对齐损失
    lambda_adv: float = 0.1         # 语种对抗损失
    lambda_mm_complete: float = 0.5 # 跨模态补全损失
    l2_reg: float = 1e-5
    temperature: float = 0.2        # InfoNCE 温度

    # ---- 文献驱动的个性化内容残差（B0-B3）----
    # none=B0；fused=B1；decoupled=B2；market_reliable=B3。
    residual: str = "none"
    residual_hidden_dim: int = 64
    residual_beta_init: float = 0.01
    completion_confidence_weight: float = 1.0
    auxiliary_item_batch_size: int = 512


@dataclass
class TrainConfig:
    model: str = "acmr"             # bpr_mf | lightgcn | vbpr | acmr
    epochs: int = 30
    batch_size: int = 2048
    kg_batch_size: int = 4096
    lr: float = 1e-3
    weight_decay: float = 0.0
    eval_every: int = 5
    topk: List[int] = field(default_factory=lambda: [10, 20])
    market_aggregate: str = "macro" # micro | macro
    # 一个 macro-batch 只做一次全图传播；不得小于普通 batch_size。
    cf_macro_batch_size: int = 16384
    train_seed: int = 2026
    # 0 = 全量用户；正数 = 用户过多时用固定 seed 抽样。实验报告必须披露该值。
    eval_max_users: int = 2000
    selection_target: str = "cross"  # overall | cross | cold；只用验证集选 checkpoint
    early_stop_patience: int = 5
    device: str = "cuda"            # 无 GPU 时代码会自动回落到 cpu
    ckpt_path: str = "./ckpt/acmr_best.pt"
    result_path: str = "./results/latest.json"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def resolved_split_seed(cfg):
    """返回数据生成/划分种子，同时保留旧 cfg.data.seed 的行为。"""
    seed = cfg.data.split_seed
    return int(cfg.data.seed if seed is None else seed)


def validate_config(cfg):
    """集中校验所有会改变数据、前向或评测语义的配置。"""
    d, m, t = cfg.data, cfg.model, cfg.train

    if d.source not in {"synthetic", "off", "xmarket"}:
        raise ValueError(f"未知 data source={d.source!r}")
    if int(d.schema_version) != 2:
        raise ValueError("当前代码只接受 data schema_version=2")
    if not 0.0 <= d.val_ratio < 1.0 or not 0.0 <= d.test_ratio < 1.0:
        raise ValueError("val_ratio/test_ratio 必须在 [0,1) 内")
    if d.val_ratio + d.test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio 必须小于 1")
    if not 0.0 <= d.cold_item_ratio < 1.0:
        raise ValueError("cold_item_ratio 必须在 [0,1) 内")
    if not 0.0 < d.cold_val_item_ratio < 1.0:
        raise ValueError("cold_val_item_ratio 必须在 (0,1) 内")
    if min(d.n_users, d.n_items, d.n_entities, d.text_dim, d.image_dim) <= 0:
        raise ValueError("数据规模与特征维度必须为正数")

    if m.aggregator not in {"bi", "gcn", "graphsage"}:
        raise ValueError(f"未知 aggregator={m.aggregator!r}")
    if m.fusion not in {"gated", "concat", "sum", "attention"}:
        raise ValueError(f"未知 fusion={m.fusion!r}")
    if m.residual not in {"none", "fused", "decoupled", "market_reliable"}:
        raise ValueError(f"未知 residual={m.residual!r}")
    if m.align_source != "parallel":
        raise ValueError(
            f"非法 align_source={m.align_source!r}；"
            "跨语言对齐只允许 schema-v2 真实平行文本"
        )
    if m.n_gnn_layers < 0 or min(m.embed_dim, m.relation_dim,
                                 m.residual_hidden_dim,
                                 m.auxiliary_item_batch_size) <= 0:
        raise ValueError("层数不得为负，表示维度必须为正数")
    if not 0.0 <= m.cold_id_dropout < 1.0:
        raise ValueError("cold_id_dropout 必须在 [0,1) 内")
    if m.temperature <= 0 or m.collab_temperature <= 0:
        raise ValueError("对比学习温度必须为正数")
    if not 0.0 <= m.interaction_prune_ratio < 1.0:
        raise ValueError("interaction_prune_ratio 必须在 [0,1) 内")
    if not 0.0 <= m.market_sampling_alpha <= 1.0:
        raise ValueError("market_sampling_alpha 必须在 [0,1] 内")
    if not 0.0 <= m.mm_graph_beta <= 1.0:
        raise ValueError("mm_graph_beta 必须在 [0,1] 内")
    nonnegative = {
        "lambda_kg": m.lambda_kg,
        "lambda_align": m.lambda_align,
        "lambda_adv": m.lambda_adv,
        "lambda_mm_complete": m.lambda_mm_complete,
        "lambda_collab_cl": m.lambda_collab_cl,
        "l2_reg": m.l2_reg,
        "completion_confidence_weight": m.completion_confidence_weight,
    }
    bad = [name for name, value in nonnegative.items() if value < 0]
    if bad:
        raise ValueError(f"损失权重必须非负：{', '.join(bad)}")

    if t.model not in {"bpr_mf", "lightgcn", "vbpr", "acmr"}:
        raise ValueError(f"未知 model={t.model!r}")
    if m.residual != "none" and t.model != "acmr":
        raise ValueError("个性化 residual 仅适用于 model=acmr")
    if m.residual != "none" and not m.use_multimodal:
        raise ValueError("个性化 residual 要求 use_multimodal=True")
    if t.market_aggregate not in {"micro", "macro"}:
        raise ValueError("market_aggregate 必须是 micro 或 macro")
    if t.selection_target not in {"overall", "cross", "cold"}:
        raise ValueError(f"未知 selection_target={t.selection_target!r}")
    if t.epochs <= 0 or t.batch_size <= 0 or t.kg_batch_size <= 0:
        raise ValueError("epochs/batch_size/kg_batch_size 必须为正数")
    if t.cf_macro_batch_size < t.batch_size:
        raise ValueError("cf_macro_batch_size 不得小于 batch_size")
    if t.lr <= 0 or t.eval_every <= 0 or t.early_stop_patience <= 0:
        raise ValueError("lr/eval_every/early_stop_patience 必须为正数")
    if not t.topk or any(int(k) <= 0 for k in t.topk):
        raise ValueError("topk 必须是非空正整数列表")
    if len(set(map(int, t.topk))) != len(t.topk):
        raise ValueError("topk 不得包含重复值")
    if t.eval_max_users < 0:
        raise ValueError("eval_max_users 必须为非负整数")
    return cfg
