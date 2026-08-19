# OFF schema-v2 半合成数据管线

本管线把 Open Food Facts 商品目录接入 ACMR。采集范围按 2025-10-26 起的
ASEAN-11 固定为 `TH/SG/PH/ID/MY/TL/VN/KH/BN/MM/LA`。中国不是东盟成员，
不会作为销售市场采集；若商品的公开产地字段为中国，`CN` 仍可作为原产国实体。

商品元数据、文本、图片、类别、品牌、原产国和在售市场来自真实公开数据；
**用户和交互由 `simulate_users.py` 生成**。因此 OFF 是“真实食品商品 + 模拟反馈”
的半合成数据，只能做商品目录分析、回归与机制筛选，不能证明真实电商用户效果。

当前目录中旧的 processed 数据和 OFF 筛选报告是在旧范围
`TH/SG/PH/ID/MY/CN/VN/KH/BN/MM/LA` 上产生的，保留仅为历史复现，不能称为
ASEAN-11 结果。必须按下面流程重新采集、构建、抽特征并重跑实验。

## 运行顺序

在 `off_pipeline/` 中执行：

```bash
# 0. 核对当前成员契约
python fetch_off_dump.py --list-markets

# 1a. 正式研究快照：官方 Parquet 全量导出（推荐）
python fetch_off_dump.py

# 已有本地快照时避免重复下载
python fetch_off_dump.py --parquet /path/to/food.parquet

# 1b. API 试采：可指定市场，不替代超大市场的全量快照
python fetch_off_metadata.py --markets TH ID VN TL

# 2. 显式构建 schema-v2 商品目录与 KG
python build_off_dataset.py

# 3. 国家层矩阵（先读取 processed/meta.json 的完整原产国词表）
python build_country_adj.py --years 2024 2023

# 4. 模拟用户反馈
python simulate_users.py --n-users 6000 --n-inter 120000

# 5. 下载图片并抽取冻结特征
python fetch_off_images.py
python extract_text_feat.py
python extract_image_feat.py

# 6. 严格加载并训练
cd ..
python train.py --data off --model acmr --residual none \
  --split-seed 20260801 --train-seed 20260901 \
  --eval-k 10 20 --market-aggregate macro
```

相对路径固定按项目根解析，默认输出到 `data_off/raw`、`data_off/processed`
和 `data_off/images`。批量采集先写隐藏的 `.part` 文件，完整扫描成功后才原子替换
对应市场快照。目录中遗留的 `china.jsonl` 不会自动删除，也不会进入新构建或哈希。

`data_off/raw/collection_manifest.json` 记录成员生效日期、ASEAN-11 列表、数据
提供方、请求市场、每市场记录数、覆盖类型和 SHA-256。只有 11 个市场均来自完整
Parquet 快照时 `formal_snapshot_ready=true`。API 搜索超过 10,000 条的市场会标记
为 `api_depth_limited_union_of_sorts`，不得描述为完整目录。

`build_off_dataset.py` 默认要求 11 个成员市场的原始文件齐全。调试时可以显式传
`--allow-partial-markets`，但该状态会写入元数据，不能用于正式实验。

## schema v2

`meta.json` 必须同时声明：

```json
{
  "schema_version": 2,
  "data_schema": {
    "name": "acmr_multimodal",
    "version": 2
  }
}
```

每个文本视图持久化以下字段：

- `language`、`source`、`role`；
- `valid`、`is_fallback`；
- `content_hash`、`dedup_mask`；
- `language_confidence`。

文本身份按 Unicode NFKC、`casefold` 和空白折叠规范化，再计算 SHA-256。相同商品的相同哈希只保留第一次参与池化。fallback 视图保留真实来源和回退标记，不冒充独立语言。

图像区分：

- `available`：目录是否提供图片 URL；
- `observed`：是否成功抽到实际图像特征；
- `completion_confidence`：观测图初值为 1、缺图初值为 0；缺图置信度由训练期模型预测，不能由验证/测试质量反推。

旧 schema 或字段不全时 `off_data.py` 会直接报错。升级必须重跑 `build_off_dataset.py`，不得从旧 `item_text_lang.npy` 静默推断 provenance。

## 跨语言文本规则

OFF 构建器保留商品的真实文本来源，并把复制视图显式标为 fallback。跨语言 InfoNCE 只接受同商品、非 fallback、非重复、不同语言且可信的真实 `product_name` 文本。

品牌、类别和其他 KG 邻居**不再**生成跨语言正样本。`align_source=kg/both` 已禁用；KG 关系只用于图传播与纯 KG TransR-style 目标。

OFF 原始 API 声明的语言标签置信度记为 1；缺失语言的回退视图即使有内容，也因 `is_fallback=true` 不会进入对齐。

## 关系与实体编号

- 实体：`[0, n_items)` 商品，随后为品类、品牌、国家和标签实体；
- KG 关系：`1`=所属品类、`2`=品牌、`3`=原产国、`4`=标签属性、`5`=在售市场；
- `0`=interact 只存在于 CF 传播图，不写入 `triples.npy` 的 KG 目标；
- 反向边由 `CKGDataset` 为传播构造，不进入纯 KG sampler。

`sold_in` 关系用于构建逐市场可售目录，统一约束训练负采样与评测候选，避免模型利用“是否可售”的评测捷径。

## 数据诚实性规则

1. `item_country` 只接受 `origins_tags` 或 `manufacturing_places` 观测值；缺失原产国的商品剔除，不推断、不模拟。
2. 模拟的只有用户及交互。跨境率由 Lazada 公开统计校准，原产伙伴分布由 UN Comtrade 双边进口份额校准。
3. 信息源必须分离：`trade_shares.npy` 只用于生成模拟反馈；模型国家图使用 `country_adj_geo.npy`，避免把生成交互的贸易矩阵再作为模型输入形成循环论证。
4. `users.npz` 中用户-商品 pair 必须唯一，并且每条交互都位于用户市场可售目录。
5. 原始数据哈希写入 `meta.json`，实验清单会再次记录并核对。
6. 成员市场必须与采集清单中的 ASEAN-11 完全一致；非成员原始文件不会被静默纳入。
7. 某个小市场在产地完整性筛选后可能没有候选；该市场保留零覆盖记录，但模拟器不会为其生成用户。

## Comtrade 原产伙伴权重

`build_country_adj.py` 必须在 `build_off_dataset.py` 之后运行。它读取商品目录实际
生成的 `countries` 轴，输出 `n_markets × n_countries` 的 schema-v2
`trade_shares.npy`，不再局限于 ASEAN 内部的 11×11 矩阵。商品构建器也不再把
低频但可解析的外国来源压入固定的 Top-16，而是保留全部观测原产国。

- 中国、日本、韩国、美国、澳大利亚等所有观测到的非 ASEAN 原产国使用各自 UN M49
  伙伴编码查询，不再统一赋值 `0.02`；
- `OTHER` 权重为 Comtrade 世界进口总额减去显式保留伙伴后的非负残差；
- 目标市场缺少自报进口时，查询原产伙伴自报出口作为镜像；仍缺失时使用 ASEAN
  汇总进口分布，并在 `country_adj_meta.json` 的 `provenance` 中标记；
- 默认混入 1% 的 ASEAN 汇总进口先验，缓解 preview 数据的稀疏零值；可用
  `--pool-prior` 显式调整，正式配对实验必须固定相同值；
- 矩阵的市场轴、原产国轴、年份、查询 URL、回退来源和 `OTHER` 定义全部写入
  `country_adj_meta.json`。商品目录重建后必须重跑本步骤。

模拟器严格按轴标签重排矩阵。若某市场当前可售外国原产池在 Comtrade 中全部为
零，才退回按该市场各原产池的目录规模采样，并在 `simulation_meta.json` 中记录
市场和实际回退选择次数。旧 11×11 文件会直接报错，不会静默兼容。

## 冷启动与评测

OFF 冷商品按商品拆为不相交 validation/test。其目标市场训练交互被移除，随机 ID 被硬屏蔽。

冷正例必须与完整目标市场可售目录共同排名，不得只用 cold pool。评测先排除训练正例，再为每个 split、K 和子集预计算市场资格：只有该市场所有受评用户都有至少 K 个候选，才进入对应宏平均。K 不动态缩小；排除市场、用户、正例和覆盖率写入结果清单。

当前 schema-v2 OFF 已通过一轮 CPU 训练及 @10/@20 完整目录冷评测。

## B0-B3 筛选边界

OFF 只运行前三组配对种子：

```text
(20260801, 20260901)
(20260802, 20260902)
(20260803, 20260903)
```

需要比较 B0 `none`、B1 `fused`、B2 `decoupled`、B3 `market_reliable`，并单独报告 genuine-multilingual、cold、missing-image 和 long-tail 子集。

筛选工具：

```bash
python research/run_off_screening.py --overwrite

# 或对已有 12 份清单单独重生统计报告
python research/validate_experiments.py 'results/off_screening/*.json' \
  --mode screening \
  --output research/off_screening_report.json \
  --markdown research/OFF_SCREENING_REPORT.md
```

当前 12 组运行已完成。整体 market-macro NDCG@10 的三种子平均差为：B2-B1 `+0.003969`、B3-B2 `-0.006170`、B3-B0 `+0.007862`。B3-B0 在 cold、missing-image、genuine-multilingual 和 long-tail 上分别为 `+0.000363`、`+0.000521`、`+0.002990` 和 `-0.003319`。详细均值、样本标准差、配对效应、覆盖分母与 11 项统计谬误扫描见 `research/OFF_SCREENING_REPORT.md`。

无论 OFF 差值方向如何，筛选报告都保持 `default_residual=none`。只有 XMarket 锁定的五种子外部验收能够决定是否考虑启用 B3。

## 已知限制

- 用户、偏好与交互是模拟的，外部效度不能由真实商品元数据补足。
- Comtrade preview 接口仍可能限流或缺报；镜像、汇总先验和目录回退均会显式记录，正式报告需披露相应覆盖率。
- API 抓取可能受搜索深度和服务稳定性影响；正式快照优先使用官方全量 dump，并记录哈希。
- Open Food Facts 是食品商品目录，不是 Shopee/Lazada 的全品类商城目录，也不含真实浏览、点击或购买日志。
- 东帝汶等小市场可能极稀疏；重加权、复制文本或模拟商品都不能补成真实市场证据。
- 商品 catalog 和 side information 在训练期已知，当前 cold 协议是 transductive catalog，不是训练时完全不可见新节点的严格 inductive 设置。
