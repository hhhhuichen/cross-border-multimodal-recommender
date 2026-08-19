# XMarket Electronics 外部验证管线

该管线只从 XMarket 官方分发地址读取 US、CN、IN、SG 的 Electronics 数据。
原始文件、图片及处理后的特征均由 `.gitignore` 排除；仓库只保存代码、官方
URL 和运行时生成的 SHA-256 清单。官方页面未提供本项目可以确认的显式
LICENSE，因此不要再分发原始文件。

当前本地 `split_seed=20260801` 已按 catalog protocol v1 构建：联合目录
36,652 个商品，US/CN/IN/SG 市场目录分别为 35,943/1,303/6,579/451；
5-core 仍只产生 21,688/21/194/8 个交互商品。两套分母会分别写入
`meta.json.catalog_protocol`，不得混用。

```bash
python xmarket_pipeline/fetch_xmarket.py
python xmarket_pipeline/build_xmarket_dataset.py \
  --split-seed 20260801 --lid-model /path/to/lid.176.ftz
python xmarket_pipeline/fetch_images.py
python xmarket_pipeline/extract_features.py
python train.py --data xmarket --model acmr --residual none \
  --split-seed 20260801 --train-seed 20260901 \
  --selection-target overall --market-aggregate macro --eval-k 10 20
```

`none` 是当前默认残差。`market_reliable` 只是待验收的 B3 实验项；它尚未通过 XMarket 外部验收，不得作为推荐默认。最终实验必须对 B0-B3 以及独立 BPR-MF、LightGCN、VBPR 共享五组配对种子：`split_seed=20260801..20260805`，对应 `train_seed=20260901..20260905`。最终验收器会强制检查三条独立基线是否齐全。

若语言识别模型暂不可用，只能显式传 `--disable-alignment`。此模式会将语言
置信度置零，跨语言 InfoNCE 不产生正例，不能用于声称多语言对齐有效。

预处理遵循以下固定协议：

- 每个市场的评分交互独立做用户、商品 5-core；任何星级评分都视为相关交互，
  与 FOREC 论文的购买行为定义一致。候选目录则由过滤前的官方市场元数据
  定义，不能随 5-core 缩小。
- ASIN 是跨市场唯一商品键；用户 ID 带市场命名空间，不跨市场链接。
- US 作为辅助训练市场；CN、IN、SG 在本地用户内按时间 leave-two-out，缺少时间戳时使用 `split_seed` 控制的确定性 leave-two-out。
- 目标商品按训练种子无关的稳定哈希和流行度层划分约 5% cold-validation、
  5% cold-test；其目标市场反馈全部移出训练，但 US 反馈保留。
- KG 只使用品牌、类别、共购和 sold-in。XMarket 不提供可信商品原产地，
  `supports_item_origin=false`，不得输出原产地意义上的跨境指标。
- 评测使用 XMarket 元数据中观测到的完整目标市场目录；被 5-core 剔除的
  低交互商品仍作为零训练度候选，只使用内容/KG 表示。评测按固定 K 预注册
  合格市场；不复现 FOREC 的 99 个采样负例，因此数值不能与论文表格直接
  横向比较。
