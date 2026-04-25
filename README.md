# M-POPE GitHub Ready

这是从 `/data/libing/SSAR` 中整理出来的轻量发布版，目标是保留**和论文《M-POPE- A Multi-dimensional Hallucination Benchmark for MLLMs with Inference-time Attention Reweighting》直接相关**的内容，同时去掉不适合上传 GitHub 的大体积数据。

## 保留内容

- `paper_materials/`
  - 论文 PDF
  - 方法说明文档 `method.md`
  - 两份可直接预览的 HTML 方法稿
- `core_reweighting/`
  - 来自 `SAR/` 的核心推理期注意力重加权代码
  - 头配置文件
  - 轻量数据路径占位（无原始数据集）
  - 少量示例结果
- `benchmark_head_analysis/`
  - 多维 POPE / mini POPE 构造脚本
  - 幻觉头识别与聚合脚本
  - 各模型、各维度的最终幻觉头 JSON 与热力图
- `experiment_runners/`
  - 一键运行脚本
  - `sydney_exp/` 下与方法实验直接相关的代码脚本

## 已剔除内容

- 所有原始图像与图片目录内容
- 大体积 `dataset/`、`datasets/`、`dataset_pope/` 数据文件
- 各类 `result/`、`results/` 中的大体积原始推理输出
- `__pycache__/`、`.git/` 等缓存或仓库元数据
- 上游大型基线仓库的完整拷贝

## 目录建议

- 如果你要**直接传 GitHub**，优先使用这个文件夹作为新仓库根目录。
- 如果你要保留最核心的可读代码，先看 `core_reweighting/`。
- 如果你要补论文方法与头分析说明，重点看 `paper_materials/` 和 `benchmark_head_analysis/`。

## 注意

- 本目录已完全移除 `dataset/`、`datasets/` 类型目录，不包含原始数据集文件。
- 若后续需要复现实验，请在本地另行放置数据，并按脚本中的路径或 README 说明补齐。
- 当前整理版更偏向**论文开源/代码展示**，不是“拿来即跑的全量实验包”。
