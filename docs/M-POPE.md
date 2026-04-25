# M-POPE: 面向多维幻觉评测的细粒度基准构建 (M-POPE Benchmark Construction)

## 1. 概述与问题形式化 (Overview and Formulation)

现有的基于轮询的物体探测评估（POPE）主要集中在二元的存在性（Existence）验证上。虽然有效，但这种单维度的评估掩盖了多模态大模型（LVLM）在更复杂的细粒度语义理解上的缺陷。为了弥补这一差距，我们提出了 **M-POPE (Multi-dimensional POPE)**，这是一个标准化的多维幻觉评测框架。

我们将视觉幻觉的评估形式化为一个四元组 $E = \{D, S, T, M\}$，其中：
- $D$ 代表评估维度集合，$D = \{Exist, Count, Position, Color\}$。
- $S$ 代表负采样策略（Negative Sampling Strategies），用于构建困难负样本。
- $T$ 代表提示模板（Prompt Templates），用于将视觉事实转化为二元问答（Yes/No）。
- $M$ 代表评估指标（Metrics）。

## 2. 多维负采样策略 (Multi-dimensional Negative Sampling)

为了严格评估模型在不同语义维度上的真实感知能力，M-POPE 针对每个维度设计了特定的负采样机制，以构建具有挑战性的幻觉诱发问题。对于给定的图像 $I$ 和基准事实（Ground Truth），我们采用了如下策略（详细 Prompt 设置见附录）：

### 2.1 存在性维度 (Existence Dimension)
继承原始 POPE 的设计，我们采用三种层级的负采样策略来评估对象识别的鲁棒性：
- **Random Sampling**: 随机采样图像中不存在的类别作为负样本。
- **Popular Sampling**: 采样数据集中高频出现但当前图像中不存在的类别，评估模型对高先验分布的依赖。
- **Adversarial Sampling**: 基于共现频率（Co-occurrence），采样与图像中现有物体高度相关但实际缺失的物体（例如，图中有“键盘”时询问是否存在“鼠标”），以测试模型对上下文一致性的过度推断。

### 2.2 数量维度 (Count Dimension)
我们设计了三种策略来区分计数错误与虚构幻觉：
- **Strategy 1: Fabrication-based**: 针对图像中**完全不存在**的物体类别询问具体数量（例如，“图中有两只老虎吗？”）。这检测模型是否会基于语言先验虚构数量。
- **Strategy 2: Basic Counting Error-based**: 针对图像中**存在**的物体，询问一个与其真实数量显著不同的数值（例如，实际有3只，询问“是否有7只”或“是否有0只”）。
- **Strategy 3: Multi-category Quantity Interference-based**: 在多类别场景中，将物体 B 的真实数量错误地归因于物体 A（例如，图中有5只猫和3只狗，询问“是否有5只狗？”）。这测试模型在复杂场景下的属性绑定能力。

### 2.3 位置/空间维度 (Position Dimension)
该维度考察模型对物体间几何关系及全局布局的理解，包含三种策略：
- **Strategy 1: Fabrication-based**: 询问图像中**不存在**的物体与其他物体的空间关系（例如，“兔子在老虎前面吗？”）。
- **Strategy 2: Relative Position Reversal-based**: 反转或替换两个**真实存在**物体间的相对空间关系（例如，真实关系是“杯子在桌子上”，询问“杯子在桌子下吗？”）。
- **Strategy 3: Spatial Region Error-based**: 错误描述物体在全局场景中的分布区域（例如，真实位于“左上角”，询问是否在“右下角”）。

### 2.4 属性/颜色维度 (Color Dimension)
该维度评估细粒度的视觉属性绑定能力，包含四种策略：
- **Strategy 1: Fabrication-based**: 询问**不存在**物体的颜色属性（例如，“蓝色的湖存在吗？”）。
- **Strategy 2: Primary Color Conflict-based**: 将现有物体的颜色替换为明显的冲突颜色（色相相反，如“红玫瑰”问成“蓝玫瑰”）。
- **Strategy 3: Linguistic Prior-induced**: 利用常识先验构建与视觉事实矛盾的问题（例如，图中有绿色的香蕉，询问“香蕉是黄色的吗？”），检测模型是否忽略视觉事实而依赖语言幻觉。
- **Strategy 4: Color Confusion-based**: 在多物体场景中，将物体 B 的颜色错误绑定到物体 A 上（例如，图中有红柿子和绿苹果，询问“柿子是绿色的吗？”）。

## 3. 评测协议 (Evaluation Protocol)

为了保证测试的标准化和不同模型间的可比性：
1. **统一问答格式**: 所有维度的查询均被格式化为封闭式二元问题（"Is there a...", "Are there...", "Is the... left of...", "Is the... red?"），模型仅需回答 "Yes" 或 "No"。
2. **对抗性类别平衡 (Adversarial Class Balance)**: 考虑到 VLM 普遍存在严重的“肯定性偏差”（倾向于回答 Yes），我们在每个维度内将正样本（Yes）和负样本（No）的比例设定为 **3:7**。这种非均衡分布不仅打破了模型对 50/50 分布的预期，更严厉地惩罚了盲猜 "Yes" 的策略（盲猜 Yes 的准确率将仅为 30%），从而更显著地暴露模型的幻觉问题。
3. **指标**: 我们报告各类别的 Accuracy, Precision, Recall 以及 F1 Score，重点关注 "Yes" 回答的比例（Yes-Ratio）以监控模型的肯定性偏差（Yes-bias）。
