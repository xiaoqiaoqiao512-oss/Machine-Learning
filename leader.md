# MLLM 学习路线与导师协作约定

> 最后更新：2026-08-05  
> 当前主线：完成 MiniGPT 闭环，逐步具备阅读、微调和修改开源大模型与多模态大模型的能力

## 1. 最终目标

本路线的目标不是停留在：

```text
Qwen2.5-VL + 自定义数据集 + LoRA
```

而是最终做到：

> 能独立阅读和修改大型多模态模型（MLLM）的核心源码，完成一次规范的模型微调，并设计、实现和验证一个新的视觉模块。

期望形成五层能力：

1. **原理层**：理解 Transformer、ViT、视觉语言对齐和自回归生成。
2. **源码层**：能追踪图像从预处理到视觉 token，再到语言模型输出的完整数据流。
3. **训练层**：能准备数据、完成 SFT/LoRA、排查训练问题并评估模型。
4. **研究层**：能提出可验证的结构改进，设计基线、消融实验和评价指标。
5. **工程层**：理解真实项目的模块组织、参数管理、训练流程与部署边界。

### 主次关系

当前的第一主线是：

```text
Transformer 原理与实现
        ↓
LLM 源码与训练
        ↓
ViT 与 MLLM
        ↓
微调与架构研究
```

C++ 后端与模型部署是后续工程支线：

```text
C++ HTTP Server → 模型推理服务 → 返回结果
```

它可以在模型学习主线成熟后重新接入，但不能挤占当前理解模型和训练机制的时间。

### 暂不作为近期重点

- C++ 推理服务、ONNX 和 TensorRT；
- RAG 和 Agent 应用开发；
- 大规模分布式预训练；
- 一开始就同时接入 SAM、OCR、Layout Encoder 和 MoE。

C++ 经历有助于理解内存、模块边界和工程源码，未来可以用于模型部署，但近期不以部署作为阶段验收目标。

## 2. 当前进度与能力定位

### 已有基础

- 熟悉 C++、STL、内存、编译链接和 CMake；
- 有算法与 C++ 后端项目经验；
- 理解前向传播、损失函数、反向传播和梯度下降；
- 使用 PyTorch MLP 完成过 MNIST 分类；
- 已具备基础图像分类训练经验，但当前仓库尚没有独立的 CNN 实现；
- 理解并手写过 Self-Attention、Multi-Head Attention 和 Transformer Block；
- MiniGPT 已完成 Embedding、Position Embedding、Post-LN Block 堆叠和 LM Head；
- 已实现字符级 Tokenizer、滑动窗口 TextDataset 和 causal attention mask；
- `train.py` 已包含 DataLoader、AdamW、CrossEntropyLoss、backward 和 optimizer step 的基本骨架。

### 当前 MiniGPT 代码审计

仓库结构已经形成：

```text
MiniGPT/
├── config.py
├── dataset.py
├── inference.py
├── train.py
├── model/
│   ├── attention.py
│   ├── block.py
│   ├── embedding.py
│   ├── gpt.py
│   └── mlp.py
└── tokenizer/
    └── tokenizer.py
```

当前已经完成的代码不等于训练闭环已经完成。实际待解决问题：

- `train.py` 引用了尚不存在的 `ToyDataset`，与当前 `TextDataset` 不一致；
- tokenizer 的真实 `vocab_size` 尚未接入模型配置；
- 真实文本 → tokenizer → token ids → TextDataset 的链路尚未接通；
- `inference.py` 为空，尚无自回归生成；
- 尚无 validation、checkpoint、随机种子和 loss 曲线；
- `config.d_ff` 与 `dropout` 目前没有真正用于模型；
- `device="mps"` 写死，需要增加可用设备选择或回退；
- 当前 Block 是 Post-LN，后续需要在闭环完成后再改为 Pre-LN。

### 当前仓库结构判断

当前文件可以分成三层：

```text
Machine-Learning/
├── linear.py、train.py
│   └── NumPy 手写 Linear、ReLU、反向传播与参数更新
├── self_attention.py、multi_head_attention.py、transform_block.py、model.py
│   └── 从 Self-Attention 到早期单文件 MiniGPT 的学习草稿
├── mnist/
│   └── PyTorch MLP 分类训练与验证
└── MiniGPT/
    └── 当前唯一需要继续维护的模块化 Transformer 主项目
```

这一结构能保留学习轨迹，但需要明确主次：

- `MiniGPT/` 是当前主实现，后续功能和测试只在这里推进；
- 根目录 Attention/Transformer 文件视为历史练习，不继续复制功能；
- `mnist/` 是 PyTorch 训练流程练习，不应称为 CNN 项目；
- `data/MNIST/` 只是数据，不代表已经实现 CNN；
- 当前没有 ViT、CLIP 或 MLLM 代码，因此路线尚未进入视觉阶段。

### 已发现但暂不扩大的代码问题

- 根目录 `multi_head_attention.py` 使用 `sqrt(num_heads)` 缩放 attention score，正确缩放量应为 `sqrt(head_dim)`；`MiniGPT/model/attention.py` 已使用正确形式；
- 根目录 `train.py` 的 `backward` 定义与调用参数不一致，属于早期 NumPy 草稿；
- `MiniGPT/model/mlp.py` 固定使用 `4*d_model`，尚未读取 `config.d_ff`；
- `MiniGPT/config.py` 声明了 `dropout`，但模型尚未使用；
- 目前测试文件主要是打印 shape/结果，还不是带断言的自动化测试。

这些问题中，历史练习文件暂不重构；先确保 `MiniGPT/` 成为单一、正确、可验证的实现。

### 当前定位

目前处于：

> 已能实现 Transformer 核心模块，但还缺少完整训练闭环、视觉 Transformer 基础、MLLM 数据流理解和真实模型微调经验。

因此不应立即修改 Qwen2.5-VL。正确顺序是：

```text
完成 MiniGPT
    ↓
掌握 ViT 与视觉语言模型基础
    ↓
阅读 Qwen2.5-VL 论文与源码
    ↓
复现微调基线
    ↓
研究同类模型
    ↓
设计一个可验证的结构改进
```

## 3. 学习、研究与导师协作原则

### 学习者特点

- 以 C++ 为主要编程背景，Python 能阅读并编写基础代码；
- 喜欢追踪底层计算、梯度、tensor shape 和工程组织；
- 已理解 Transformer 基础，不需要从概念科普重新开始；
- 希望先实现核心模块，再使用高级库和阅读真实源码；
- 数学上能理解矩阵、梯度和链式法则，需要加强复杂张量计算与 autograd。

### 教学方式

每次教学应遵守以下约定：

1. 解释结论时同时说明设计动机、数学计算、数据流和 shape 变化；
2. 避免用过度拟人的语言解释 Q/K/V，优先使用向量、相似度和加权求和；
3. 代码按模块和最小改动给出，不一次提供整个项目答案；
4. 先让学习者预测 shape、输出或梯度，再运行代码验证；
5. 检查理解时采用“提问 → 等待回答 → 点评 → 专业表达”的方式；
6. 指出错误时说明错误模型、正确模型以及两者差异；
7. 阅读源码时优先追踪一次真实 forward，而不是从文件第一行顺序阅读；
8. 每个主题最终落实到代码、测试或实验，而不是只完成阅读。

### 每次辅导的标准节奏

```text
确定一个最小问题
    ↓
让学习者先解释或预测
    ↓
画数据流与 shape
    ↓
实现一个小模块
    ↓
运行测试或打印验证
    ↓
总结为源码/面试表达
```

### 学习与研究原则

1. **先闭环，后扩展**：先把最小系统跑通，再增加复杂模块。
2. **先论文，后源码**：先画清架构图，再进入数千行实现。
3. **先数据流，后细节**：先追踪张量形状和模块边界，再研究优化技巧。
4. **先复现，后创新**：没有可信 baseline，就无法证明新模块有效。
5. **一次只改一个变量**：结构、数据和训练策略不能同时变化。
6. **所有结论都要有证据**：至少需要对照实验、消融或错误案例分析。
7. **代码能跑不等于理解**：必须能解释输入输出、参数、梯度和设计目的。

## 4. 十六周总路线

| 时间 | 主任务 | 阶段产物 |
| --- | --- | --- |
| 第 1～2 周 | 完成 MiniGPT 训练与生成闭环 | 可训练、验证、保存、加载和生成的 MiniGPT |
| 第 3～4 周 | Transformer、ViT | ViT 小实验与两份结构笔记 |
| 第 5 周 | CLIP、BLIP-2、LLaVA | 一张 MLLM 架构演化对比图 |
| 第 6～7 周 | Qwen2.5-VL 论文与处理流程 | 完整数据流图和论文阅读报告 |
| 第 8 周 | Qwen2.5-VL 核心源码 | 模块调用图、shape 表和源码笔记 |
| 第 9～10 周 | 复现 Qwen2.5-VL 微调 | 可复现的 SFT/LoRA baseline |
| 第 11～12 周 | 文档/OCR MLLM 横向研究 | 模型对比表与候选创新点评估 |
| 第 13～14 周 | 实现一个最小创新模块 | 可训练的新模块与单元测试 |
| 第 15～16 周 | 对照、消融和论文整理 | 实验结果、错误分析和大创报告初稿 |

十六周是理想安排。若算力、数据或课程时间不足，应延长周期，而不是跳过 baseline 和消融实验。

## 5. 阶段零：完成 MiniGPT 闭环（第 1～2 周）

### 为什么仍然需要完成

MiniGPT 是后续阅读 Qwen2.5-VL 语言模型部分的最小沙盘。必须亲手走完：

```text
文本 → token → batch → logits → loss → backward
     → checkpoint → prompt → 自回归生成
```

### 已完成

- [x] 字符级 tokenizer 的 `encode`、`decode` 和词表；
- [x] 滑动窗口 TextDataset，`x` 与 `y` 相差一个 token；
- [x] Multi-Head Attention 中的 Causal Mask；
- [x] MiniGPT 的 Embedding、Block、LM Head 和 forward；
- [x] 训练循环的基本组成：DataLoader、loss、backward 和 optimizer step。

### 下一步任务

按依赖顺序推进，不并行展开：

#### P0：先让真实训练链路运行

- [ ] 修复 `ToyDataset`/`TextDataset` 接口不一致；
- [ ] 准备一个仓库内的小型 UTF-8 文本语料；
- [ ] 用 tokenizer 得到 token ids，并将实际 `vocab_size` 接入 config；
- [ ] 增加随机种子与 CPU/MPS/CUDA 设备选择；
- [ ] 运行一次 forward/backward，并打印关键 shape 与梯度是否有限；
- [ ] 用固定单 batch 过拟合，验证模型、标签和优化器链路。

#### P1：再补训练质量与可验证性

- [ ] 划分 train/validation 数据，增加 `model.eval()` 与 `torch.no_grad()`；
- [ ] 记录 train/validation loss；
- [ ] 为 causal mask、shape、tokenizer round-trip 添加带断言测试；
- [ ] 让 `d_ff` 和 `dropout` 配置真正生效；
- [ ] 使用 `reshape` 或确认连续性后再展平 logits/labels。

#### P2：最后完成模型使用闭环

- [ ] 实现 checkpoint 保存、加载与恢复训练；
- [ ] 保存足以还原模型的 config 和 tokenizer 词表；
- [ ] 实现 greedy、temperature 和 top-k 生成；
- [ ] 测试生成长度与上下文截断；
- [ ] 绘制 loss 曲线并记录一个生成样例。

### MiniGPT 阶段目录目标

闭环完成后再整理为类似结构，不要求现在一次性重构：

```text
MiniGPT/
├── config.py
├── data/
│   └── input.txt
├── dataset.py
├── train.py
├── inference.py
├── model/
├── tokenizer/
├── tests/
└── checkpoints/       # 由 .gitignore 忽略
```

建议补充 `MiniGPT/README.md`，记录运行方式、数据流、shape 表、超参数、loss 和生成结果。这会成为以后阅读真实 LLM 源码时的对照文档。

### 当前必须掌握的 shape

设：

- `B`：batch size；
- `T`：sequence length；
- `C`：`d_model`；
- `H`：head 数；
- `D = C / H`：每个 head 的维度；
- `V`：vocabulary size。

| 数据 | Shape | 每个维度的含义 |
| --- | --- | --- |
| token ids | `[B, T]` | B 个序列，每个序列 T 个离散 token id |
| embedding | `[B, T, C]` | 每个 token 被映射为 C 维向量 |
| Q/K/V 投影后 | `[B, T, C]` | 每个 token 的完整投影表示 |
| 拆分多头后 | `[B, H, T, D]` | H 个子空间分别处理长度 T 的序列 |
| attention score | `[B, H, T, T]` | 每个 head 中每个 query 对所有 key 的分数 |
| attention output | `[B, H, T, D]` | 每个 query 对 V 加权求和后的结果 |
| concat 后 | `[B, T, C]` | H 个 head 重新拼接为完整特征 |
| logits | `[B, T, V]` | 每个位置对词表中 V 个 token 的未归一化分数 |
| 展平 logits | `[B*T, V]` | 交叉熵将每个位置视为一个分类样本 |
| 展平 labels | `[B*T]` | 每个位置对应一个目标 token id |

要求不只是记住 shape，还要能说明每次 `view`、`transpose`、矩阵乘法为什么维度合法。

### 当前必须掌握的 PyTorch 工程机制

- `nn.Module.__setattr__` 如何注册子模块和参数；
- 为什么普通 Python `list` 不能替代 `nn.ModuleList` 管理 Block；
- `model.parameters()` 如何递归找到 Embedding、Attention、MLP 和 LM Head 参数；
- optimizer 保存的是哪些 Parameter 引用与优化状态；
- `state_dict` 为什么同时包含参数和 persistent buffer；
- causal mask 为什么用 `register_buffer`，而不是 `nn.Parameter`；
- `zero_grad()`、`backward()` 和 `step()` 分别改变什么。

### Attention 梯度主路径

先掌握主干，不要求一次推完 softmax Jacobian：

```text
loss
  ↓
logits = hidden @ W_lm^T + b_lm
  ↓
最后一个 Block 输出
  ↓
residual 分成恒等路径与子层路径
  ↓
W_o → attention @ V
           ↓          ↓
        softmax      V = XW_v
           ↓
        QK^T / √D
        ↓       ↓
     Q = XW_q  K = XW_k
```

反向传播时，每个分支都会依据链式法则贡献梯度；Residual 的加法会把上游梯度同时传给恒等路径和子层路径。最终 `W_q.grad`、`W_k.grad`、`W_v.grad` 被 optimizer 使用，`optimizer.step()` 才真正修改参数值。

### 验收

- loss 能稳定下降；
- 加载 checkpoint 后能够继续训练和生成；
- 能解释梯度如何传到 Embedding、Q/K/V 和 FFN；
- 能从 `[B, T]` 连续推导到 `[B*T, V]`；
- 能说明 `ModuleList`、Parameter、buffer、`state_dict` 和 optimizer 的关系；
- 能解释 `train()`、`eval()` 和 causal attention 的作用。

### 进入 ViT 前的阶段门槛

以下项目全部满足后，才进入第 6 节：

- [ ] `MiniGPT/train.py` 能从真实文本开始独立运行；
- [ ] 单 batch 过拟合测试通过；
- [ ] 修改未来 token 不会改变此前位置的 logits；
- [ ] train/validation loss 都能输出；
- [ ] checkpoint 在新进程中可加载；
- [ ] prompt 可以生成多个新 token；
- [ ] README 能解释完整数据流与关键 shape；
- [ ] 没有依赖根目录旧版 `model.py` 或 Attention 草稿。

## 6. 阶段一：Transformer 与视觉基础（第 3～5 周）

### 6.1 Attention Is All You Need

#### 重点阅读

- Abstract 与 Introduction：Transformer 要解决什么问题；
- Model Architecture：Encoder、Decoder 和残差结构；
- Attention：Scaled Dot-Product 与 Multi-Head Attention；
- Position-wise FFN 与 Positional Encoding；
- 训练部分只需先理解优化器、warmup 和正则化的作用。

#### 必须回答

- Attention 为什么比 RNN 更容易并行？
- `QKᵀ / √d` 为什么需要缩放？
- 多头注意力与单头注意力相比增加了什么表达能力？
- Encoder self-attention、Decoder causal attention 和 cross-attention 有何区别？

#### 小实验

- [ ] 打印每一步的 tensor shape；
- [ ] 可视化一张 attention map；
- [ ] 去掉缩放或 mask，观察训练和输出变化；
- [ ] 将自己的实现与 PyTorch `MultiheadAttention` 对齐。

### 6.2 An Image is Worth 16x16 Words（ViT）

#### 重点阅读

- 图像如何切成 patch；
- Patch Embedding 与卷积实现的等价关系；
- class token 与 position embedding；
- Transformer Encoder 如何处理视觉 token；
- 分辨率、patch size 与 token 数量的关系。

#### 必须回答

```text
224 × 224 image
    ↓ patch size = 16
14 × 14 patches
    ↓
196 visual tokens
```

- patch 为什么可以当作 token？
- patch size 变小时，计算量为什么会快速增加？
- ViT 与 CNN 的归纳偏置有什么不同？

#### 小实验

- [ ] 用 `unfold` 或卷积实现 Patch Embedding；
- [ ] 在 CIFAR-10 上训练 TinyViT；
- [ ] 对比两个 patch size 的精度、速度和 token 数；
- [ ] 可视化 position embedding 或 attention。

### 6.3 CLIP、BLIP-2 与 LLaVA

#### CLIP

重点理解：

- image encoder 与 text encoder；
- 对比学习与相似度矩阵；
- 正负样本和 temperature；
- zero-shot classification。

小实验：使用预训练 CLIP 对一组图片做 zero-shot 分类，并输出相似度矩阵。

#### BLIP-2

重点理解：

- 冻结视觉编码器和 LLM 的意义；
- Q-Former 如何连接视觉特征与语言模型；
- 为什么需要一个信息瓶颈，而不是把所有视觉 token 直接送入 LLM。

#### LLaVA

重点理解：

```text
Image
  ↓
Vision Encoder
  ↓
Projector
  ↓
Visual Tokens + Text Tokens
  ↓
LLM
```

- Projector 如何对齐视觉和语言的 hidden size；
- visual tokens 如何插入文本序列；
- 预训练对齐与 instruction tuning 分别解决什么问题。

小实验：打印一个 LLaVA 样例中 image token、text token 和 labels 的排列方式。

### 阶段产物

- [ ] 一张 Transformer → ViT → CLIP/BLIP-2 → LLaVA 的演化图；
- [ ] 一个 TinyViT 训练实验；
- [ ] 一个 CLIP zero-shot 实验；
- [ ] 一份“图像如何进入 LLM”的说明文档。

## 7. 阶段二：理解 Qwen2.5-VL（第 6～8 周）

### 7.1 先读论文和模型文档

不要一开始逐行看源码。先画出概念数据流：

```text
Image / Video
     ↓
Dynamic Resolution + Preprocessing
     ↓
Vision Transformer
     ↓
Visual Token Merger
     ↓
Visual Embeddings 插入文本序列
     ↓
Qwen Language Model
     ↓
Autoregressive Output
```

重点关注：

- 原生动态分辨率如何影响 grid 和视觉 token 数；
- Patch Embedding 如何处理图像与视频；
- Window Attention 与少量 Full Attention 如何组合；
- Visual Merger 如何压缩并映射视觉特征；
- 多模态位置编码如何表示文本、图像和视频位置；
- 图像占位 token 如何被真实视觉 embedding 替换；
- 文档、定位、视频任务分别需要什么输入输出格式。

> 画图时以论文和所用代码版本为准。Qwen2.5-VL 的连接模块应按实际实现中的 Visual Merger 理解，不要先验地假设一定存在 LLaVA 式独立线性 Projector。

### 7.2 再读预处理代码

建议顺序：

1. 配置类：记录 vision/text config 和关键维度；
2. image/video processor：追踪 resize、normalize、patch/grid 信息；
3. processor：追踪模板、图像占位符和 tokenizer 输出；
4. collator：理解 batch、padding、labels 和 image grid；
5. 用一个样例打印全部输入字段、dtype、device 和 shape。

### 7.3 最后读模型代码

不要从训练 `main` 入口开始。沿 forward 数据流阅读：

1. `Qwen2_5_VLForConditionalGeneration` 的输入和输出；
2. 语言模型主体；
3. Vision Transformer 的 Patch Embed 与 Block；
4. Window/Full Attention；
5. Visual Merger；
6. visual embedding 替换 image token 的位置；
7. position id / multimodal RoPE；
8. loss 与 generation 路径。

在 Hugging Face 版本中，优先查找类似以下文件，具体名称以安装版本为准：

- `configuration_qwen2_5_vl.py`；
- `image_processing_qwen2_5_vl.py`；
- `processing_qwen2_5_vl.py`；
- `modeling_qwen2_5_vl.py`。

### 源码阅读方法

为每个模块建立一张表：

| 项目 | 需要记录的内容 |
| --- | --- |
| 输入 | 参数名、shape、dtype、device |
| 输出 | shape 与语义 |
| 参数 | 可训练参数及其维度 |
| 关键操作 | reshape、merge、attention、mask |
| 上游/下游 | 谁调用它，它又调用谁 |
| 与论文对应 | 对应哪张图或哪一节 |
| 疑问 | 尚未验证的猜测 |

### 小实验

- [ ] 对不同尺寸图片打印视觉 grid 与 token 数；
- [ ] 注册 forward hook，记录 Vision Block 和 Merger 输出 shape；
- [ ] 比较纯文本、单图和多图输入的序列长度；
- [ ] 冻结所有参数，只验证一次 forward/backward 的梯度路径；
- [ ] 画出一个样例的端到端调用图。

### 验收

- 不看源码也能画出完整模型数据流；
- 能定位视觉特征进入语言模型的具体代码；
- 能解释视觉 token 数如何随输入分辨率变化；
- 能说清哪些参数属于 vision、merger 和 language model；
- 能修改一个非关键配置并预测其 shape/显存影响。

## 8. 阶段三：建立可信的微调基线（第 9～10 周）

### 目标

微调不是最终创新，但它是后续架构实验的控制组和训练基础。

### 任务

- [ ] 先选择较小模型和小数据集打通流程；
- [ ] 明确 conversation template 与图像占位符；
- [ ] 检查 labels，只在目标回答部分计算 loss；
- [ ] 先冻结 vision encoder，只训练 LoRA 或少量连接层；
- [ ] 记录 trainable parameters、显存和训练速度；
- [ ] 保存训练配置、随机种子、数据划分和 checkpoint；
- [ ] 在固定 validation/test 集上比较微调前后结果；
- [ ] 保存失败样例，而不只展示成功案例。

### 推荐递进

```text
官方模型零样本评估
      ↓
单 batch 过拟合
      ↓
小数据集 LoRA/SFT
      ↓
扩大数据量
      ↓
视需要解冻 merger 或 vision 后层
```

### 必须理解

- LoRA 的 rank、alpha、dropout 和 target modules；
- SFT 中 padding token 与 label mask；
- 冻结、部分解冻和全参数微调的差异；
- 梯度累计、混合精度与 gradient checkpointing；
- 数据质量与模型结构对结果的不同影响。

### 验收

- [ ] 一条命令可复现实验；
- [ ] 训练集、验证集和测试集严格分离；
- [ ] 有微调前 baseline；
- [ ] 有定量指标和错误案例；
- [ ] 能确认提升不是由数据泄漏或 prompt 差异造成。

## 9. 阶段四：研究相关模型（第 11～12 周）

### 阅读范围

围绕“文档、OCR、布局和视觉 token 压缩”研究：

- LLaVA 系列；
- InternVL 系列；
- GOT-OCR；
- MinerU；
- PaddleOCR-VL；
- DeepSeek-OCR；
- olmOCR 或实际要比较的对应项目。

> 模型名称、版本、发布日期和架构描述必须回到论文与官方仓库核对，不能只引用二手总结。相近名称的项目尤其容易混淆。

### 不要只问“谁的分数更高”

对每个项目比较：

| 维度 | 核心问题 |
| --- | --- |
| 任务 | 通用 VQA、OCR、文档解析还是布局恢复？ |
| 视觉编码器 | 原生 ViT、CLIP、SAM 风格骨干还是专用编码器？ |
| 分辨率 | 固定、动态、切片还是多视图？ |
| Token 压缩 | pooling、merger、resampler、query 还是其他方法？ |
| 对齐方式 | Linear/MLP、cross-attention、Q-Former 或其他结构？ |
| 数据 | 数据来源、规模、合成方式和标注质量？ |
| 训练策略 | 预训练、SFT、分阶段训练或联合训练？ |
| 指标 | OCR、表格、阅读顺序、定位分别如何评估？ |
| 代价 | 参数、视觉 token、显存、速度和训练成本？ |

### 阶段产物

- [ ] 一张至少包含 5 个模型的横向对比表；
- [ ] 每个候选改进对应一个明确痛点；
- [ ] 选出一个主方案和一个低成本备用方案；
- [ ] 写出“为什么现有 Qwen2.5-VL baseline 在该任务上不足”的证据。

## 10. 阶段五：设计并实现创新（第 13～16 周）

### 先定义研究问题

一个合格的问题应类似：

> 在复杂文档解析中，Qwen2.5-VL 对小文字或跨区域阅读顺序的建模不足。加入轻量布局特征后，能否在视觉 token 增幅可控的情况下提高结构化解析指标？

而不是：

> 我加入了 SAM、OCR 和 Cross-Attention，所以模型更先进。

### 候选方案与优先级

#### 方案 A：区域裁剪或 SAM 辅助预处理

```text
Image → Region/Mask → Crop or Multi-view → Qwen2.5-VL
```

- 实现难度：低；
- 适合：先验证“重点区域是否有帮助”；
- 风险：提升可能来自更高分辨率或更多 token，而非 SAM 本身。

必须设置相同 token 预算下的 random crop、规则 crop 或原图 baseline。

#### 方案 B：Layout Encoder（推荐主线）

```text
Image → Vision Encoder ─────────┐
                               ├→ Fusion → LLM
Layout/Box → Layout Encoder ───┘
```

- 实现难度：中；
- 适合：表格、PPT、论文、票据和阅读顺序；
- 优点：研究问题容易定义，消融也较清晰；
- 风险：布局标注、检测误差和额外计算成本。

第一版应从 box 坐标 embedding + 小型融合层开始，不要直接设计大型网络。

#### 方案 C：视觉特征融合

```text
ViT Feature + OCR/SAM Feature → Gated/Cross Attention → Merger → LLM
```

- 实现难度：中高；
- 风险：不同特征的分辨率、维度、位置和训练稳定性较难处理。

只有单一路径 baseline 已稳定时再尝试。

#### 方案 D：新型 Visual Merger/Projector

```text
Vision Feature → Resampler/Cross Attention/MoE → LLM Space
```

- 实现难度：高；
- 研究价值：视觉信息压缩与 token 效率；
- 风险：训练成本高，MoE 还会引入路由和负载均衡问题。

建议先做小型 cross-attention resampler，不把 MoE 作为第一版。

### 最小实现顺序

1. 冻结原模型，单独验证新模块的输入输出和梯度；
2. 用随机 tensor 完成 shape/unit test；
3. 用 10～100 个样本过拟合；
4. 在小训练集上与 baseline 对比；
5. 确认有效后再扩大数据和解冻范围；
6. 最后分析速度、显存和 token 数代价。

## 11. 实验设计与论文可信度

### 必须有的实验组

| 实验 | 目的 |
| --- | --- |
| 原始模型 zero-shot | 确定初始能力 |
| 原始模型 + 相同数据微调 | 核心 baseline |
| 新模块 + 相同数据与训练配置 | 判断结构是否有效 |
| 去掉新模块某部分 | 消融模块贡献 |
| 参数量或 token 数相近的控制组 | 排除单纯增加容量/计算量 |

### 控制变量

尽量保持一致：

- 数据与划分；
- prompt/template；
- 随机种子；
- optimizer、学习率和训练步数；
- 图像分辨率和视觉 token 预算；
- 解码策略；
- 评价脚本。

### 指标

根据任务选择，不要只使用 loss：

- OCR：CER、WER、Edit Distance；
- 文档问答：Exact Match、ANLS 或任务官方指标；
- 表格：TEDS 或结构相关指标；
- 定位：IoU、mAP 或坐标准确率；
- 生成：任务指标 + 人工错误分类；
- 效率：参数量、视觉 token 数、显存、吞吐量和延迟。

### 错误分析

至少分类记录：

- 小文字；
- 密集排版；
- 阅读顺序；
- 表格结构；
- 图文混排；
- 幻觉或漏识别；
- 输入过长导致的截断。

## 12. 每周论文阅读模板

每篇论文只先回答这些问题：

```text
论文解决什么具体问题？
此前方法的瓶颈是什么？
输入如何经过每个模块变成输出？
真正新增的模块或训练目标是什么？
与最接近的 baseline 相比改了什么？
最关键的消融实验是什么？
提升是否可能来自更多数据或计算量？
哪些结论可以迁移到我的项目？
```

每篇论文的最小产物：

- 一张手绘或电子架构图；
- 一张输入输出 shape 表；
- 三个关键结论；
- 两个仍未解决的问题；
- 一个可以用代码验证的小实验。

## 13. 每周源码阅读模板

不要按文件从第一行读到最后一行。按一次真实 forward 追踪：

```text
样例输入
  ↓
Processor / Collator
  ↓
Model.forward
  ↓
Vision Encoder
  ↓
Merger / Alignment
  ↓
LLM
  ↓
Loss or Generate
```

阅读工具与方法：

- 用 `rg` 搜类名和调用位置；
- 用 IDE 跳转定义和查找引用；
- 用断点或 forward hook 看真实 shape；
- 用小输入单步运行，避免直接启动完整训练；
- 记录调用图，不依赖记忆；
- 每次只解决一个问题。

## 14. 并行补充的机器学习基础

每周约 20% 时间用于基础，不中断主项目。

### 数学

- 线性代数：矩阵乘法、投影、特征值、SVD；
- 概率：最大似然、交叉熵、KL 散度；
- 微积分：链式法则、梯度、Jacobian；
- 优化：SGD、AdamW、weight decay、学习率调度。

### 深度学习

- CNN、BatchNorm、LayerNorm、RMSNorm；
- 数据增强、过拟合和正则化；
- RoPE、SwiGLU、GQA、KV Cache；
- mixed precision、gradient accumulation、checkpointing。

### PyTorch 工程

- Dataset、Sampler、Collator；
- `state_dict` 与 checkpoint；
- hook、autograd 和梯度检查；
- profiling、显存分析和复现性；
- pytest、配置管理和实验日志。

## 15. 每周节奏与复盘

建议分配：

- 50%：代码与实验；
- 25%：论文与课程；
- 15%：源码阅读；
- 10%：记录与复盘。

每周必须回答：

1. 本周完成了什么可运行或可展示的产物？
2. 哪些结论由实验支持？
3. 当前最大的不确定性是什么？
4. 下周要验证的唯一核心问题是什么？
5. 哪些新想法应该暂时放入 backlog？

## 16. 当前最重要的下一步

当前不要直接改 Qwen2.5-VL，也不要同时学习所有 OCR 模型。

唯一优先级是：

> 把 MiniGPT 从“模块和训练骨架已经存在”推进到“能够使用真实文本训练、验证、保存、加载和生成”，同时整理 tensor shape、参数注册与梯度流笔记。

### 第一次辅导：接通真实训练数据

目标：修复当前训练入口，使一批真实文本能完成 forward、loss 和 backward。

1. 解释 `TextDataset.__getitem__` 中 `x`、`y` 的关系；
2. 准备一份足够长、可提交的小型文本语料；
3. 修复 `ToyDataset` 与 `TextDataset` 的接口不一致；
4. 将 tokenizer 的 `vocab_size` 接入 config；
5. 打印一个 batch 的 `x.shape`、`y.shape`、`logits.shape`；
6. 只训练一个 batch，确认 loss 明显下降；
7. 检查 `W_q.grad`、Embedding gradient 是否存在且为有限值。

验收结果：一个真实文本 batch 可以被模型反复拟合。

### 第二次辅导：验证 causal attention 与完整训练

目标：证明 mask 正确，而不是只确认代码能运行。

1. 画出一个长度为 4 的 causal mask；
2. 解释 score `[B, H, T, T]` 的两个 `T`；
3. 测试位置 `t` 的输出不会受未来 token 改变影响；
4. 加入 train/validation split；
5. 记录并比较 train loss 与 validation loss。

验收结果：mask 有自动化测试，训练和验证 loss 可以被独立统计。

### 第三次辅导：参数注册与 Attention 反向传播

目标：把 PyTorch 工程机制与数学计算图连接起来。

1. 列出 `named_parameters()` 与 `state_dict()`；
2. 区分 Parameter、Module、buffer 和普通 Tensor；
3. 给 Wq/Wk/Wv 注册 hook 或直接查看 `.grad`；
4. 沿 loss → lm_head → Block → Attention 解释梯度路径；
5. 比较 `zero_grad()` 前后以及 `step()` 前后的数值变化。

验收结果：能够用源码级语言解释 optimizer 为什么能更新所有 Block。

### 第四次辅导：checkpoint 与生成

目标：完成从训练到使用模型的闭环。

1. 保存 model、optimizer、epoch、config 和 tokenizer 信息；
2. 在新进程加载 checkpoint；
3. 实现逐 token greedy generation；
4. 再增加 temperature 与 top-k；
5. 限制上下文到 `max_seq_len`；
6. 验证相同随机种子下的可复现性。

验收结果：输入 prompt 后能生成文本，并能从 checkpoint 恢复训练。

完成后立即进入 ViT 小实验。这样到阅读 Qwen2.5-VL 时，看到的就不再是几千行陌生代码，而是一组已经理解过、只是规模更大且组合更复杂的模块。
