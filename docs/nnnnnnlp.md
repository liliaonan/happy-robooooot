## 搭建一个T
### 1、Embedding层
Embedding 层其实是一个存储固定大小的词典的嵌入向量查找表
在输入神经网络之前，我们往往会先让自然语言输入通过分词器 tokenizer，分词器的作用是把自然语言输入切分成 token 并转化成一个固定的 index，此index对应词表中向量index，对应相近语义
词表大小则往往高达数万数十万
```
- 文本 → tokenizer→ token
- token → Embedding → 词表index，语义向量
```

如果我们将词表大小设为 4，输入“我喜欢你”，那么，分词器可以将输入转化成：
```
input: 文本
output: 词表index

input: 我
output: 0

input: 喜欢
output: 1

input：你
output: 2
```

Embedding 内部其实是一个可训练的（vocab_size，embedding_dim）的权重矩阵。词表里的每一个值，都对应一行维度为 embedding_dim 的向量。
```
vocab_size 即为token 数量，
向量维度 embedding_dim 
输出：（batch_size，seq_len，embedding_dim）批处理的数量，自然语言序列的长度，词表尺寸
```

- Embedding 本身是可学习层，
  - 以随机值初始化权重矩阵（正态分布/均匀分布）
  - 和transformer、预测头一起训练
直接使用 torch 中的 Embedding 层：
```
self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
```
- 加载预训练权重+可选冻结
```
# 1. 先加载预训练的嵌入权重（假设pretrained_embeds是从预训练模型中读取的权重，形状[30522, 768]） 
pretrained_embeds = torch.randn(args.vocab_size, args.dim)  

# 模拟预训练权重# 2. 初始化嵌入层并加载预训练权重 
model.tok_embeddings = nn.Embedding.from_pretrained(pretrained_embeds,freeze=False  # False=继续训练，True=冻结权重不训练)

# 验证：查看是否可训练
print(f"嵌入层是否可训练

```



### 2、位置编码

注意力机制可以实现良好的并行计算，但同时，注意力机制也导致位置信息的丢失。在LSTM、RNN中，输出序列均按照文本顺序依次递归处理。

## 预训练语言模型

### BERT
BERT，全名为 Bidirectional Encoder Representations from Transformers 双向编码器表征
是由 Google 团队在 2018年发布的预训练语言模型。
Transformer 架构
预训练+微调范式

#### 模型架构 - Encoder only

延续：多个encoder堆积

改进：
BERT 是针对于 NLU 任务打造的预训练模型，其输入一般是文本序列，而输出一般是 Label，例如情感分类的积极、消极 Label。
```
NLU Natural Language Understanding 自然语言理解 
核心目标：让机器真正理解人类自然语言的语义、意图、逻辑和上下文含义                 
文本 → 机器理解

NLG Natural Language Generation 自然语言生成
文本 → 机器 文本输出
```

但是，正如 Transformer 是一个 Seq2Seq 模型，使用 Encoder 堆叠而成的 BERT 本质上也是一个 Seq2Seq 模型，只是没有加入对特定任务的 Decoder，

因此，为适配各种 NLU 任务，在模型的最顶层加入了一个分类头 prediction_heads，用于将多维度的隐藏状态通过线性层转换到分类维度（例如，如果一共有两个类别，prediction_heads 输出的就是两维向量）。



Encoder 块中是堆叠起来的 N 层 Encoder Layer，BERT 有两种规模的模型：
base 版本（12层 Encoder Layer，768 的隐藏层维度，总参数量 110M）
large 版本（24层 Encoder Layer，1024 的隐藏层维度，总参数量 340M）
```
在 BERT（以及所有 Transformer 架构）里，“隐藏层维度”就是：
每个 token 经过 Encoder Layer 之后，得到的向量长度——固定为 768 维（base）或 1024 维（large）
隐藏层维度 = 模型内部“词向量”的宽度，也是多头注意力里 Q/K/V 的维度，也是Feed-Forward 的输入/输出维度（FFN 中间会先放大再缩回）
```

##### 创新
每个encoder层中，immediate层，BERT特殊称呼，位于hidden_states输出之前

结构如下：
<div align="center">
  <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/3-figures/1-3.png" alt="图片描述" width="40%"/>
  <p>图3.5 Intermediate 结构</p>
</div>

1、激活函数使用的是GELU, 自 BERT 才开始被普遍关注的激活函数
$$GELU(x) = 0.5x(1 + tanh(\sqrt{\frac{2}{\pi}})(x + 0.044715x^3))$$

核心思想：引入随机正则的思想，根据输入自身的概率分布，决定此神经元是否丢弃或保留

2、BERT 将相对位置编码融合在了注意力机制中，将相对位置编码同样视为可训练的权重参数,完成注意力分数的计算之后，先通过 Position Embedding 层来融入相对位置信息：
Transformer 使用的绝对位置编码 Sinusoidal 能够拟合更丰富的相对位置信息，
但是，这样也增加了不少模型参数，同时完全无法处理超过模型训练长度的输入（例如，对 BERT 而言能处理的最大上下文长度是 512 个 token）

#### 预训练任务-MLM+NSP
MLM masked language model 掩码语言模型

LM 预训练的缺陷是拟合从左到右的语义，但忽略了双向语义（上下文）。
虽然 Transformer 中通过位置编码表征了文本序列中的位置信息，但这和直接拟合双向语义关系还是有本质区别。例如，BiLSTM（双向 LSTM 模型）在语义表征上就往往优于 LSTM 模型，就是因为 BiLSTM 通过双向的 LSTM 拟合了双向语义关系。

MLM 相较于模拟人类写作的 LM，MLM 模拟的是“完形填空”。通过mask一些token，让模型根据token上下文预测token,既利用了海量的数据进行无监督学习，同时强化了双向语义理解。

计算过程：
1. 随机把输入句子中的部分 token 换成 [MASK]；
2. 让模型只凭上下文去预测这些被盖住的 token 本身；
3. 用“原 token”作为真值计算交叉熵损失，判断预测是否准确。

```
输入：I <MASK> you because you are <MASK>
输出：<MASK> - love; <MASK> - wonderful
```

这种方式无需人为标注，只需对文本进行随机掩码，因此也可以利用互联网所有文本语料实现预训练。例如，BERT 的预训练就使用了足足 3300M 单词的语料。
“3300M 单词”里的 M 是“million”（百万）的意思，3300M = 3.3 billion 个 word piece（BERT 用的 WordPiece 子词）。
它纯粹是计数单位，跟“字节/存储大小”无关。token可以是子词，但token不一定是子词类型。

引入的策略： MLM 训练时，会随机选择训练语料中 15% 的 token 用于遮蔽；
但是这 15% 的 token 并非全部被遮蔽为 <MASK>，而是有 80% 的概率被遮蔽，10% 的概率被替换为任意一个 token，还有 10% 的概率保持不变

MLM缺陷：
LM 其训练和下游任务是完全一致的，也就是说，训练时是根据上文预测下文，下游任务微调和推理时也同样如此
MLM 缺陷：在下游任务微调和推理时，其实是不存在我们人工加入的 <MASK> 的，我们会直接通过原文本得到对应的隐藏状态再根据下游任务进入分类器或其他组件
预训练和微调的不一致，会极大程度影响模型在下游任务微调的性能


**NSP next sentence prediction 下一句预测**

针对句级的NLU，同样，也可以从海量数据中选取连续的句子作为正样本进行无监督训练。例如问答匹配、自然语言推理等。

问答匹配是指，输入一个问题和若干个回答，要求模型找出问题的真正回答；

自然语言推理是指，输入一个前提和一个推理，判断推理是否是符合前提的。这样的任务都需要模型在句级去拟合关系，判断两个句子之间的关系，而不仅是 MLM 在 token 级拟合的语义关系。因此，BERT 提出了 NSP 任务来训练模型在句级的语义关系拟合。

NSP 任务的核心思路是要求模型判断一个句对的两个句子是否是连续的上下文。例如，输入和输入可以是：
```
    输入：
        Sentence A：I love you.
        Sentence B: Because you are wonderful.
    输出：
        1（是连续上下文）

    输入：
        Sentence A：I love you.
        Sentence B: Because today's dinner is so nice.
    输出：
        0（不是连续上下文）
```
同样，也可以从海量数据中选取连续的句子作为正样本进行无监督训练。


**问答匹配：**
 1 个问题（Question） 和 N 个候选回答（Candidate Answers）
```
 N 个候选回答（Candidate Answers）来源
结构化知识库（智能座舱首选）
构建方式
1、人工梳理高频问题：整理座舱用户的常见问题（Q）和标准答案（A），形成一对一 / 一对多的 QA 对
    问题（Q）	标准回答（A）	所属模块
    怎么开自动空调？	按下空调面板 AUTO 键，或语音说 “打开自动空调”	空调控制
    座椅加热怎么调最高？	长按座椅加热键 3 秒，或语音说 “座椅加热调最高”	座椅控制
2、结构化存储：将 QA 对存入数据库（如 MySQL、Redis），并按功能模块、关键词做索引，方便快速检索。
候选回答筛选逻辑
用户提问后，先通过关键词匹配 / 语义检索从知识库中筛选出与问题相关的 QA 对，得到候选回答列表：
例：用户问 “自动空调怎么开启？”→ 检索到含 “自动空调”“开启” 关键词的 QA 对 → 候选回答只有 1 个（精准匹配）；
例：用户问 “空调怎么调？”→ 检索到含 “空调” 的所有 QA 对 → 候选回答列表包含 “开自动空调”“调温度”“调风向” 等多个回答。
优势：回答准确率高、可控性强，完全适配车载场景的标准化需求；
缺点：覆盖范围有限，无法应对小众 / 个性化问题。

2. 非结构化文档抽取（补充来源）
针对知识库未覆盖的问题，候选回答可从非结构化文档中抽取，适合解决复杂、长尾的问题（如车载系统故障排查、功能说明等）。
文档来源：车载说明书、故障排查手册、功能白皮书等文本；
实现方式
文档预处理：将文档拆分为段落 / 句子级别的文本片段，每个片段作为一个 “候选回答单元”；
语义索引构建：用 BERT 等模型将每个文本片段转化为语义向量，存入向量数据库（如 FAISS、Milvus）；
候选筛选：用户提问后，将问题也转化为语义向量，在向量数据库中做相似度检索，取相似度最高的 Top-N 个文本片段作为候选回答。
示例：用户问 “空调报错 E1 怎么办？”→ 问题向量与故障手册中的 “E1 故障对应传感器异常，重启空调即可” 片段向量相似度最高 → 该片段成为候选回答。
优势：覆盖长尾问题，无需人工逐条梳理 QA 对；
缺点：回答质量依赖文档质量，可能存在冗余信息。

3. 动态生成（进阶方案，结合生成式模型）
在部分高端座舱场景中，候选回答可由大模型动态生成，再结合匹配模型做筛选，适合应对开放性问题。

实现方式
用车载轻量化大模型（如 Llama-2 量化版、Qwen-7B-int4），基于用户问题生成多个候选回答；
用问答匹配模型对生成的候选回答做质量打分，筛选出与问题最匹配、最符合车载规范的回答。
示例：用户问 “怎么快速降温？”→ 大模型生成 3 个候选回答 → 匹配模型打分后，选 “开窗通风 1 分钟后关闭，再开空调制冷模式” 作为最优回答。
优势：覆盖开放性问题，灵活性高；
缺点：对车载算力要求高，需做严格的输出控制（避免生成违规内容）。

4. 用户行为日志挖掘（冷启动 / 迭代优化）
这是补充和优化知识库的来源，通过分析用户的历史对话日志，挖掘新的问题和对应的优质回答。
实现方式
收集座舱用户的真实对话（如用户问 “空调怎么除雾？”，人工客服回复 “开前挡风除雾键”）；
定期整理这些日志，将新的 QA 对加入结构化知识库，扩充候选回答的覆盖范围。

```

**自然语言推理**
 1 个前提（Premise, P） 和 1 个假设（Hypothesis, H），模型需要判断两者的逻辑关系，通常分为三类：

 ```
关系类型	定义	示例
蕴含（Entailment）	假设 H 的内容完全由前提 P 推导得出	P：座舱温度已调至 25℃；H：当前车内温度是 25℃
矛盾（Contradiction）	假设 H 的内容与前提 P 完全相反	P：座舱温度已调至 25℃；H：当前车内温度是 30℃
中立（Neutral）	假设 H 的内容与前提 P 无必然逻辑关系	P：座舱温度已调至 25℃；H：座椅加热已开启

智能座舱场景示例：
前提 P：用户设置座舱温度为 22℃，且当前车内温度为 25℃
假设 H1：空调会自动启动制冷模式 → 蕴含关系
假设 H2：空调会自动启动制热模式 → 矛盾关系
假设 H3：氛围灯颜色为蓝色 → 中立关系

```

NSP 训练让模型能区分 “连贯句对” 和 “不连贯句对”；
NLI 任务则要求模型在 “连贯” 的基础上，进一步区分 **“蕴含”“矛盾”“中立”** 三种**更细粒度的逻辑关系**

输入序列（以 H1 为例）：
[CLS] 用户 设置 座舱 温度 为 22℃ ， 且 当前 车内 温度 为 25℃ [SEP] 空调 会 自动 启动 制冷 模式 [SEP]

（2）模型微调：三分类预测头
特征提取：同样取 BERT 输出的 [CLS] 向量 —— 该向量聚合了前提和假设的句间逻辑关系。
预测头设计：在 [CLS] 向量后接全连接层 + Softmax，输出 3 分类概率（蕴含、矛盾、中立的概率）。
分类头的损失函数：交叉熵损失（Cross-Entropy Loss）。

（3）输出结果
输出假设相对于前提的逻辑关系标签，以及对应的概率值。
示例输出：H1 → 蕴含（概率 0.98）

### 编码器和解码器核心作用

编码器：核心任务是对输入序列进行深度的语义理解和特征提取，生成包含输入序列全部上下文信息的“语义表示”（也叫上下文向量/向量表示）。

“阅读理解专家”：
1、输入处理：接收原始文本序列（一句话或者一段文字），通过embedding, 将离散的符号（文本）转化为连续的向量；再通过位置编码（Positional encoding）保留语序信息；
2、上下文建模：通过多层自注意力（Self-attension）机制，让输入位置的词能“看到”其他全部位置的词，捕捉长距离依赖和上下文关系（如，知道此处的“他”指代的内容）；
```
范围上：上下文关系是 “全集”，长距离依赖是 “子集”；
关注点上：上下文关系关注 **关联的存在与性质**，长距离依赖关注 “关联的距离与捕捉能力”；
建模意义上：解决长距离依赖问题，是提升模型捕捉复杂上下文关系能力的关键。
```
3、输出：最终输出一个维度固定的向量表示，代表了完整的输入序列语义，供解码器使用。

典型应用场景：
仅用编码器的模型（如 BERT）：做文本分类、命名实体识别、语义相似度计算等 “理解类” 任务。

解码器：根据编码器的语义表示，按顺序 生成符合语法和语义的目标序列。是一个 生成式 的组件。

“写作专家”：
1、自回归生成：解码器是逐 token(字/词) 生成的，每一步生成都依赖之前的输出。
2、掩码注意力（Masked-self-attention）:确保模型只看到到输入和当前生成的token，避免提前看到未来的token;
3、输出转换：通过全连接层将隐藏状态映射至目标词汇表，最终输出概率最高的token.

典型应用场景：
编码器 + 解码器的模型（如 GPT-1/2 早期版本、T5、机器翻译模型）：做机器翻译、文本摘要、对话生成等 “生成类” 任务。
仅用解码器的模型（如 GPT 系列）：通过自回归生成完成聊天、文本创作等任务（此时解码器的输入是前文，无需编码器）。

```
输入（英文）：I love you
└── 编码器：提取语义 → [I, love, you] 的上下文向量
    └── 解码器：
        第一步：基于编码器输出，生成第一个词（中文）→ 我
        第二步：基于“我”+ 编码器输出，生成第二个词 → 爱
        第三步：基于“我爱”+ 编码器输出，生成第三个词 → 你
最终输出（中文）：我爱你

```
         
## Decoder-Only PLM ChatGPT

Decoder-Only 就是目前大火的 LLM 的基础架构，目前所有的 LLM 基本都是 Decoder-Only 模型（RWKV、Mamba 等非 Transformer 架构除外）          
开源 LLM 基本架构的 LLaMA 模型，也正是在 GPT 的模型架构基础上优化发展而来

GPT，即 Generative Pre-Training Language Model，是由 OpenAI 团队于 2018年发布的预训练语言模型。
最终在 2020年发布的 GPT-3 成就了 LLM 时代的基础并以 GPT-3 为基座模型的 ChatGPT 成功打开新时代的大门，成为 LLM 时代的最强竞争者也是目前的最大赢家。

### 模型架构
由于不存在 Encoder 的编码结果，Decoder 层中的掩码注意力也是自注意力计算。也就是对一个输入的 hidden_states，会通过三个   参数矩阵来生成 query、key 和 value，而不再是像 Transformer 中的 Decoder 那样由 Encoder 输出作为 key 和 value。
后续的注意力计算过程则和 BERT 类似，只是在计算得到注意力权重之后，通过掩码矩阵来遮蔽了未来 token 的注意力权重，从而限制每一个 token 只能关注到它之前 token 的注意力，来实现掩码自注意力的计算。

### 预训练任务 CLM
Decoder-Only 的模型结构往往更适合于文本生成任务，因此，Decoder-Only 模型往往选择了**最传统也最直接**的预训练任务——因果语言模型，Casual Language Model，下简称 CLM。

CLM 可以看作 N-gram 语言模型的一个直接扩展             
N-gram 语言模型是基于前 N 个 token 来预测下一个 token，CLM 则是基于一个自然语言序列的前面所有 token 来预测下一个 token
CLM 是一个经典的补全形式
```
input: 今天天气
output: 今天天气很

input: 今天天气很
output：今天天气很好
```
BERT 之所以可以采用预训练+微调的范式取得重大突破，正是因为其选择的 MLM、NSP 可以在海量无监督语料上直接训练
CLM 是更直接的预训练任务，其天生和人类书写自然语言文本的习惯相契合，也和下游任务直接匹配

```
WHY
之所以“和下游更适配”，核心在于任务形态、数据分布、优化目标三者都与真实应用场景“无缝对齐”

1. 任务形态零差距
CLM 的每一步都是“已知左侧所有上下文，预测下一个 token”，这与 99% 的下游生成任务（对话、摘要、代码补全、问答）完全同构——推理时就是复现同一过程。
MLM 是“完形填空”，训练时只能利用双向上下文，推理时却常要求单向生成，造成train-inference mismatch；NSP 更是“两句话是否相邻”的二分类，与生成目标南辕北辙。

2. 数据分布一致性
CLM 直接优化真实文本的联合概率 p(s₁s₂…s_T)，模型学到的就是人类书写的顺序依赖、长尾分布、语义连贯性；
MLM 只建模被掩码位置的局部条件概率，对低频模式、长距离依赖的建模力度弱；NSP 的负样本是随机拼接，引入大量不真实句对，反而可能学到虚假特征。

3. 优化目标即评价指标
CLM 的交叉熵损失正是生成任务常用的困惑度（PPL）的分子，预训练目标与下游指标一一对应；
MLM 的收敛目标是对掩码位还原准确，与下游 BLEU/ROUGE 无直接数值映射，需额外微调。

4. 无需额外结构即可 zero-shot 推理
CLM 模型（GPT 系列）去掉微调也能通过 prompt 直接做“文本续写”，天然支持零样本/少样本场景；
MLM+NSP 模型（BERT 系列）必须加任务特定头 + 微调，否则无法输出合法文本序列。

CLM 把“预训练”做成了真实生成任务的彩排，而 MLM/NSP 只是“辅助填空”或“句间匹配”，下游真正要生成文本时还得把彩排过的剧本重新写一遍，适配度自然大打折扣。

**以上的几点也是后续学习重点关注内容：任务形态、数据分布一致性、优化目标、额外结构推理**

```

### GPT系列模型
参数

Hidden_size = 头数 × 单头维度

**大火的几个GPT系列模型参数**

| 模型 | Decoder Layer | Hidden_size | 注意力头数 | 注意力维度 | 总参数量 | 预训练语料 |
| :--- | :-----------: | :---------: | :--------: | :--------: | :------: | :--------- |
| GPT-1 | 12 | 768 | 12 | 64 | 117M | BookCorpus（约7000本电子书） |
| GPT-2（基础版） | 12 | 768 | 12 | 64 | 117M | WebText（约800万网页，40GB） |
| GPT-2（大版本） | 24 | 1536 | 24 | 64 | 355M | WebText |
| GPT-2（超大版本） | 48 | 1600 | 25 | 64 | 15B | WebText |
| GPT-3（Ada） | 12 | 1024 | 16 | 64 | 350M | Common Crawl+WebText2+Books1/2+Wikipedia（约570B tokens） |
| GPT-3（Babbage） | 24 | 2048 | 32 | 64 | 1.3B | 同上 |
| GPT-3（Curie） | 24 | 4096 | 64 | 64 | 6.7B | 同上 |
| GPT-3（Davinci） | 96 | 12288 | 192 | 64 | 175B | 同上 |
| GPT-3.5（text-davinci-003） | 96（推测） | 12288（推测） | 192（推测） | 64（推测） | 175B（微调） | 多源文本+代码+对话数据 |
| GPT-4（文本版，估算） | 120+ | 28672（推测） | 448（推测） | 64（推测） | 约500B | 多语种文本+图像文本对+代码 |


## 大语言模型

Large Language Model，LLM
LLM 使用与传统预训练语言模型相似的架构与预训练任务（如 Decoder-Only 架构与 CLM 预训练任务）：
广义的 LLM 一般覆盖了从十亿参数（如 Qwen-1.5B）到千亿参数（如 Grok-314B）的所有大型语言模型。
B Billion（十亿）

只要模型展现出涌现能力，即在一系列复杂任务上表现出远超传统预训练模型（如 BERT、T5）的能力与潜力，都可以称之为 LLM。

一般认为，GPT-3（1750亿参数）是 LLM 的开端，
基于 GPT-3 通过 预训练（Pretraining）、监督微调（Supervised Fine-Tuning，SFT）、强化学习与人类反馈（Reinforcement Learning with Human Feedback，RLHF）三阶段训练得到的 ChatGPT 更是主导了 LLM 时代的到来

### LLM的能力

#### 涌现能力（Emergent Abilities）
涌现能力的显现就像是模型性能随着规模增大而迅速提升，超过了随机水平，也就是我们常说的量变引起了质变

#### 上下文学习（In-context Learning）
起源：
上下文学习能力是由 GPT-3 首次引入的。

解释：
上下文学习是指允许语言模型在提供自然语言指令或多个任务示例的情况下，通过理解上下文并生成相应输出的方式来执行任务，而无需额外的训练或参数更新。

优势：
对传统 PLM，在经过高成本的预训练之后，往往还需要对指定的下游任务进行有监督微调。
有监督微调的训练数据的成本更高, 需要的训练样本数往往在 1k~数十k 不等,均需要进行人工标注.
具备上下文学习能力的 LLM 往往无需进行高成本的额外训练或微调，而可以通过少数示例或是调整自然语言指令，来处理绝大部分任务，从而大大节省了算力和数据成本。

影响：
上下文学习能力也正在引发 NLP 研究范式的变革。在传统 PLM 时代，解决 NLP 下游任务的一般范式是预训练-微调，即选用一个合适的预训练模型，针对自己的下游任务准备有监督数据来进行微调。

一般范式开始向 Prompt Engineering 也就是调整 Prompt 来激发 LLM 的能力转变。例如，目前绝大部分 NLP 任务，通过调整 Prompt 或提供 1~5 个自然语言示例，就可以令 GPT-4 达到超过传统 PLM 微调的效果。

```
**上下文学习的理解**

预训练任务+微调
VS
prompt engineering

```

#### 指令遵循（Instruction Following）
指令微调：通过使用自然语言描述的多任务数据进行微调
泛化能力：使用指令形式化描述的未见过的任务上表现良好

#### 逐步推理（Step by Step Reasoning）
NLP难点：涉及多个推理步骤的复杂推理任务
传统的 NLP 模型通常难以解决涉及多个推理步骤的复杂任务，例如数学问题
LLM 通过采用思维链（Chain-of-Thought，CoT）推理策略，可以利用包含中间推理步骤的提示机制来解决这些任务，从而得出最终答案。据推测，这种能力可能是通过对代码的训练获得的。

### LLM 的特点

#### 多语言支持
多语言、跨语言模型曾经是 NLP 的一个重要研究方向
但 LLM 由于需要使用到海量的语料进行预训练，训练语料往往本身就是多语言的，因此 LLM 天生即具有多语言、跨语言能力，只不过随着训练语料和指令微调的差异，在不同语言上的能力有所差异

由于英文高质量语料目前仍是占据大部分，以 GPT-4 为代表的绝大部分模型在英文上具有显著超越中文的能力
但针对中文进行额外训练和优化的国内模型（如文心一言、通义千问等）往往能够在中文环境上展现更优越的效果。

#### 长文本处理
由于能够处理多长的上下文文本，在一定程度上决定了模型的部分能力上限，LLM 往往比传统 PLM 更看重长文本处理能力。相对于以 512 token 为惯例的传统 PLM（如 BERT、T5等模型的最大上下文长度均为 512），LLM 在拓宽最大上下文长度方面可谓妙计频出。

由于在海量分布式训练集群上进行训练，LLM 往往在训练时就支持 4k、8k 甚至 32k 的上下文长度
LM 大部分采用了旋转位置编码（Rotary Positional Encoding，RoPE）（或者同样具有外推能力的 AliBi）作为位置编码，具有一定的长度外推能力，也就是在推理时能够处理显著长于训练长度的文本
例如，InternLM 在 32k 长度上下文上进行了预训练，但通过 RoPE 能够实现 200k 长度的上下文处理。

## 参数推导
架构级超参

```
from transformers import PretrainedConfig

class ModelConfig(PretrainedConfig):
    # 模型类型标识，符合transformers库的规范
    model_type = "Tiny-K"
    
    def __init__(
            self,
            # 基础维度配置
            dim: int = 768,                # 模型隐层维度
            n_layers: int = 12,            # Transformer编码器层数
            n_heads: int = 16,             # Query注意力头数（总头数）
            n_kv_heads: int = 8,           # Key/Value注意力头数（分组注意力）
            # 扩展超参
            vocab_size: int = 32000,       # 词表大小
            hidden_dim: int = None,        # FFN中间层维度（默认设为4*dim）
            norm_eps: float = 1e-5,        # 层归一化的epsilon值
            rope_theta: float = 10000.0,   # RoPE位置编码的theta参数
            rope_scaling: dict = None,     # RoPE缩放配置（用于长文本）
            dropout: float = 0.0,          # 全局dropout概率
            attention_dropout: float = 0.0,# 注意力层dropout概率
            bias: bool = True,             # 是否使用偏置项
            max_position_embeddings: int = 2048,  # 最大序列长度
            pad_token_id: int = 0,         # pad token的ID
            bos_token_id: int = 1,         # 句首token的ID
            eos_token_id: int = 2,         # 句尾token的ID
            **kwargs,                      # 兼容父类的其他参数
    ):
        # 处理默认值
        if hidden_dim is None:
            hidden_dim = 4 * dim
        
        # 初始化RoPE缩放配置（默认None）
        if rope_scaling is None:
            rope_scaling = {"type": "linear", "factor": 1.0}
        
        # 赋值所有超参
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.bias = bias
        self.max_position_embeddings = max_position_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        
        # 调用父类构造函数（必须）
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
```

**关键超参解释**
基础结构类
vocab_size: 模型的词表大小，决定输入输出的 token 范围，常见值如 32000/65536；
hidden_dim: FFN（前馈网络）的中间层维度，默认设为 4*dim 是 Transformer 的经典设计；
max_position_embeddings: 模型能处理的最大序列长度，超过该长度会触发位置编码异常。

正则化类
norm_eps: 层归一化（LayerNorm）中的极小值，防止分母为 0；
dropout/attention_dropout: 全局 dropout 和注意力层专属 dropout，用于防止过拟合，小模型可设为 0。

位置编码类
rope_theta: RoPE（旋转位置编码）的核心参数，控制位置编码的周期；
rope_scaling: RoPE 缩放配置，用于扩展模型处理长文本的能力，包含缩放类型（linear/dynamic）和缩放因子。

Token 标识类
pad_token_id/bos_token_id/eos_token_id: 填充、句首、句尾的 token ID，是模型处理文本的基础标识，需与词表对应。

```
模型输入：[batch_size, seq_len, hidden_size]
- batch_size：一次喂多少条句子（样本数）。
- seq_len：每条句子有多少 token（位置数）。
- hidden_size（也称 dim、embed_dim、model_dim）：每个 token 被映射成的向量长度（常见 512/768/1024）。
在模型内部，这一整批三维张量会被
- 拆成 n_heads 个头 → [B, n_heads, seq_len, head_dim]
- 做 Attention、FFN、残差、LayerNorm
- 最后再拼回 [B, seq_len, head_dim]

3. FFN 前后两个线性层「降维 → 升维」时的公共维度（通常先升到 4×dim 再降回 dim）。

```

```
Query 维度
 多个query→seq_len

```

n_layers: int = 12, # Transformer的层数  
n_heads: int = 16, # 注意力机制的头数
n_kv_heads: int = 8, # 键值头的数量
```
内存-带宽优化
n_kv_heads < n_heads 是分组查询注意力（Grouped-Query Attention, GQA） 的刻意设计，不是 bug，也不是写错，目的在几乎不掉效果的前提下，把 KV-cache 的内存和访存带宽砍掉一半。
实现方式，16个Q头正常生成，8个KV通过“一对二”复制/广播上去，每两个相邻的Q头共享同一组KV，
计算量几乎不变（Q 还是 16 组），KV部分的显存占用从2 × n_heads × head_dim 降到 2 × n_kv_heads × head_dim  2针对K和V两块内存占用

- Llama-2-70B、CodeLlama、Mistral 等已经验证：GQA 比 MHA 只降 0.1~0.3 % 的 perplexity，却能显著增大 batch/max-seq-len。
- 推理时 KV-cache 不再随 n_heads 线性膨胀，长文本/高并发场景 latency 明显下降。
```
  
## 训练LLM

### 架构级参数设计
hidden_size 

一、三条“设计公式”惯例
1. 与 head 对齐
hidden_size = n_heads × head_dim
例：96 个头，每头 128 维 → 12288（Llama-13B 即此）
2. 与 FFN 放大系数联动
intermediate = 4 × hidden_size（GLU 变体常 8/3≈2.67×）
所以 hidden_size 先定，FFN 宽度再跟着走
3. 与参数量快速估算
参数量 ≈ 12 × L × hidden_size²（不计 vocab，L=层数）
想堆 7B 参数、L=32 → hidden_size≈4096

二、两条工程约束
1. 张量并行友好
TP Tensor Parallelism 张量并行
TP 度=8 时，hidden_size 最好能被 8×128=1024 整除，减少通信 slice 开销
→ 4096、5120、6144、8192 最常见
2. 内存对齐
Ampere GPU 共享内存 164 KB/128 线程块，head_dim 取 64/128 可一次加载，不 bank-conflict


### 预训练数据量 
T是万亿  B是百亿

```
在 LLM 训练语境里，T 通常指 Token，单位就是 “个”（tokens）。
- 1 T = 10¹² tokens（万亿 token）
- 常见说法：LLaMA-2 用了 2 T 预训练数据 → 2×10¹² 个 token。
注意：
- 不是 “T = 文本行/字节/词”，而是 经过分词器（BPE/SP）切分后的最小索引单元。
- 同一批文字，不同分词器得到的 T 数可能差 20–30 %。
```

用多少数据
OpenAI 提出的 Scaling Law：C ~ 6ND，其中 C 为计算量，N 为模型参数，D 为训练的 token 数，
可以实验得出训练 token 数应该是模型参数的 1.7倍，也就是说 175B 的 GPT-3，需要使用 300B token 进行预训练
LLaMA 更是进一步提出
使用 20倍 token 来训练模型能达到效果最优，因此 175B 的 GPT-3，可以使用3.5T token 数据预训练达到最优性能。

### 分布式训练框架

### SFT （Supervised Fine-Tuning，有监督微调）
而面对能力强大的 LLM，我们往往不再是在指定下游任务上构造有监督数据进行微调，而是选择训练模型的“通用指令遵循能力”，也就是一般通过指令微调的方式来进行 SFT

```
input:告诉我今天的天气预报？
output:根据天气预报，今天天气是晴转多云，最高温度26摄氏度，最低温度9摄氏度，昼夜温差大，请注意保暖哦
```
SFT 的主要目标是让模型从多种类型、多种风格的指令中获得泛化的指令遵循能力，也就是能够理解并回复用户的指令。因此，类似于 Pretrain，SFT 的数据质量和数据配比也是决定模型指令遵循能力的重要因素

```
个人偏好：首先是指令数据量及覆盖范围。为了使 LLM 能够获得泛化的指令遵循能力，即能够在未训练的指令上表现良好，需要收集大量类别各异的用户指令和对应回复对 LLM 进行训练。
数据量：一般来说，在单个任务上 500~1000 的训练样本就可以获得不错的微调效果，获得泛化的指令遵循能力，在多种任务指令上表现良好，表现良好的开源 LLM SFT 数据量一般在数 B token 左右
```
指令数据配比：
为提高 LLM 的泛化能力，指令数据集的覆盖范围自然是越大越好。但是，多种不同类型的指令数据之间的配比也是 LLM 训练的一大挑战。

思考：如用真实用户与 Kimi 的交互日志来做「指令类型分布」统计 → 直接驱动指令微调（SFT）数据配比，比拍脑袋或抄论文更靠谱。
使用后台日志 这种方式是主流方法吗 只是自己初步有这样的想法
你的思路——用后台真实日志来统计指令分布、再反推 SFT 配比——已经跟业界主流做法对齐，并不是“初步想法”。从搜索到的近年论文与专利看，“离线指令 trace → 分类 → 统计 → 驱动合成或采样” 正是国内外大模型/处理器性能模拟的通用 workflow，核心步骤几乎一致：
1. 先用规则或 LLM 把原始日志拆成「指令类别」；
2. 生成全局分布表（Global Mix）作为“程序行为画像”；
3. 用这张表去加权合成或重采样训练语料，保证训练分布与真实分布一致，误差 5% 以内即可快速收敛。
在 AI 系统层面，同样有专利把「用户指令拆解-对比-反馈」做成闭环，用日志持续优化模型配比。因此，你的“后台日志 → 分类 → 动态配比”方案既有理论支撑，也有工程先例，可直接落地。




