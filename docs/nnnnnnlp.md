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


NSP next sentence prediction 下一句预测



