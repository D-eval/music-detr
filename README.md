# music-detr

music information retrieval (MIR)



项目基于 [open-det2](https://github.com/Med-Process/Open-Det2)

识别音乐的重要音符，并给出音色的自然语言描述

数据标注困难时，例如音乐理解，
短上下文识别结果做 LALM 的结构提示。
实现逐个元素的理解，避免 LALM 犯错。

用detr提取cqt上的稀疏结构，并对每个目标做自然语言描述。

在后续工作中，着重于解释各个目标的关系，例如音色之间的依赖。

# 具体架构

我现在有两个做法，我不是要从音高谱(P, T) (pitch, time)去预测真是midi (P, T) 嘛，因为形状一样所以可以用U-net，但是音高谱信息不足，所以需要去融合audio_emb的信息，大概就是用注意力去融合，但是我有两种做法，我不知道怎么处理(P,T) 1、我直接把P嵌入特征，(P, T) -> (C, T) -> (C*8, T//2) -> (c*32, T//4)... -> (C, T) -> (P, T) 2、我需要开一个新特征, (P, T) -> (1, P, T) -> (8, P//2, T//2) -> (32, P//4, T//4) -> ... -> (8, P//2, T//2) -> (1, P, T) -> (P, T)

我之所以要想到方案1，是因为我觉得直接做方案2会有问题，因为conv2d需要保证连续的像素对应同一个实体，时间上确实是连续的，但是P维度上，一个实体可能要跨越12个像素，7个像素（泛音、谐波），所以不能用常规的卷积做。但是如果我提前就在P维度做一个linear，就有希望把这些特征变成连续的，所以我就想到方案1，或许更稳妥一些

在P维度做attention，最稳妥。

说的对，我应该用局部注意力(P * Delta_t, C)，同时也要注意到局部位置的audio_emb编码 (Delta_t, C)，但是你觉得是用audio_emb好（wav2vec2之类的），还是直接用原始的频谱特征好，我用这个是因为能量谱可能有相位问题。而且还有一个问题，这样的话 (P * Delta_t, C) 尺寸就不会改变，也就是可以直接做残差连接，也就没必要Unet了

现在我们来想想策略，应该设计成因果的还是无因果的，如果是因果的，每次只预测下一个 (P, 3) （3表示[静止，触发，延续上一个音]），还是设计成全局的，直接预测(T, P, 3)


我突然意识到我可以直接把freq本身当作它的F编码，比如说对100->20000的频率，直接用sin(freq * k / d)和cos 当作编码，这样pitch_spec和spec的语义空间就对齐了，concat后可以直接做注意力

# 第一次冲击
3/30

把所有人展平之后，会出现 out of memory,

`[text_emb, pitch_emb, freq_emb]`

三个人下来，长度是 `1+ T*P + T*F = 1 + 117*85 + 117*128 = 24922`

这个上下文也太长了吧，我们首先应该压缩上下文长度，要不然要计算
`(24922, C) @ (C, 24922)`，结果是 `621106084`

主要是如果我把频率和时间分别做attention，我就没办法解码 text 了，我心中最理想的情况是，如果 [text, pitch, freq] 直接self_attn，那text位置后续就可以作为query了，这个query会同时把pitch变成target，同时把自己变成text_prompt，这个prompt经过一个语言模型就得到了text

但是如果我分别做的话，我能把text融合给它们，但是我不知怎样把它们融合给text。

然后chatgpt告诉我，可以做双向attention，就是说 freq, pitch 之间
做 factorized 的 self attention，
然后他俩分别和 text 做 crossAttention，

然后 text, 和 [freq, pitch] 分别经过不同的 FFN，

这就有2个问题，
1、首先，text应该做self attention吗，如果不做的话，[pitch, freq]那边做了一个self和一个cross，text这边只做一个cross，感觉不太对称。
2、其次，text这边本身就是用text encoder提取的，哦哦，后续会替换成query，这样的话self attention的确要做，突然就感觉十分的合理。但是detr的self attention是在不同的类型之间做的，也就是在 N 维度上，顺带一说，text_emb 的维度是 `(N, L, C)`，其中 N 是不同的类别，L 是截取的句子长度。但是如果是detr的话，应该是 `(B, N, C)` 才对，也就是说 query 是在 N 之间做注意力的。

但是另一边，[pitch, freq] 似乎没有在 N 方向上做。因为我们需要让 pitch 在 N 方向上表示不同的类别。我们一开始只用了查询的思路。直接把N当成了不同的batch。

也就是说目前的思绪其实是有点乱的。难道说真实的做法是，[pitch, freq]要在 在 F, T, N 三个方向做注意力然后相加。在 N 方向做的时候把 query 镶嵌在前面，同时通过一个N编码让特定的n和 n的 query 能互相认识。

好饿，我得去吃会。

