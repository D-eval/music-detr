from torch.utils.data import DataLoader
from configs.config import get_config

def build_corpus(dataset):
    cfg = get_config()
    
    corpus = []
    print("开始构建语料库")
    for i in range(len(dataset)):
        _, target = dataset[i]

        texts = target["text"]

        for t in texts:
            assert len(t) != 0
            if cfg.tokenizer.only_last_detail:
                parts = t.split("，")
                desc = [cfg.tokenizer.begin, parts[-1],cfg.tokenizer.end]
                corpus.append(" ".join(desc))
            else:
                parts = t.split("，")
                if len(parts)==1:
                    t = " ".join([cfg.tokenizer.begin,
                        cfg.tokenizer.role,
                        t,
                        cfg.tokenizer.end])
                elif len(parts)==2:
                    t = " ".join([cfg.tokenizer.begin,
                        cfg.tokenizer.role,
                        parts[0],
                        cfg.tokenizer.inst,
                        parts[1],
                        cfg.tokenizer.end])
                else:
                    desc = "，".join(parts[2:])
                    t = " ".join([cfg.tokenizer.begin,
                        cfg.tokenizer.role,
                        parts[0],
                        cfg.tokenizer.inst,
                        parts[1],
                        cfg.tokenizer.desc,
                        desc,
                        cfg.tokenizer.end])
                corpus.append(t)
    return corpus

from tokenizers import Tokenizer, models, trainers, pre_tokenizers

def train_bpe(corpus, vocab_size=8000):
    
    cfg = get_config()
    save_path = cfg.tokenizer.save_path
    
    # ===== 1. 初始化 =====
    tokenizer = Tokenizer(models.BPE())

    # 中文建议用 char-level
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

    # ===== 2. trainer =====
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[
            cfg.tokenizer.pad,
            cfg.tokenizer.begin,
            cfg.tokenizer.end,
            cfg.tokenizer.unk,
            cfg.tokenizer.role,
            cfg.tokenizer.inst,
            cfg.tokenizer.desc,
        ]
    )

    # ===== 3. 训练 =====
    tokenizer.train_from_iterator(corpus, trainer)

    # ===== 4. 保存 =====
    tokenizer.save(save_path)

    return tokenizer

