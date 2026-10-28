"""Tiền xử lý cho LAB 05 — Sequence Models (IMDb).

Gồm:
- load_imdb / split_train_val / clean_text / tokenize: GIỮ NGUYÊN như LAB 04 để split và
  tokenizer giống hệt -> so sánh trực tiếp được với baseline TF-IDF + LR của LAB 04.
- Vocabulary: token -> id, chỉ build trên TRAIN. Id 0 = <pad>, id 1 = <unk>.
- pad_truncate / encode_corpus: cắt (truncation) hoặc đệm (padding) mọi sequence về max_len.
- make_loader: DataLoader trả về (ids, lengths, labels) cho PyTorch.

Padding chỉ để gom các sequence khác độ dài vào cùng một tensor; nó KHÔNG phải thông tin
ngôn ngữ. Vì vậy model dùng pack_padded_sequence (bỏ qua <pad>) và lengths được lưu riêng.
"""

import html
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
LABEL_NAMES = {0: "negative", 1: "positive"}
PAD, UNK = 0, 1
PAD_TOKEN, UNK_TOKEN = "<pad>", "<unk>"

_BR_RE = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")
# Giữ dạng rút gọn như "don't", "isn't" thành một token -> không mất phủ định.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)*")


# ----------------------------------------------------------------------------
# Data (giống LAB 04)
# ----------------------------------------------------------------------------
def load_imdb():
    """Tải IMDb từ Hugging Face. Trả về (train_df, test_df) với cột text, label."""
    from datasets import load_dataset

    ds = load_dataset("stanfordnlp/imdb")
    train_df = ds["train"].to_pandas()[["text", "label"]]
    test_df = ds["test"].to_pandas()[["text", "label"]]
    return train_df, test_df


def split_train_val(df, val_size=0.2, seed=SEED):
    """Chia train/validation có stratify theo label để giữ nguyên tỉ lệ class."""
    train_df, val_df = train_test_split(
        df, test_size=val_size, stratify=df["label"], random_state=seed
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def class_distribution(labels):
    counts = pd.Series(labels).map(LABEL_NAMES).value_counts()
    return pd.DataFrame({"Count": counts, "Percentage": (100 * counts / counts.sum()).round(2)})


# ----------------------------------------------------------------------------
# Text normalization (giống LAB 04)
# ----------------------------------------------------------------------------
def clean_text(text):
    """Lowercase, bỏ thẻ HTML (<br />), decode HTML entity, chuẩn hóa dấu nháy/khoảng trắng."""
    text = html.unescape(text)
    text = _BR_RE.sub(" ", text)
    text = _TAG_RE.sub(" ", text)
    text = text.replace("’", "'").replace("`", "'")
    text = text.lower()
    return _SPACE_RE.sub(" ", text).strip()


def tokenize(text):
    """Tách từ sau khi clean_text. Dấu câu bị bỏ, giữ contraction (don't, it's)."""
    return _TOKEN_RE.findall(clean_text(text))


def tfidf_params(**overrides):
    """Tham số TfidfVectorizer của baseline LAB 04 (unigram, min_df=2, cùng tokenizer)."""
    params = dict(tokenizer=tokenize, token_pattern=None, lowercase=False, min_df=2)
    params.update(overrides)
    return params


# ----------------------------------------------------------------------------
# Vocabulary & token ids
# ----------------------------------------------------------------------------
class Vocabulary:
    """Ánh xạ token <-> id. Chỉ giữ max_size token phổ biến nhất (tần suất >= min_freq) của TRAIN.

    Token ngoài vocabulary -> <unk> (id 1). Id 0 dành cho <pad>.
    """

    def __init__(self, token_lists, max_size=20_000, min_freq=2):
        counts = Counter(tok for toks in token_lists for tok in toks)
        kept = [t for t, c in counts.most_common(max_size) if c >= min_freq]
        self.itos = [PAD_TOKEN, UNK_TOKEN] + kept
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.counts = counts

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        return [self.stoi.get(t, UNK) for t in tokens]

    def decode(self, ids):
        return [self.itos[i] for i in ids if i != PAD]


def pad_truncate(ids, max_len, truncation="head"):
    """Trả về (mảng độ dài max_len, độ dài thật sau khi cắt).

    truncation="head": giữ max_len token ĐẦU (mất phần cuối review);
    truncation="tail": giữ max_len token CUỐI (mất phần đầu review).
    Sequence rỗng được thay bằng [<unk>] để luôn có ít nhất 1 timestep.
    """
    ids = list(ids) or [UNK]
    ids = ids[:max_len] if truncation == "head" else ids[-max_len:]
    out = np.full(max_len, PAD, dtype=np.int64)
    out[: len(ids)] = ids
    return out, len(ids)


def encode_corpus(token_lists, vocab, max_len, truncation="head"):
    """List các list token -> (ids [N, max_len], lengths [N])."""
    pairs = [pad_truncate(vocab.encode(toks), max_len, truncation) for toks in token_lists]
    ids = np.stack([p[0] for p in pairs])
    lengths = np.array([p[1] for p in pairs], dtype=np.int64)
    return ids, lengths


def pct_tokens_retained(full_lengths, max_len):
    """% tổng số token còn lại sau khi cắt mọi document về max_len."""
    full_lengths = np.asarray(full_lengths)
    return 100 * np.minimum(full_lengths, max_len).sum() / full_lengths.sum()


def make_loader(ids, lengths, labels, batch_size=64, shuffle=False, seed=SEED):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    ds = TensorDataset(torch.as_tensor(ids), torch.as_tensor(lengths),
                       torch.as_tensor(labels, dtype=torch.float32))
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=gen)
