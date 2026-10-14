"""Tiền xử lý & biểu diễn văn bản cho LAB 04 — Text Classification (IMDb).

Gồm:
- load_imdb / split_train_val: tải dữ liệu và chia train/validation (stratified).
- clean_text / tokenize: chuẩn hóa văn bản, dùng CHUNG cho TF-IDF và Word2Vec
  để hai representation được so sánh trên cùng một cách tách từ.
- MeanEmbeddingVectorizer: Average Word2Vec dưới dạng sklearn transformer,
  Word2Vec chỉ được train trong .fit() (tức là chỉ trên TRAIN) -> không leakage.
"""

import html
import re

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split

SEED = 42
LABEL_NAMES = {0: "negative", 1: "positive"}

_BR_RE = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")
# Giữ dạng rút gọn như "don't", "isn't" thành một token -> không mất phủ định.
# (token_pattern mặc định của sklearn sẽ tách "don't" thành "don" + "t" rồi bỏ "t".)
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)*")


# ----------------------------------------------------------------------------
# Data
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


# ----------------------------------------------------------------------------
# Text normalization
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


def tokenize_no_stopwords(text):
    """Như tokenize nhưng bỏ stopwords theo danh sách ENGLISH_STOP_WORDS của sklearn."""
    return [t for t in tokenize(text) if t not in ENGLISH_STOP_WORDS]


def tfidf_params(remove_stopwords=False, **overrides):
    """Tham số TfidfVectorizer dùng chung tokenizer ở trên (lowercase đã làm trong clean_text)."""
    params = dict(
        tokenizer=tokenize_no_stopwords if remove_stopwords else tokenize,
        token_pattern=None,
        lowercase=False,
        min_df=2,
    )
    params.update(overrides)
    return params


# ----------------------------------------------------------------------------
# Average Word2Vec
# ----------------------------------------------------------------------------
class MeanEmbeddingVectorizer(BaseEstimator, TransformerMixin):
    """v(x) = (1/|x|) * sum_{w in x} v_w, chỉ tính trên các từ có trong vocabulary Word2Vec.

    Văn bản không có từ nào trong vocabulary -> vector 0.
    """

    def __init__(self, vector_size=100, window=5, min_count=2, sg=1, epochs=10,
                 workers=4, seed=SEED):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.epochs = epochs
        self.workers = workers
        self.seed = seed

    def fit(self, texts, y=None):
        sentences = [tokenize(t) for t in texts]
        self.w2v_ = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            sg=self.sg,
            epochs=self.epochs,
            workers=self.workers,
            seed=self.seed,
        )
        return self

    @property
    def wv(self):
        return self.w2v_.wv

    def doc_vector(self, tokens):
        tokens = [t for t in tokens if t in self.wv.key_to_index]
        if not tokens:
            return np.zeros(self.vector_size, dtype=np.float32)
        return self.wv[tokens].mean(axis=0)

    def transform(self, texts):
        return np.vstack([self.doc_vector(tokenize(t)) for t in texts])

    def oov_rate(self, texts):
        """Tỉ lệ token không có trong vocabulary Word2Vec (bị bỏ khi lấy trung bình)."""
        total = oov = 0
        for t in texts:
            toks = tokenize(t)
            total += len(toks)
            oov += sum(tok not in self.wv.key_to_index for tok in toks)
        return oov / max(total, 1)


# ----------------------------------------------------------------------------
# Inspection helpers
# ----------------------------------------------------------------------------
def class_distribution(labels):
    counts = pd.Series(labels).map(LABEL_NAMES).value_counts()
    return pd.DataFrame({"Count": counts, "Percentage": (100 * counts / counts.sum()).round(2)})
