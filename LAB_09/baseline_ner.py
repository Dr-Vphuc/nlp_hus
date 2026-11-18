"""Baseline NER (LAB 09, mục 13): token features -> DictVectorizer (sparse, one-hot / TF-IDF-like) -> Logistic Regression.

Mỗi token được phân loại ĐỘC LẬP (không có CRF / không có ràng buộc BIO), chỉ nhìn một cửa sổ ±2 từ.
Không dùng POS/chunk tag của CoNLL (để so sánh công bằng với Transformer, vốn chỉ nhận chữ).

Chạy:  python baseline_ner.py
Xuất:  baseline_predictions.json  (nhãn dự đoán trên test, dùng chung test set với Transformer)
       baseline_metrics.json      (P/R/F1 entity-level, theo type, theo độ dài)
"""

import json
import re
import time

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from ner_utils import (
    ID2LABEL,
    LABELS,
    SEED,
    entity_metrics,
    load_conll,
    tags_to_labels,
    token_accuracy,
    token_level_prf,
)


def word_shape(w):
    s = re.sub("[A-Z]", "X", w)
    s = re.sub("[a-z]", "x", s)
    s = re.sub("[0-9]", "d", s)
    return re.sub(r"(.)\1+", r"\1\1", s)  # nén lặp: "Xxxxx" -> "Xxx"


def word_feats(prefix, w):
    lw = w.lower()
    return {
        f"{prefix}w={lw}": 1,
        f"{prefix}title": int(w.istitle()),
        f"{prefix}upper": int(w.isupper()),
        f"{prefix}digit": int(any(c.isdigit() for c in w)),
        f"{prefix}shape={word_shape(w)}": 1,
    }


def sent2feats(words):
    """Danh sách dict đặc trưng, mỗi token một dict."""
    out = []
    n = len(words)
    for i, w in enumerate(words):
        lw = w.lower()
        f = {"bias": 1}
        f.update(word_feats("0:", w))
        f.update({f"suf3={lw[-3:]}": 1, f"suf2={lw[-2:]}": 1, f"pre3={lw[:3]}": 1, "len": min(len(w), 15) / 15})
        f["first_word"] = int(i == 0)
        for off in (-2, -1, 1, 2):
            j = i + off
            if 0 <= j < n:
                f.update(word_feats(f"{off}:", words[j]))
            else:
                f[f"{off}:PAD"] = 1
        out.append(f)
    return out


def flatten(ds):
    X, y = [], []
    for ex in ds:
        X.extend(sent2feats(ex["tokens"]))
        y.extend(ex["ner_tags"])
    return X, y


def predict_sentences(vec, clf, sentences):
    """sentences: list[list[str]] -> list[list[str]] nhãn BIO dự đoán."""
    out = []
    for words in sentences:
        X = vec.transform(sent2feats(words))
        out.append([ID2LABEL[int(k)] for k in clf.predict(X)])
    return out


def main():
    ds = load_conll()
    t0 = time.time()
    Xtr, ytr = flatten(ds["train"])
    vec = DictVectorizer()
    Xtr = vec.fit_transform(Xtr)
    print(f"train tokens: {Xtr.shape[0]}, features: {Xtr.shape[1]} ({time.time() - t0:.1f}s)")
    clf = LogisticRegression(C=10.0, max_iter=300, solver="lbfgs", random_state=SEED)
    clf.fit(Xtr, ytr)
    train_time = time.time() - t0
    print(f"fit done in {train_time:.1f}s")

    test_words = [ex["tokens"] for ex in ds["test"]]
    gold = [tags_to_labels(ex["ner_tags"]) for ex in ds["test"]]
    pred = predict_sentences(vec, clf, test_words)
    m = entity_metrics(gold, pred)
    print("overall", {k: round(v, 4) if isinstance(v, float) else v for k, v in m["overall"].items()})
    for t, r in m["per_type"].items():
        print(t, round(r["precision"], 4), round(r["recall"], 4), round(r["f1"], 4), r["support"])
    print("token accuracy", round(token_accuracy(gold, pred), 4), "token P/R/F1 (non-O)", [round(x, 4) for x in token_level_prf(gold, pred)])

    with open("baseline_predictions.json", "w", encoding="utf-8") as f:
        json.dump({"pred": pred}, f)
    m["train_time_s"] = train_time
    m["token_accuracy"] = token_accuracy(gold, pred)
    m["n_features"] = Xtr.shape[1]
    with open("baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(m, f, indent=1)


if __name__ == "__main__":
    main()
