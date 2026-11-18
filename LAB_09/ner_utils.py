"""Tiện ích dùng chung cho LAB 09 — NER trên CoNLL-2003.

Gồm:
- load_conll(): đọc CoNLL-2003 (bản parquet của `lhoestq/conll2003`, vì `conll2003` gốc dùng loading script
  mà `datasets` mới không còn hỗ trợ). Nhãn đã ở dạng IOB2 (= BIO: mọi entity bắt đầu bằng B-).
- BIO helpers: spans_from_tags (chuyển chuỗi BIO -> list entity), words_to_bio (annotation entity -> nhãn BIO).
- align_labels_first_subtoken: Strategy A (nhãn chỉ gán cho subword đầu của mỗi word, phần còn lại = -100 / IGNORE).
- Metric entity-level tự cài (precision / recall / F1, tổng + theo type + theo độ dài), đối chiếu được với seqeval.
"""

from collections import Counter, defaultdict

SEED = 42
DATASET_NAME = "lhoestq/conll2003"
LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}
ENTITY_TYPES = ["PER", "ORG", "LOC", "MISC"]
IGNORE_INDEX = -100  # CrossEntropyLoss(ignore_index=-100) bỏ qua các vị trí này


# --------------------------------------------------------------------------- data
def load_conll():
    """Trả về DatasetDict (train / validation / test) với cột `tokens`, `ner_tags` (id theo LABELS)."""
    from datasets import load_dataset

    return load_dataset(DATASET_NAME)


def tags_to_labels(tag_ids):
    return [ID2LABEL[int(t)] for t in tag_ids]


# --------------------------------------------------------------------------- BIO
def spans_from_tags(tags):
    """Chuyển chuỗi nhãn BIO (list[str]) thành tập entity {(type, start, end)} với end là chỉ số *exclusive*.

    Quy ước giống conlleval/seqeval (chế độ mặc định, lenient với I- đứng đầu):
    - B-X mở entity mới; I-X tiếp tục entity đang mở nếu cùng type X;
    - I-X mà không có entity đang mở, hoặc khác type, thì mở entity mới (lỗi định dạng vẫn được tính một entity);
    - O đóng entity.
    """
    spans = []
    start, etype = None, None
    for i, tag in enumerate(list(tags) + ["O"]):  # sentinel O để đóng entity cuối
        if tag == "O":
            prefix, t = "O", None
        else:
            prefix, t = tag.split("-", 1)
        if etype is not None and (prefix in ("O", "B") or t != etype):
            spans.append((etype, start, i))
            start, etype = None, None
        if prefix == "B" or (prefix == "I" and etype is None):
            start, etype = i, t
    return set(spans)


def words_to_bio(words, entities):
    """Annotation entity (theo cụm từ) -> nhãn BIO theo word.

    entities: list[(cụm_từ, type)], cụm từ phải khớp liên tiếp các word (so khớp theo từ, từ trái sang phải).
    Ví dụ: words_to_bio(["Elon","Musk","founded","SpaceX"], [("Elon Musk","PER"),("SpaceX","ORG")]).
    """
    tags = ["O"] * len(words)
    for phrase, etype in entities:
        p = phrase.split()
        for i in range(len(words) - len(p) + 1):
            if words[i : i + len(p)] == p and all(t == "O" for t in tags[i : i + len(p)]):
                tags[i] = f"B-{etype}"
                for j in range(i + 1, i + len(p)):
                    tags[j] = f"I-{etype}"
                break
        else:
            raise ValueError(f"Không tìm thấy entity {phrase!r} trong {words}")
    return tags


# --------------------------------------------------------------------------- alignment
def align_labels_first_subtoken(word_ids, word_labels):
    """Strategy A: subword đầu tiên của mỗi word nhận nhãn của word; các subword sau + token đặc biệt = -100.

    word_ids: kết quả `encoding.word_ids()` (None cho [CLS]/[SEP]/[PAD], i cho token thuộc word i).
    word_labels: list[int] nhãn theo word (đã là id).
    """
    aligned, prev = [], None
    for wid in word_ids:
        if wid is None:
            aligned.append(IGNORE_INDEX)
        elif wid != prev:
            aligned.append(word_labels[wid])
        else:
            aligned.append(IGNORE_INDEX)
        prev = wid
    return aligned


def align_labels_propagate(word_ids, word_labels):
    """Strategy B (chỉ để minh họa/so sánh): lan truyền nhãn; B-X -> I-X cho subword tiếp theo."""
    aligned, prev = [], None
    for wid in word_ids:
        if wid is None:
            aligned.append(IGNORE_INDEX)
        elif wid != prev:
            aligned.append(word_labels[wid])
        else:
            lab = ID2LABEL[word_labels[wid]]
            if lab.startswith("B-"):
                lab = "I-" + lab[2:]
            aligned.append(LABEL2ID[lab])
        prev = wid
    return aligned


# --------------------------------------------------------------------------- metrics
def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def entity_metrics(gold_seqs, pred_seqs):
    """Entity-level TP/FP/FN. gold_seqs/pred_seqs: list[list[str]] (nhãn BIO theo word).

    Một entity dự đoán là TP chỉ khi (type, start, end) trùng *hoàn toàn* với một entity gold (exact match).
    Trả về dict: overall, per_type (PER/ORG/LOC/MISC), per_length (1,2,3,4+); mỗi mục có tp, fp, fn, precision, recall, f1, support.
    """
    counts = defaultdict(lambda: [0, 0, 0])  # key -> [tp, fp, fn]
    for g, p in zip(gold_seqs, pred_seqs):
        gs, ps = spans_from_tags(g), spans_from_tags(p)
        for e in ps & gs:
            _add(counts, e, 0)
        for e in ps - gs:
            _add(counts, e, 1)
        for e in gs - ps:
            _add(counts, e, 2)

    def pack(c):
        tp, fp, fn = c
        p, r, f = _prf(tp, fp, fn)
        return dict(tp=tp, fp=fp, fn=fn, precision=p, recall=r, f1=f, support=tp + fn)

    out = {"per_type": {}, "per_length": {}}
    tot = [0, 0, 0]
    for t in ENTITY_TYPES:
        c = counts[("type", t)]
        out["per_type"][t] = pack(c)
        for k in range(3):
            tot[k] += c[k]
    out["overall"] = pack(tot)
    for L in ("1", "2", "3", "4+"):
        out["per_length"][L] = pack(counts[("len", L)])
    return out


def _add(counts, e, idx):
    etype, s, t = e
    n = t - s
    counts[("type", etype)][idx] += 1
    counts[("len", "4+" if n >= 4 else str(n))][idx] += 1


def token_accuracy(gold_seqs, pred_seqs):
    """Accuracy theo token (kể cả O) — chỉ để minh họa vì sao KHÔNG nên dùng làm metric chính cho NER."""
    total = correct = 0
    for g, p in zip(gold_seqs, pred_seqs):
        for a, b in zip(g, p):
            total += 1
            correct += a == b
    return correct / total


def token_level_prf(gold_seqs, pred_seqs):
    """Micro P/R/F1 theo token *chỉ trên các token không phải O* (token-level, bỏ qua nhãn O đúng)."""
    tp = fp = fn = 0
    for g, p in zip(gold_seqs, pred_seqs):
        for a, b in zip(g, p):
            if a != "O" and a == b:
                tp += 1
            elif b != "O" and a != b:
                fp += 1
            if a != "O" and a != b:
                fn += 1
    return _prf(tp, fp, fn)


def error_lists(words_list, gold_seqs, pred_seqs):
    """Phân loại lỗi entity-level: false positive, false negative và boundary error.

    - boundary: entity dự đoán và entity gold cùng type, chồng lấn (overlap) nhưng span khác nhau.
    - fp (spurious): entity dự đoán không chồng lấn entity gold nào (hoặc chồng lấn nhưng khác type -> wrong type, ghi riêng).
    - fn (missing): entity gold không chồng lấn entity dự đoán nào.
    - wrong_type: span trùng hoàn toàn nhưng khác type, hoặc chồng lấn khác type.
    """
    out = {"fp": [], "fn": [], "boundary": [], "wrong_type": []}
    for idx, (w, g, p) in enumerate(zip(words_list, gold_seqs, pred_seqs)):
        gs, ps = spans_from_tags(g), spans_from_tags(p)
        for e in ps - gs:
            over = [x for x in gs - ps if x[1] < e[2] and e[1] < x[2]]
            if not over:
                out["fp"].append((idx, e, None))
            else:
                x = over[0]
                kind = "boundary" if x[0] == e[0] else "wrong_type"
                out[kind].append((idx, e, x))
        for x in gs - ps:
            over = [e for e in ps - gs if x[1] < e[2] and e[1] < x[2]]
            if not over:
                out["fn"].append((idx, None, x))
    return out


def span_text(words, span):
    return " ".join(words[span[1] : span[2]])
