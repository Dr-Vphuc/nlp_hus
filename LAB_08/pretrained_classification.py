"""LAB 08 — Pretrained Transformer (DistilBERT, BERT-style) + fine-tuning cho IMDb sentiment.

File này chứa toàn bộ phần code nặng của lab:

- ``load_imdb_splits``      : IMDb, split train/val giống LAB 05/06 (20k/5k, stratify, seed 42) rồi lấy subset cố định train 10k / val 2k;
                              test: ``test_subset_index`` (5,000 mẫu stratified từ 25k test).
- ``load_domain_sets``      : 2 tập dùng cho domain shift (Amazon product reviews 2,000 mẫu, Rotten Tomatoes 1,066 mẫu).
- ``CLSClassifier``         : [CLS] của encoder -> Dropout -> Linear (head là phần task-specific).
- ``SmallTransformer``      : Transformer encoder nhỏ, khởi tạo ngẫu nhiên (kiểu LAB 07) -> "from scratch".
- ``run_experiment``        : train 1 cấu hình, đo thời gian / bộ nhớ / metrics, lưu xác suất dự đoán.
- ``RUNS``                  : registry các cấu hình cho Exp 1-5.
- ``collect_results``       : gộp ``outputs/*.json`` thành ``results.csv`` và ``domain_shift.csv``.

Chạy ở máy 1 GPU 6GB dùng chung (RTX 3050 Laptop): batch hiệu dụng 32 được chia micro-batch + gradient accumulation (``accum_steps``),
OOM thì tự chờ rồi chạy lại bước đó (``retry_oom``).

Cách chạy::

    python pretrained_classification.py --list
    python pretrained_classification.py --runs ft_len128_100 frozen_len128_100
    python pretrained_classification.py --runs all          # toàn bộ (bỏ qua run đã có outputs/<run>.json)
    python pretrained_classification.py --collect           # chỉ gộp kết quả -> results.csv

Mỗi run ghi ``outputs/<run>.json`` (metrics) và ``outputs/<run>_probs.npz`` (xác suất trên các tập đánh giá).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
SEED = 42
MODEL_NAME = "distilbert-base-uncased"   # BERT-style, 6 layer (66M). bert-base (110M) quá nặng cho GPU 6GB dùng chung -> xem README
TRAIN_POOL, VAL_SUBSET, TEST_SUBSET = 10_000, 2_000, 5_000   # GPU dùng chung nên giảm cỡ dữ liệu: subset cố định (stratified, seed 42) cho mọi run
LABEL_NAMES = {0: "negative", 1: "positive"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def hardware_string() -> str:
    if torch.cuda.is_available():
        return f"{torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB), torch {torch.__version__}"
    return f"CPU ({platform.processor() or platform.machine()}), torch {torch.__version__}"


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def _clean(text: str) -> str:
    """IMDb chứa thẻ <br /> — thay bằng dấu cách (giống tinh thần clean_text của LAB 04)."""
    return text.replace("<br />", " ").replace("<br/>", " ").replace("<br>", " ").strip()


def load_imdb_splits(val_size: float = 0.2, seed: int = SEED):
    """Trả về (train_df, val_df, test_df). Split gốc giống LAB 05/06 (20,000 / 5,000 / 25,000), sau đó lấy subset cố định: train pool 10,000, val 2,000. Test (25,000) giữ nguyên ở đây; run dùng test subset 5,000 (test_subset_index)."""
    from datasets import load_dataset

    ds = load_dataset("stanfordnlp/imdb")
    full_train = ds["train"].to_pandas()[["text", "label"]]
    test_df = ds["test"].to_pandas()[["text", "label"]]
    train_df, val_df = train_test_split(full_train, test_size=val_size, stratify=full_train["label"], random_state=seed)   # 20k / 5k như LAB 05/06
    train_df, _ = train_test_split(train_df, train_size=TRAIN_POOL, stratify=train_df["label"], random_state=seed)       # pool 10k
    val_df, _ = train_test_split(val_df, train_size=VAL_SUBSET, stratify=val_df["label"], random_state=seed)             # 2k
    out = []
    for df in (train_df, val_df, test_df):
        df = df.reset_index(drop=True).copy()
        df["text"] = df["text"].map(_clean)
        out.append(df)
    return tuple(out)


def load_domain_sets(n_amazon: int = 2000, seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Hai tập khác IMDb: product reviews (khác domain) và Rotten Tomatoes (cùng domain phim, khác style: câu rất ngắn)."""
    from datasets import load_dataset

    amazon = load_dataset("fancyzhx/amazon_polarity", split="test").to_pandas()
    amazon["text"] = (amazon["title"].fillna("") + ". " + amazon["content"].fillna("")).map(_clean)
    amazon = amazon[["text", "label"]]
    amazon, _ = train_test_split(amazon, train_size=n_amazon, stratify=amazon["label"], random_state=seed)
    rt = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="test").to_pandas()[["text", "label"]]
    return {"amazon": amazon.reset_index(drop=True), "rotten_tomatoes": rt.reset_index(drop=True)}


def test_subset_index(test_df: pd.DataFrame, n: int = TEST_SUBSET, seed: int = SEED) -> np.ndarray:
    """Chỉ số (đã sắp xếp) của test subset n mẫu, stratified. Cố định cho mọi run."""
    idx, _ = train_test_split(np.arange(len(test_df)), train_size=n, stratify=test_df["label"].to_numpy(), random_state=seed)
    return np.sort(idx)


def subsample(df: pd.DataFrame, frac: float, seed: int = SEED) -> pd.DataFrame:
    """Stratified subset của training set (validation/test giữ nguyên)."""
    if frac >= 1.0:
        return df
    sub, _ = train_test_split(df, train_size=frac, stratify=df["label"], random_state=seed)
    return sub.reset_index(drop=True)


def encode(tokenizer, texts, max_len: int) -> list[list[int]]:
    """Tokenize không padding (padding làm động theo batch trong collate). Giữ [CLS] ... [SEP]."""
    return tokenizer(list(texts), truncation=True, max_length=max_len)["input_ids"]


def collate(batch, pad_id: int = 0):
    ids = [b[0] for b in batch]
    max_t = max(len(x) for x in ids)
    input_ids = torch.full((len(ids), max_t), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(ids), max_t), dtype=torch.long)
    for i, x in enumerate(ids):
        input_ids[i, : len(x)] = torch.tensor(x)
        attention_mask[i, : len(x)] = 1
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return input_ids, attention_mask, labels


def make_batches(ids, labels, batch_size, shuffle, rng=None, sort_by_len=False):
    """Generator trả về các batch (input_ids, attention_mask, labels), kèm chỉ số gốc (để ghép lại thứ tự)."""
    idx = np.arange(len(ids))
    if sort_by_len:
        idx = np.argsort([len(x) for x in ids], kind="stable")
    elif shuffle:
        idx = rng.permutation(len(ids))
    for s in range(0, len(idx), batch_size):
        sel = idx[s : s + batch_size]
        yield sel, *collate([(ids[i], labels[i]) for i in sel])


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
class CLSClassifier(nn.Module):
    """encoder -> [CLS] hidden state (vị trí 0 của last_hidden_state) -> Dropout -> Linear -> logits.

    ``pretrained=True``  : nạp trọng số BERT đã pretrain (fine-tuning hoặc frozen).
    ``pretrained=False`` : cùng kiến trúc BERT nhưng khởi tạo ngẫu nhiên.
    ``freeze=True``      : encoder bị đóng băng (requires_grad=False, chạy eval + no_grad) -> chỉ train head.
    """

    def __init__(self, name: str = MODEL_NAME, pretrained: bool = True, freeze: bool = False, num_labels: int = 2, dropout: float = 0.1):
        super().__init__()
        if pretrained:
            self.encoder = AutoModel.from_pretrained(name)
        else:
            self.encoder = AutoModel.from_config(AutoConfig.from_pretrained(name))
        self.freeze = freeze
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.encoder.eval()  # frozen: không dropout trong encoder
        return self

    def forward(self, input_ids, attention_mask):
        if self.freeze:
            with torch.no_grad():
                h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        else:
            h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.head(self.dropout(h[:, 0]))


class SmallTransformer(nn.Module):
    """Transformer encoder nhỏ (kiểu LAB 07) khởi tạo ngẫu nhiên, dùng cùng tokenizer WordPiece của BERT.

    token emb + learned positional emb -> N x (MHA + FFN, residual + LayerNorm) -> [CLS] -> Dropout -> Linear.
    """

    def __init__(self, vocab_size: int, d_model=256, n_heads=4, d_ff=1024, n_layers=4, max_pos=512, dropout=0.1, num_labels=2):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Embedding(max_pos, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, num_labels)
        self.freeze = False

    def forward(self, input_ids, attention_mask):
        pos = torch.arange(input_ids.size(1), device=input_ids.device)
        x = self.drop(self.norm(self.tok(input_ids) + self.pos(pos)[None]))
        x = self.enc(x, src_key_padding_mask=(attention_mask == 0))
        return self.head(self.drop(x[:, 0]))


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ----------------------------------------------------------------------------
# Train / evaluate
# ----------------------------------------------------------------------------
def is_oom(e: BaseException) -> bool:
    return "out of memory" in str(e).lower()


def retry_oom(fn, *args, tries: int = 60, wait: float = 15.0, **kw):
    """GPU dung chung voi tien trinh khac -> neu OOM thi doi roi thu lai chinh buoc do (khong mat ca epoch)."""
    for k in range(tries):
        try:
            return fn(*args, **kw)
        except (torch.OutOfMemoryError, RuntimeError) as e:
            if not is_oom(e) or k == tries - 1:
                raise
            import gc
            gc.collect(); torch.cuda.empty_cache()
            print(f"    [OOM] retry {k + 1}/{tries} after {wait:.0f}s", flush=True)
            time.sleep(wait)



@torch.no_grad()
def predict_probs(model, ids, labels, batch_size: int = 64) -> np.ndarray:
    """Softmax probabilities (N, 2), đúng thứ tự dữ liệu đầu vào."""
    model.eval()
    probs = np.zeros((len(ids), 2), dtype=np.float32)
    for sel, x, m, _ in make_batches(ids, labels, batch_size, shuffle=False, sort_by_len=True):
        def fwd():
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                return torch.softmax(model(x.to(DEVICE), m.to(DEVICE)).float(), dim=-1).cpu().numpy()
        probs[sel] = retry_oom(fwd)
    return probs


def metrics_from_probs(probs: np.ndarray, labels) -> dict:
    labels = np.asarray(labels)
    pred = probs.argmax(1)
    p, r, f, _ = precision_recall_fscore_support(labels, pred, average="binary", pos_label=1, zero_division=0)
    loss = float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1)).mean())
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    return dict(accuracy=float(accuracy_score(labels, pred)), precision=float(p), recall=float(r), f1=float(f), loss=loss,
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), n=int(len(labels)))


@dataclass
class RunConfig:
    name: str
    exp: str
    kind: str = "bert"            # "bert" | "small"
    pretrained: bool = True
    freeze: bool = False
    train_frac: float = 1.0
    max_len: int = 128
    epochs: int = 2
    batch_size: int = 32           # effective batch size (= micro_batch * accum_steps)
    accum_steps: int = 1           # gradient accumulation: GPU dùng chung với tiến trình khác nên chia batch nhỏ để tránh OOM
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = SEED
    full_test: bool = False        # đánh giá thêm trên toàn bộ 25k test (dùng cho Application / error analysis)
    domain_eval: bool = False      # đánh giá thêm trên Amazon / Rotten Tomatoes
    demo: bool = False             # dự đoán vài câu Application
    note: str = ""


_DATA_CACHE: dict = {}


def get_data():
    if "imdb" not in _DATA_CACHE:
        _DATA_CACHE["imdb"] = load_imdb_splits()
    return _DATA_CACHE["imdb"]


DEMO_SENTENCES = [
    "This movie is great.",
    "I do not like this movie.",
    "The film is not bad at all.",
    "The plot was predictable, but the acting was absolutely excellent.",
    "Oh great, another two hours of my life I will never get back.",
    "I expected to hate it, yet I ended up loving every minute.",
    "The product broke after two days. Terrible quality.",
    "The movie was not good, not even close to being good.",
]


def run_experiment(cfg: RunConfig, force: bool = False) -> dict:
    OUT.mkdir(exist_ok=True)
    res_path = OUT / f"{cfg.name}.json"
    if res_path.exists() and not force:
        print(f"[skip] {cfg.name} (đã có {res_path.name})")
        return json.loads(res_path.read_text(encoding="utf-8"))

    set_seed(cfg.seed)
    train_df, val_df, test_df = get_data()
    train_df = subsample(train_df, cfg.train_frac, cfg.seed)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    enc = lambda df: encode(tok, df["text"], cfg.max_len)
    tr_ids, va_ids = enc(train_df), enc(val_df)
    sub = test_subset_index(test_df)
    if cfg.full_test:
        te_ids = enc(test_df)                      # toàn bộ 25k
    else:                                          # chỉ tokenize subset 10k (giữ nguyên chỉ số gốc)
        te_ids = [None] * len(test_df)
        for i, ids_ in zip(sub, encode(tok, test_df["text"].iloc[sub], cfg.max_len)):
            te_ids[i] = ids_
    tr_y, va_y, te_y = train_df["label"].to_numpy(), val_df["label"].to_numpy(), test_df["label"].to_numpy()

    if cfg.kind == "small":
        model = SmallTransformer(tok.vocab_size, max_pos=512)
    else:
        model = CLSClassifier(pretrained=cfg.pretrained, freeze=cfg.freeze)
    model.to(DEVICE)
    n_total, n_train = count_params(model)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(tr_ids) / cfg.batch_size)
    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup_ratio * steps_per_epoch * cfg.epochs), steps_per_epoch * cfg.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    lossf = nn.CrossEntropyLoss()
    rng = np.random.RandomState(cfg.seed)

    print(f"\n=== {cfg.name} | {cfg.exp} | train={len(tr_ids)} val={len(va_ids)} test={TEST_SUBSET} | "
          f"params={n_total:,} trainable={n_train:,} | max_len={cfg.max_len} ===", flush=True)
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    best = dict(val_loss=1e9, epoch=-1, state=None)
    history = []
    t_start = time.time()
    for ep in range(1, cfg.epochs + 1):
        model.train()
        t_ep = time.time()
        run_loss, n_seen = 0.0, 0
        for sel, x, m, y in make_batches(tr_ids, tr_y, cfg.batch_size, shuffle=True, rng=rng):
            def train_step():
                opt.zero_grad(set_to_none=True)
                step_loss = 0.0
                for k in range(0, len(sel), math.ceil(len(sel) / cfg.accum_steps)):   # micro-batches (accumulate gradient)
                    sl = slice(k, k + math.ceil(len(sel) / cfg.accum_steps))
                    xb, mb, yb = x[sl].to(DEVICE), m[sl].to(DEVICE), y[sl].to(DEVICE)
                    with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                        loss = lossf(model(xb, mb).float(), yb) * (len(yb) / len(sel))
                    scaler.scale(loss).backward()
                    step_loss += loss.item()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                return step_loss
            step_loss = retry_oom(train_step)
            sched.step()
            run_loss += step_loss * len(sel)
            n_seen += len(sel)
            if (n_seen // cfg.batch_size) % 100 == 0:
                print(f"    step {n_seen // cfg.batch_size}/{steps_per_epoch} loss={run_loss / n_seen:.4f} ({n_seen / (time.time() - t_ep):.0f} samples/s)", flush=True)
        ep_time = time.time() - t_ep
        vp = predict_probs(model, va_ids, va_y)
        vm = metrics_from_probs(vp, va_y)
        history.append(dict(epoch=ep, train_loss=run_loss / n_seen, val_loss=vm["loss"], val_acc=vm["accuracy"], epoch_time_s=ep_time))
        print(f"  epoch {ep}/{cfg.epochs} train_loss={run_loss / n_seen:.4f} val_loss={vm['loss']:.4f} val_acc={vm['accuracy']:.4f} ({ep_time:.0f}s)", flush=True)
        if vm["loss"] < best["val_loss"]:
            best.update(val_loss=vm["loss"], epoch=ep, state={k: v.detach().cpu().clone() for k, v in model.state_dict().items() if not (cfg.freeze and k.startswith("encoder."))})
    train_time = time.time() - t_start
    peak_mem = torch.cuda.max_memory_allocated() / 2**20 if DEVICE.type == "cuda" else float("nan")
    model.load_state_dict(best["state"], strict=False)

    # --- Đánh giá (không tính vào thời gian train) ---
    probs = {"imdb_val": predict_probs(model, va_ids, va_y)}
    t0 = time.time()
    if cfg.full_test:
        probs["imdb_test_full"] = predict_probs(model, te_ids, te_y)
        probs["imdb_test"] = probs["imdb_test_full"][sub]
    else:
        probs["imdb_test"] = predict_probs(model, [te_ids[i] for i in sub], te_y[sub])
    infer_time = time.time() - t0
    labels = {"imdb_val": va_y, "imdb_test": te_y[sub]}
    if cfg.full_test:
        labels["imdb_test_full"] = te_y
    if cfg.domain_eval:
        for dname, ddf in load_domain_sets().items():
            probs[dname] = predict_probs(model, encode(tok, ddf["text"], cfg.max_len), ddf["label"].to_numpy())
            labels[dname] = ddf["label"].to_numpy()
    demo = None
    if cfg.demo:
        d_ids = encode(tok, DEMO_SENTENCES, cfg.max_len)
        dp = predict_probs(model, d_ids, [0] * len(d_ids))
        demo = [dict(text=t, p_pos=float(p[1])) for t, p in zip(DEMO_SENTENCES, dp)]

    result = dict(
        **asdict(cfg), model=MODEL_NAME if cfg.kind == "bert" else "SmallTransformer(4L,d256,h4)", tokenizer=MODEL_NAME,
        n_train=len(tr_ids), n_val=len(va_ids), n_test=int(len(labels["imdb_test"])),
        n_params=n_total, trainable_params=n_train, train_time_s=train_time,
        time_per_epoch_s=float(np.mean([h["epoch_time_s"] for h in history])), peak_gpu_mem_mb=peak_mem,
        infer_time_test_s=infer_time, best_epoch=best["epoch"], history=history,
        hardware=hardware_string(), optimizer="AdamW", metrics={k: metrics_from_probs(probs[k], labels[k]) for k in probs}, demo_preds=demo,
    )
    res_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(OUT / f"{cfg.name}_probs.npz", **probs)
    t = result["metrics"]["imdb_test"]
    print(f"  -> TEST acc={t['accuracy']:.4f} f1={t['f1']:.4f} | train {train_time:.0f}s | peak mem {peak_mem:.0f} MB | best epoch {best['epoch']}", flush=True)
    del model, opt
    torch.cuda.empty_cache()
    return result


# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------
def build_runs() -> dict[str, RunConfig]:
    R: dict[str, RunConfig] = {}
    add = lambda c: R.__setitem__(c.name, c)
    # Core: fine-tuning BERT (dùng chung cho Exp 1, 2, 3, 4, 5 và application / error analysis)
    add(RunConfig("ft_len128_100", "core+exp1+exp2+exp3+exp4", accum_steps=4, domain_eval=True, demo=True))
    # Exp 1: from scratch
    add(RunConfig("scratch_small_len128_100", "exp1", kind="small", pretrained=False, epochs=6, lr=5e-4, batch_size=64, domain_eval=True, demo=True,
                  note="Transformer nhỏ 4 layer, d=256, random init; 6 epoch vì train nhanh và cần nhiều epoch hơn"))
    add(RunConfig("bert_random_len128_100", "exp1", pretrained=False, lr=1e-4, accum_steps=2, domain_eval=True, demo=True,
                  note="Cùng kiến trúc DistilBERT nhưng random init; lr 1e-4 (2e-5 là lr cho pretrained)"))
    # Exp 2: frozen
    add(RunConfig("frozen_len128_100", "exp2", freeze=True, lr=1e-3, domain_eval=True, demo=True, note="encoder frozen, chỉ train head; lr head 1e-3"))
    # Exp 3: data efficiency
    for f in (10, 25, 50):
        add(RunConfig(f"ft_len128_{f}", "exp3+exp4" if f == 50 else "exp3", train_frac=f / 100, accum_steps=2 if f == 50 else 4))
        add(RunConfig(f"scratch_small_len128_{f}", "exp3", kind="small", pretrained=False, epochs=6, lr=5e-4, batch_size=64, train_frac=f / 100))
        add(RunConfig(f"frozen_len128_{f}", "exp3", freeze=True, lr=1e-3, train_frac=f / 100))
    # Exp 3 control: 10% data nhưng train nhiều epoch hơn (320 bước thay vì 64) để tách "ít dữ liệu" khỏi "ít bước cập nhật"
    add(RunConfig("ft_len128_10_ep10", "exp3-control", train_frac=0.10, epochs=10, accum_steps=2, note="10% data nhưng 10 epoch (320 bước)"))
    # Exp 4: sequence length (dùng 50% train = 5,000 mẫu để tiết kiệm GPU; T=128 chính là ft_len128_50)
    add(RunConfig("ft_len64_50", "exp4", train_frac=0.5, max_len=64, accum_steps=1))
    add(RunConfig("ft_len256_50", "exp4", train_frac=0.5, max_len=256, accum_steps=8))
    return R


RUNS = build_runs()



# ----------------------------------------------------------------------------
# Benchmark chi phí theo độ dài chuỗi (Exp 4): cùng micro-batch, đo thời gian/step và peak memory
# ----------------------------------------------------------------------------
def benchmark_seq_len(lengths=(64, 128, 256, 512), batch_size: int = 4, steps: int = 10, warmup: int = 3) -> pd.DataFrame:
    """Forward+backward BERT-base trên batch toàn token thật có độ dài T (không dùng dữ liệu thật). Cùng batch_size cho mọi T."""
    rows = []
    for T in lengths:
        set_seed()
        model = CLSClassifier().to(DEVICE).train()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5)
        x = torch.randint(1000, 20000, (batch_size, T), device=DEVICE)
        m = torch.ones_like(x)
        y = torch.zeros(batch_size, dtype=torch.long, device=DEVICE)
        scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        for i in range(warmup + steps):
            if i == warmup:
                torch.cuda.synchronize(); t0 = time.time()
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                loss = nn.functional.cross_entropy(model(x, m).float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        torch.cuda.synchronize()
        rows.append(dict(max_len=T, batch_size=batch_size, sec_per_step=(time.time() - t0) / steps, peak_gpu_mem_mb=torch.cuda.max_memory_allocated() / 2**20))
        del model, opt
        torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df["time_rel_to_64"] = df["sec_per_step"] / df["sec_per_step"].iloc[0]
    df["T2_rel_to_64"] = (df["max_len"] / df["max_len"].iloc[0]) ** 2
    return df

# ----------------------------------------------------------------------------
# Collect
# ----------------------------------------------------------------------------
def collect_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, drows = [], []
    for p in sorted(OUT.glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if "metrics" not in r:
            continue
        m = r["metrics"]["imdb_test"]
        rows.append(dict(
            run=r["name"], exp=r["exp"], model=r["model"], pretrained=r["pretrained"], encoder_trainable=not r["freeze"],
            train_frac=r["train_frac"], n_train=r["n_train"], max_len=r["max_len"], epochs=r["epochs"], batch_size=r["batch_size"], lr=r["lr"],
            n_params=r["n_params"], trainable_params=r["trainable_params"], train_time_s=round(r["train_time_s"], 1),
            time_per_epoch_s=round(r["time_per_epoch_s"], 1), peak_gpu_mem_mb=round(r["peak_gpu_mem_mb"], 0), best_epoch=r["best_epoch"],
            val_acc=round(r["metrics"]["imdb_val"]["accuracy"], 4), test_acc=round(m["accuracy"], 4), test_precision=round(m["precision"], 4),
            test_recall=round(m["recall"], 4), test_f1=round(m["f1"], 4), test_loss=round(m["loss"], 4), seed=r["seed"], hardware=r["hardware"],
        ))
        for dname, dm in r["metrics"].items():
            if dname in ("imdb_val",):
                continue
            drows.append(dict(run=r["name"], eval_set=dname, n=dm["n"], accuracy=round(dm["accuracy"], 4), precision=round(dm["precision"], 4),
                              recall=round(dm["recall"], 4), f1=round(dm["f1"], 4), loss=round(dm["loss"], 4)))
    df, ddf = pd.DataFrame(rows), pd.DataFrame(drows)
    df.to_csv(HERE / "results.csv", index=False)
    ddf.to_csv(HERE / "domain_shift.csv", index=False)
    return df, ddf


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=[], help="tên run hoặc 'all'")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.list:
        for n, c in RUNS.items():
            print(f"{n:30s} {c.exp:28s} frac={c.train_frac} len={c.max_len} ep={c.epochs} lr={c.lr} {c.note}")
        return
    names = list(RUNS) if a.runs == ["all"] else a.runs
    for n in names:
        for attempt in range(1, 21):   # GPU dùng chung với tiến trình khác: nếu OOM thì chờ rồi chạy lại
            try:
                run_experiment(RUNS[n], force=a.force)
                break
            except (torch.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" not in str(e).lower():
                    raise
                print(f"[OOM] {n} attempt {attempt}: {str(e)[:80]} -> wait 60s and retry", flush=True)
                import gc; gc.collect(); torch.cuda.empty_cache(); time.sleep(60)
    if names or a.collect:
        df, ddf = collect_results()
        print(df[["run", "n_params", "train_time_s", "test_acc", "test_f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
