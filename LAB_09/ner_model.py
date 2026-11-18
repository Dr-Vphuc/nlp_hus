"""Transformer token classification cho NER (LAB 09, mục 14-16): tokenizer -> pretrained encoder -> Linear -> logits mỗi token.

Không dùng `pipeline("ner")`: ta tự tokenize (is_split_into_words=True), tự căn nhãn theo word_ids (Strategy A),
tự viết vòng huấn luyện và tự giải mã dự đoán về mức word.
"""

import json
import os
import random
import time

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

from ner_utils import (
    ID2LABEL,
    IGNORE_INDEX,
    LABEL2ID,
    LABELS,
    SEED,
    align_labels_first_subtoken,
    entity_metrics,
    tags_to_labels,
)

MODEL_NAME = "distilbert-base-cased"  # cased: viết hoa là tín hiệu rất quan trọng cho NER
MAX_LEN = 128
CKPT_DIR = "ner_checkpoint"
RESUME_PATH = "resume_state.pt"  # trạng thái huấn luyện (xoá sau khi xong)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def encode_sentence(tokenizer, words, word_tags=None, max_len=MAX_LEN):
    """Tokenize một câu đã tách từ. Trả về dict: input_ids, attention_mask, word_ids (và labels nếu có nhãn)."""
    enc = tokenizer(words, is_split_into_words=True, truncation=True, max_length=max_len)
    out = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "word_ids": enc.word_ids()}
    if word_tags is not None:
        out["labels"] = align_labels_first_subtoken(out["word_ids"], list(word_tags))
    return out


def encode_split(tokenizer, ds):
    return [encode_sentence(tokenizer, ex["tokens"], ex["ner_tags"]) for ex in ds]


def collate(batch, pad_id):
    """Padding động theo batch. labels pad = -100 (IGNORE) nên không đóng góp vào loss."""
    L = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    lab = torch.full((len(batch), L), IGNORE_INDEX, dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"])
        att[i, :n] = 1
        if "labels" in b:
            lab[i, :n] = torch.tensor(b["labels"])
    return ids, att, lab


def batches(encoded, batch_size, pad_id, shuffle, rng=None):
    idx = list(range(len(encoded)))
    if shuffle:
        rng.shuffle(idx)
    else:  # sắp theo độ dài khi suy luận để giảm padding
        idx.sort(key=lambda i: len(encoded[i]["input_ids"]))
    for s in range(0, len(idx), batch_size):
        chunk = idx[s : s + batch_size]
        yield chunk, collate([encoded[i] for i in chunk], pad_id)


@torch.no_grad()
def predict(model, tokenizer, encoded, n_words, batch_size=64, device="cpu"):
    """Suy luận. Trả về (pred_labels, word_conf): với mỗi câu, nhãn BIO theo word và xác suất của nhãn đó.

    Nhãn của word = argmax logits của subword ĐẦU TIÊN (nhất quán với Strategy A khi train).
    Word bị cắt do truncation (không còn subword nào) được gán 'O'.
    """
    model.eval()
    preds = [None] * len(encoded)
    confs = [None] * len(encoded)
    for chunk, (ids, att, _) in batches(encoded, batch_size, tokenizer.pad_token_id, shuffle=False):
        logits = model(input_ids=ids.to(device), attention_mask=att.to(device)).logits.cpu()
        probs = logits.softmax(-1)
        for row, i in enumerate(chunk):
            wp = ["O"] * n_words[i]
            wc = [1.0] * n_words[i]
            prev = None
            for t, wid in enumerate(encoded[i]["word_ids"]):
                if wid is not None and wid != prev:
                    k = int(probs[row, t].argmax())
                    wp[wid] = ID2LABEL[k]
                    wc[wid] = float(probs[row, t, k])
                prev = wid
            preds[i], confs[i] = wp, wc
    return preds, confs


def train_model(ds, epochs=3, batch_size=16, lr=5e-5, device="cpu", log_path="training_log.json", ckpt_dir=CKPT_DIR):
    set_seed()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    ).to(device)
    enc_tr = encode_split(tokenizer, ds["train"])
    enc_va = encode_split(tokenizer, ds["validation"])
    va_words = [len(ex["tokens"]) for ex in ds["validation"]]
    va_gold = [tags_to_labels(ex["ner_tags"]) for ex in ds["validation"]]

    steps_per_epoch = (len(enc_tr) + batch_size - 1) // batch_size
    total = steps_per_epoch * epochs
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)  # token IGNORE (-100) không vào loss

    # Cơ chế resume: tiến trình có thể bị kill giữa chừng (máy dùng chung CPU) -> lưu trạng thái định kỳ.
    log, best_f1, start_ep, start_step, run, n, t_prev = [], -1.0, 1, 0, 0.0, 0, 0.0
    if os.path.exists(RESUME_PATH):
        st = torch.load(RESUME_PATH, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        log, best_f1, start_ep, start_step, run, n, t_prev = st["log"], st["best_f1"], st["epoch"], st["step"], st["run"], st["n"], st["elapsed"]
        print(f"RESUME from epoch {start_ep} step {start_step}", flush=True)
    t0 = time.time() - t_prev

    def save_state(ep, step):
        tmp = RESUME_PATH + ".tmp"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "log": log, "best_f1": best_f1,
                    "epoch": ep, "step": step, "run": run, "n": n, "elapsed": time.time() - t0}, tmp)
        os.replace(tmp, RESUME_PATH)

    for ep in range(start_ep, epochs + 1):
        model.train()
        if ep != start_ep:
            run, n = 0.0, 0
        rng = random.Random(SEED + ep)  # thứ tự shuffle xác định theo epoch -> resume giữa epoch cho cùng thứ tự batch
        for step, (_, (ids, att, lab)) in enumerate(batches(enc_tr, batch_size, tokenizer.pad_token_id, True, rng), 1):
            if ep == start_ep and step <= start_step:
                continue
            logits = model(input_ids=ids.to(device), attention_mask=att.to(device)).logits  # (B, T, C)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), lab.to(device).reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            run += loss.item()
            n += 1
            if step % 100 == 0:
                print(f"epoch {ep} step {step}/{steps_per_epoch} loss {run / n:.4f} ({time.time() - t0:.0f}s)", flush=True)
                save_state(ep, step)
        pred, _ = predict(model, tokenizer, enc_va, va_words, device=device)
        f1 = entity_metrics(va_gold, pred)["overall"]["f1"]
        log.append({"epoch": ep, "train_loss": run / n, "val_f1": f1, "elapsed_s": time.time() - t0})
        print(f"== epoch {ep}: train loss {run / n:.4f}, val entity F1 {f1:.4f}", flush=True)
        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
        with open(log_path, "w") as fh:
            json.dump({"model": MODEL_NAME, "epochs": epochs, "batch_size": batch_size, "lr": lr, "log": log,
                       "train_time_s": time.time() - t0, "n_params": sum(p.numel() for p in model.parameters())}, fh, indent=1)
        start_step, run, n = 0, 0.0, 0
        save_state(ep + 1, 0)
    return log


def load_model(ckpt_dir=CKPT_DIR, device="cpu"):
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForTokenClassification.from_pretrained(ckpt_dir).to(device)
    model.eval()
    return model, tokenizer
