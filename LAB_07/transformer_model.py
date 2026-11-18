"""Model PyTorch cho LAB 07: Transformer encoder classifier vs LSTM (cùng dataset/split/tokenizer với LAB 05, 06).

Pipeline:  ids -> Embedding(d_model) [+ CLS] -> + Positional Encoding -> N x TransformerBlock (tự viết,
transformer_block.py) -> pooling (mean | cls) -> Dropout -> Linear -> sigmoid.

Cấu hình train giữ giống LAB 05/06: Adam 1e-3, batch 64, 8 epochs, clip 5.0, seed 42, chọn epoch có val loss
nhỏ nhất. Khác LAB 05/06: embedding dim = 128 (= d_model) cho CẢ LSTM lẫn Transformer để so sánh công bằng.
"""

import copy
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.nn.utils.rnn import pack_padded_sequence

import preprocessing as pp
from positional_encoding import sinusoidal_pe
from transformer_block import TransformerBlock

EMB_DIM, HIDDEN, DROPOUT = 128, 128, 0.3
D_MODEL, N_HEADS, D_FF, N_LAYERS, BLOCK_DROPOUT = 128, 4, 256, 1, 0.1
LR, BATCH_SIZE, EPOCHS, CLIP, SEED = 1e-3, 64, 8, 5.0, 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def length_mask(lengths, T):
    """[B, T] bool, True = token thật (không phải <pad>)."""
    return torch.arange(T, device=lengths.device)[None, :] < lengths[:, None]


class LSTMClassifier(nn.Module):
    """Embedding -> LSTM -> h_T -> Linear (Model A)."""

    def __init__(self, vocab_size, emb_dim=EMB_DIM, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pp.PAD)
        self.rnn = nn.LSTM(emb_dim, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, ids, lengths):
        packed = pack_padded_sequence(self.embedding(ids), lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.rnn(packed)
        return self.fc(self.dropout(h_n[-1])).squeeze(-1)


class TransformerClassifier(nn.Module):
    """Embedding -> (+CLS) -> + PE -> N x TransformerBlock -> pooling -> Linear (Model B).

    pooling = "mean": trung bình H_t trên các token thật.  pooling = "cls": thêm một vector [CLS] học được
    ở vị trí 0 và dùng H_0 làm biểu diễn câu.  use_pe=False: bỏ positional encoding (ablation).
    """

    def __init__(self, vocab_size, d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF, n_layers=N_LAYERS,
                 block_dropout=BLOCK_DROPOUT, dropout=DROPOUT, pooling="mean", use_pe=True, max_len=600):
        super().__init__()
        assert pooling in ("mean", "cls")
        self.pooling, self.use_pe = pooling, use_pe
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pp.PAD)
        self.cls = nn.Parameter(torch.randn(d_model) * 0.02) if pooling == "cls" else None
        self.register_buffer("pe", torch.tensor(sinusoidal_pe(max_len + 1, d_model), dtype=torch.float32))
        self.blocks = nn.ModuleList(TransformerBlock(d_model, n_heads, d_ff, block_dropout) for _ in range(n_layers))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, ids, lengths, return_attention=False):
        lengths = lengths.to(ids.device)
        X = self.embedding(ids)                                               # [B, T, D]
        mask = length_mask(lengths, ids.size(1))
        if self.pooling == "cls":
            X = torch.cat([self.cls.expand(X.size(0), 1, -1), X], dim=1)      # [B, T+1, D]
            mask = torch.cat([torch.ones_like(mask[:, :1]), mask], dim=1)
        if self.use_pe:
            X = X + self.pe[: X.size(1)]                                      # X' = X + PE
        attns = []
        for block in self.blocks:
            X, A = block(X, key_padding_mask=mask, return_attention=True)     # A: [B, h, T, T]
            attns.append(A)
        if self.pooling == "cls":
            rep = X[:, 0]
        else:
            m = mask.unsqueeze(-1).to(X.dtype)
            rep = (X * m).sum(1) / m.sum(1)                                   # mean pooling trên token thật
        logit = self.fc(self.dropout(rep)).squeeze(-1)
        return (logit, attns) if return_attention else logit


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def attention_memory_mb(T, n_heads, batch=BATCH_SIZE, n_layers=N_LAYERS, bytes_per=4):
    """Bộ nhớ lý thuyết của tensor attention weights [B, h, T, T] float32 (một tensor, mọi layer)."""
    return batch * n_heads * T * T * bytes_per * n_layers / 2**20


@torch.no_grad()
def predict_logits(model, ids, lengths, batch_size=256):
    model.eval()
    out = []
    for i in range(0, len(ids), batch_size):
        b_ids = torch.as_tensor(ids[i:i + batch_size], device=DEVICE)
        b_len = torch.as_tensor(lengths[i:i + batch_size])
        out.append(model(b_ids, b_len).float().cpu())
    return torch.cat(out).numpy()


def bce(logits, y):
    return nn.functional.binary_cross_entropy_with_logits(
        torch.as_tensor(logits), torch.as_tensor(y, dtype=torch.float32)).item()


def evaluate(y_true, y_pred):
    return {"accuracy": accuracy_score(y_true, y_pred), "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0), "f1": f1_score(y_true, y_pred, zero_division=0)}


def train_model(name, model, data, y_train, y_val, y_test, max_len, epochs=EPOCHS, verbose=True, seed=SEED):
    """data = {'train'|'val'|'test': (ids, lengths)}. Trả về dict model/history/row/proba/y_pred."""
    set_seed(seed)
    loader = pp.make_loader(*data["train"], y_train, batch_size=BATCH_SIZE, shuffle=True, seed=seed)
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()
    history, best, train_time = [], None, 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        total, n = 0.0, 0
        for ids, lengths, y in loader:
            ids, y = ids.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(ids, lengths), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            total += loss.item() * len(y)
            n += len(y)
        et = time.perf_counter() - t0
        train_time += et
        vl = predict_logits(model, *data["val"])
        vp = (vl > 0).astype(int)
        history.append({"epoch": epoch, "train_loss": total / n, "val_loss": bce(vl, y_val),
                        "val_acc": accuracy_score(y_val, vp), "val_f1": f1_score(y_val, vp), "epoch_time_s": et})
        if best is None or history[-1]["val_loss"] < best["val_loss"]:
            best = {**history[-1], "state": copy.deepcopy(model.state_dict())}
        if verbose:
            h = history[-1]
            print(f"  [{name}] epoch {epoch}: train {h['train_loss']:.4f} | val {h['val_loss']:.4f} | "
                  f"val acc {h['val_acc']:.4f} | {et:.1f}s", flush=True)
    model.load_state_dict(best["state"])
    tl = predict_logits(model, *data["test"])
    yp = (tl > 0).astype(int)
    row = {"model": name, "max_len": max_len, "n_params": count_params(model), "epochs": epochs,
           "best_epoch": best["epoch"], "train_time_s": train_time, "time_per_epoch_s": train_time / epochs,
           "best_val_loss": best["val_loss"], "val_f1": best["val_f1"], "test_loss": bce(tl, y_test),
           **evaluate(y_test, yp)}
    return {"model": model, "history": pd.DataFrame(history), "row": row, "proba": sigmoid(tl), "y_pred": yp}
