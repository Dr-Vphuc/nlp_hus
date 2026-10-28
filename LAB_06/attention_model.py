"""Model PyTorch cho LAB 06: LSTM (lấy lại từ LAB 05) vs Embedding -> Self-Attention -> classifier.

Cấu hình train giữ giống LAB 05 (emb 100, hidden 128, dropout 0.3, Adam 1e-3, batch 64,
8 epochs, clip 5.0, seed 42, chọn epoch có val loss nhỏ nhất) để so sánh công bằng.
Phần core NumPy nằm ở attention_numpy.py; ở đây dùng torch chỉ để TRAIN trên IMDb.
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

EMB_DIM, HIDDEN, DROPOUT = 100, 128, 0.3
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
    """Embedding -> LSTM -> h_T -> Linear (giống SeqClassifier('LSTM') của LAB 05)."""

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


class AttentionClassifier(nn.Module):
    """Embedding -> Self-Attention (Q=XW_Q, K=XW_K, V=XW_V) -> pooling bằng learned query -> Linear.

    - Self-attention: mỗi token nhìn mọi token thật khác (mask <pad>) -> biểu diễn theo ngữ cảnh.
    - Pooling: một query học được q* attend lên các biểu diễn đó -> context vector c = sum_i a_i h_i.
      a_i chính là "token nào quyết định sentiment" để visualize.
    - KHÔNG có positional encoding (để dành LAB 07) -> model không biết thứ tự từ.
    """

    def __init__(self, vocab_size, emb_dim=EMB_DIM, d_model=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pp.PAD)
        self.W_q = nn.Linear(emb_dim, d_model, bias=False)
        self.W_k = nn.Linear(emb_dim, d_model, bias=False)
        self.W_v = nn.Linear(emb_dim, d_model, bias=False)
        self.pool_query = nn.Parameter(torch.randn(d_model) / math.sqrt(d_model))
        self.pool_key = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, ids, lengths, return_attention=False):
        T = ids.size(1)
        mask = length_mask(lengths.to(ids.device), T)                     # [B, T]
        X = self.embedding(ids)                                           # [B, T, E]
        Q, K, V = self.W_q(X), self.W_k(X), self.W_v(X)
        scores = Q @ K.transpose(-1, -2) / math.sqrt(self.d_model)        # [B, T, T]
        scores = scores.masked_fill(~mask[:, None, :], float("-inf"))
        A = torch.softmax(scores, dim=-1)                                 # self-attention matrix
        H = A @ V                                                         # [B, T, d]
        pscores = self.pool_key(H) @ self.pool_query / math.sqrt(self.d_model)   # [B, T]
        pscores = pscores.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(pscores, dim=-1)                            # [B, T]
        c = (alpha.unsqueeze(-1) * H).sum(1)                              # context vector [B, d]
        logit = self.fc(self.dropout(c)).squeeze(-1)
        return (logit, A, alpha) if return_attention else logit


def count_params(model):
    return sum(p.numel() for p in model.parameters())


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


def train_model(kind, vocab_size, data, y_train, y_val, y_test, max_len, epochs=EPOCHS, verbose=True):
    """kind in {'LSTM', 'Attention'}; data = {'train'|'val'|'test': (ids, lengths)}."""
    set_seed()
    loader = pp.make_loader(*data["train"], y_train, batch_size=BATCH_SIZE, shuffle=True)
    model = (LSTMClassifier if kind == "LSTM" else AttentionClassifier)(vocab_size).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
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
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
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
            print(f"  [{kind}] epoch {epoch}: train {h['train_loss']:.4f} | val {h['val_loss']:.4f} | "
                  f"val acc {h['val_acc']:.4f} | {et:.1f}s")
    peak = (torch.cuda.max_memory_allocated() - mem_before) / 2**20 if DEVICE.type == "cuda" else np.nan
    model.load_state_dict(best["state"])
    tl = predict_logits(model, *data["test"])
    yp = (tl > 0).astype(int)
    row = {"model": kind, "max_len": max_len, "n_params": count_params(model), "epochs": epochs,
           "best_epoch": best["epoch"], "train_time_s": train_time, "time_per_epoch_s": train_time / epochs,
           "peak_gpu_mem_mb": peak, "best_val_loss": best["val_loss"], "val_f1": best["val_f1"],
           "test_loss": bce(tl, y_test), **evaluate(y_test, yp)}
    return {"model": model, "history": pd.DataFrame(history), "row": row, "proba": sigmoid(tl), "y_pred": yp}
