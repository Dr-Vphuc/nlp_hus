"""Attention từ đầu bằng NumPy — LAB 06.

Gồm: softmax ổn định số (có temperature), scaled dot-product attention, self-attention,
entropy. Không dùng torch.nn.MultiheadAttention / keras.layers.Attention.
Chạy `python attention_numpy.py` để chạy các kiểm tra (kết quả để ĐỐI CHIẾU với phần tính tay).
"""

import numpy as np


def softmax(x, axis=-1, temperature=1.0):
    """softmax(x / T) ổn định số: trừ max trước khi exp.

    exp(x_i - m) / sum_j exp(x_j - m) = exp(x_i) / sum_j exp(x_j) vì thừa số exp(-m)
    xuất hiện ở cả tử và mẫu nên triệt tiêu; còn exp(x - m) <= 1 nên không bị overflow.
    """
    x = np.asarray(x, dtype=np.float64) / temperature
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def naive_softmax(x):
    """Softmax không xử lý số học — chỉ để minh họa overflow."""
    e = np.exp(np.asarray(x, dtype=np.float64))
    return e / e.sum(axis=-1, keepdims=True)


def entropy(p, axis=-1):
    """Entropy (nats) của phân phối p; 0*log0 được coi là 0."""
    p = np.asarray(p, dtype=np.float64)
    return -np.sum(np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0), axis=axis)


def scaled_dot_product_attention(Q, K, V, mask=None, scale=True, temperature=1.0):
    """Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V.

    Q: [..., T_q, d_k]; K: [..., T_k, d_k]; V: [..., T_k, d_v].
    mask: bool [..., T_q, T_k] (True = được phép attend); vị trí False nhận weight 0.
    scale=False -> dot-product attention thường (không chia sqrt(d_k)).
    Trả về (scores, attention_weights, context).
    """
    Q, K, V = (np.asarray(a, dtype=np.float64) for a in (Q, K, V))
    d_k = K.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2)
    if scale:
        scores = scores / np.sqrt(d_k)
    masked = scores if mask is None else np.where(mask, scores, -np.inf)
    weights = softmax(masked, axis=-1, temperature=temperature)
    context = weights @ V
    return scores, weights, context


def self_attention(X, W_Q, W_K, W_V, mask=None):
    """Self-attention: Q = X W_Q, K = X W_K, V = X W_V rồi scaled dot-product attention.

    Trả về (Q, K, V, scores, weights, context).
    """
    X = np.asarray(X, dtype=np.float64)
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V
    scores, weights, context = scaled_dot_product_attention(Q, K, V, mask=mask)
    return Q, K, V, scores, weights, context


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    print("== Bài 1: scores, q=[1,2], K=[[1,0],[0,1],[1,1]]")
    q, K = np.array([1, 2]), np.array([[1, 0], [0, 1], [1, 1]])
    print(K @ q)

    print("== Bài 2: softmax([1,2,3])")
    a = softmax([1, 2, 3]); print(a, a.sum())

    print("== Bài 3: context, alpha=[.1,.2,.7]")
    V = np.array([[1, 0], [0, 1], [2, 2]])
    print(np.array([0.1, 0.2, 0.7]) @ V)

    print("== Bài 4: full attention, q=[1,0]")
    q = np.array([[1, 0]]); K = np.array([[1, 0], [0, 1], [1, 1]]); V = np.array([[1, 0], [0, 2], [2, 1]])
    s, w, c = scaled_dot_product_attention(q, K, V, scale=False)
    print("scores", s, "weights", w, "context", c)
    s, w, c = scaled_dot_product_attention(q, K, V)
    print("(scaled) scores", s, "weights", w, "context", c)

    print("== Scaling: d_k=4, QK^T=[4,2,0]")
    s = np.array([4.0, 2.0, 0.0])
    print("no scale", softmax(s), "max", softmax(s).max())
    print("scaled  ", softmax(s / 2), "max", softmax(s / 2).max())

    print("== Temperature, s=[4,2,0]")
    for T in (0.5, 1.0, 2.0):
        p = softmax(s, temperature=T)
        print(T, p, "max", round(p.max(), 4), "entropy", round(float(entropy(p)), 4))

    print("== Numerical stability, x=[1000,1001,1002]")
    with np.errstate(all="ignore"):
        print("naive ", naive_softmax([1000, 1001, 1002]))
    print("stable", softmax([1000, 1001, 1002]), "== softmax([0,1,2])", softmax([0, 1, 2]))

    print("== Self-attention (W_Q/W_K/W_V thật sự do đề cung cấp -> điền vào đây)")
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # I, like, cats (ví dụ, thay bằng ma trận đề cho)
    W = np.eye(2)
    _, _, _, S, A, C = self_attention(X, W, W, W)
    print("scores", S, "A", A, "C", C, sep="\n")

    print("== Kiểm tra với PyTorch (chỉ để verify)")
    try:
        import torch
        import torch.nn.functional as F
        rng = np.random.default_rng(0)
        Q, K, V = (rng.normal(size=(5, 8)) for _ in range(3))
        _, _, c = scaled_dot_product_attention(Q, K, V)
        ref = F.scaled_dot_product_attention(*(torch.tensor(a) for a in (Q, K, V))).numpy()
        print("max |diff| =", np.abs(c - ref).max())
    except ImportError:
        print("torch không có, bỏ qua")
