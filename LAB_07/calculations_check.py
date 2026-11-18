"""Đối chiếu kết quả tính tay trong calculations.md (NumPy thuần, không dùng lớp Transformer nào).

Chạy: python calculations_check.py
"""

import numpy as np

from positional_encoding import sinusoidal_pe

np.set_printoptions(precision=4, suppress=True)


def softmax(s):
    e = np.exp(s - s.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


# --- Bài 6: một attention head, X in R^{3x4}, d_k = 2 (đề không in ma trận => tự chọn ma trận đơn giản) ---
X = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]], float)
W_Q = np.array([[1, 0], [0, 1], [0, 0], [0, 0]], float)
W_K = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], float)
W_V = np.array([[1, 1], [0, 1], [1, 0], [0, 0]], float)
Q, K, V = X @ W_Q, X @ W_K, X @ W_V
S = Q @ K.T / np.sqrt(2)
A = softmax(S)
Z = A @ V
print("Q =\n", Q, "\nK =\n", K, "\nV =\n", V)
print("QK^T =\n", Q @ K.T, "\nS = QK^T/sqrt(2) =\n", S, "\nA =\n", A, "\nrow sums", A.sum(1), "\nZ = AV =\n", Z)

# --- Bài 7: multi-head, hai head dimension 2 -> concat -> T x 4 -> W_O ---
head1, head2 = Z, np.array([[1, 0], [0, 1], [1, 1]], float)       # head 2: ví dụ minh hoạ
concat = np.concatenate([head1, head2], axis=1)
print("\nConcat shape:", concat.shape)

# --- Bài 10: positional encoding d = 4, pos = 0,1,2 ---
print("\nPE (d=4):\n", sinusoidal_pe(3, 4))
pe = sinusoidal_pe(3, 4)
for i in range(3):
    for j in range(i + 1, 3):
        print(f"|PE{i} - PE{j}| = {np.linalg.norm(pe[i] - pe[j]):.4f}")

# --- Bài 14: FFN ---
x = np.array([1.0, 2.0])
W1, b1 = np.eye(2), np.array([0.0, -1.0])
h = np.maximum(0, W1 @ x + b1)
W2, b2 = np.array([[2.0, 0.0], [1.0, 1.0]]), np.array([0.0, 1.0])     # đề không cho W2, b2 -> tự chọn
y = W2 @ h + b2
print("\nW1 x + b1 =", W1 @ x + b1, "| h = ReLU(.) =", h, "| W2 h =", W2 @ h, "| y =", y)

# --- Residual + LayerNorm ví dụ ---
v = x + np.array([0.5, 0.5])                                          # x + F(x) với F(x) ví dụ
ln = (v - v.mean()) / np.sqrt(v.var() + 1e-5)
print("\nx + F(x) =", v, "| LayerNorm (gamma=1, beta=0) =", ln)

# --- Shape (B=8, T=100, d_model=128, h=4) và chi phí T^2 ---
B, T, D, hh = 8, 100, 128, 4
print("\nd_k =", D // hh, "| scores shape =", (B, hh, T, T), "| #scores =", B * hh * T * T)
for T in (32, 64, 128, 256):
    print(f"T={T}: T^2={T*T:,} phần tử / head, float32 = {T*T*4/1024:.1f} KiB / head / sample")
