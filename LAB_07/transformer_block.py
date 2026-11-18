"""Core LAB 07: Transformer Encoder block tối giản, tự ghép từ primitive (Linear, softmax, LayerNorm, Dropout).

KHÔNG dùng nn.TransformerEncoderLayer / nn.MultiheadAttention trong phần core.
(nn.TransformerEncoderLayer chỉ được dùng ở __main__ để đối chiếu kết quả - đúng tinh thần đề bài.)

Kiến trúc (post-LN, như paper "Attention Is All You Need"):

    x -> MultiHeadSelfAttention -> Add & LayerNorm -> FeedForward -> Add & LayerNorm -> output

Shape (B = batch, T = số token, D = d_model, h = n_heads, d_k = D / h):
    x            [B, T, D]
    Q, K, V      [B, h, T, d_k]   (Linear D->D rồi tách thành h head)
    scores       [B, h, T, T]     (Q K^T / sqrt(d_k))
    attn output  [B, h, T, d_k]   (A V) -> concat các head -> [B, T, D] -> W_O -> [B, T, D]
    FFN          [B, T, D] -> [B, T, d_ff] -> [B, T, D]
    output       [B, T, D]

Chạy `python transformer_block.py` để in bảng shape (B=8, T=100, D=128, h=4) và kiểm tra
tương đương với torch.nn.TransformerEncoderLayer.
"""

import math

import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    """MultiHead(X) = Concat(head_1..head_h) W_O, head_i = softmax(Q_i K_i^T / sqrt(d_k)) V_i,
    với Q_i = X W_Q^i, ... Triển khai gộp: một Linear D->D cho Q (rồi cắt thành h phần d_k)."""

    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model phải chia hết cho n_heads"
        self.d_model, self.n_heads, self.d_k = d_model, n_heads, d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def split_heads(self, t):
        B, T, _ = t.shape
        return t.view(B, T, self.n_heads, self.d_k).transpose(1, 2)        # [B, h, T, d_k]

    def forward(self, x, key_padding_mask=None, return_attention=False, shapes=None):
        """key_padding_mask: [B, T] bool, True = token thật, False = <pad> (bị mask khỏi key)."""
        B, T, _ = x.shape
        Q, K, V = (self.split_heads(l(x)) for l in (self.W_q, self.W_k, self.W_v))
        scores = Q @ K.transpose(-1, -2) / math.sqrt(self.d_k)              # [B, h, T, T]
        if key_padding_mask is not None:
            scores = scores.masked_fill(~key_padding_mask[:, None, None, :], float("-inf"))
        A = torch.softmax(scores, dim=-1)                                   # softmax theo hàng (key)
        heads = self.attn_dropout(A) @ V                                    # [B, h, T, d_k]
        concat = heads.transpose(1, 2).reshape(B, T, self.d_model)          # Concat(head_1..head_h) [B, T, D]
        out = self.W_o(concat)                                              # [B, T, D]
        if shapes is not None:
            shapes.update({"Q": Q.shape, "K": K.shape, "V": V.shape, "Attention scores": scores.shape,
                           "Attention weights A": A.shape, "Head outputs (A V)": heads.shape,
                           "Concat heads": concat.shape, "Attention output (after W_O)": out.shape})
        return (out, A) if return_attention else out


class FeedForward(nn.Module):
    """Position-wise FFN(x) = W2 act(W1 x + b1) + b2, áp dụng độc lập cho từng vị trí."""

    def __init__(self, d_model, d_ff, dropout=0.0, activation="relu"):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(d_model, d_ff), nn.Linear(d_ff, d_model)
        self.act = nn.ReLU() if activation == "relu" else nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Transformer Encoder block: MHSA -> Add&Norm -> FFN -> Add&Norm."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, activation="relu"):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout, activation)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None, return_attention=False, shapes=None):
        attn_out, A = self.attn(x, key_padding_mask, return_attention=True, shapes=shapes)
        x = self.norm1(x + self.drop(attn_out))        # Residual + LayerNorm (sau attention)
        ffn_out = self.ffn(x)
        out = self.norm2(x + self.drop(ffn_out))       # Residual + LayerNorm (sau FFN)
        if shapes is not None:
            shapes.update({"FFN (hidden d_ff)": (*x.shape[:2], self.ffn.fc1.out_features),
                           "FFN output": ffn_out.shape, "Output": out.shape})
        return (out, A) if return_attention else out


def shape_table(B=8, T=100, d_model=128, n_heads=4, d_ff=512):
    """Chạy 1 forward và trả về dict {tên tensor: shape} để đối chiếu phần tính shape."""
    block = TransformerBlock(d_model, n_heads, d_ff).eval()
    shapes = {"X": (B, T, d_model)}
    block(torch.randn(B, T, d_model), shapes=shapes)
    return {k: tuple(v) for k, v in shapes.items()}


def check_against_torch(d_model=32, n_heads=4, d_ff=64, B=3, T=7, seed=0):
    """Chép trọng số của block tự viết sang nn.TransformerEncoderLayer (post-LN, ReLU) rồi so output."""
    torch.manual_seed(seed)
    mine = TransformerBlock(d_model, n_heads, d_ff, dropout=0.0).eval()
    ref = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout=0.0, batch_first=True,
                                     norm_first=False, activation="relu").eval()
    with torch.no_grad():
        ref.self_attn.in_proj_weight.copy_(torch.cat([mine.attn.W_q.weight, mine.attn.W_k.weight, mine.attn.W_v.weight]))
        ref.self_attn.in_proj_bias.copy_(torch.cat([mine.attn.W_q.bias, mine.attn.W_k.bias, mine.attn.W_v.bias]))
        ref.self_attn.out_proj.load_state_dict(mine.attn.W_o.state_dict())
        ref.linear1.load_state_dict(mine.ffn.fc1.state_dict())
        ref.linear2.load_state_dict(mine.ffn.fc2.state_dict())
        ref.norm1.load_state_dict(mine.norm1.state_dict())
        ref.norm2.load_state_dict(mine.norm2.state_dict())
    x = torch.randn(B, T, d_model)
    lengths = torch.tensor([7, 5, 3])
    mask = torch.arange(T)[None, :] < lengths[:, None]
    with torch.no_grad():
        out_mine = mine(x, key_padding_mask=mask)
        out_ref = ref(x, src_key_padding_mask=~mask)
    real = mask[..., None]       # chỉ so các vị trí thật (vị trí <pad> không quan trọng)
    return (out_mine - out_ref).abs().masked_select(real).max().item()


if __name__ == "__main__":
    print("== Shape check: B=8, T=100, d_model=128, n_heads=4 (d_k = 32), d_ff=512 ==")
    for name, shp in shape_table().items():
        print(f"{name:<32s} {shp}")
    block = TransformerBlock(128, 4, 512)
    print("\nSố tham số của 1 block:", sum(p.numel() for p in block.parameters()))
    print("  MHSA:", sum(p.numel() for p in block.attn.parameters()),
          "| FFN:", sum(p.numel() for p in block.ffn.parameters()),
          "| 2xLayerNorm:", sum(p.numel() for p in block.norm1.parameters()) * 2)
    diff = check_against_torch()
    print(f"\nMax |mine - nn.TransformerEncoderLayer| (có padding mask) = {diff:.2e}")
    assert diff < 1e-5
    # weights của mỗi hàng softmax cộng bằng 1, token <pad> nhận weight 0
    x = torch.randn(2, 6, 16)
    m = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    _, A = TransformerBlock(16, 2, 32).eval()(x, m, return_attention=True)
    print("Sum of each attention row == 1:", torch.allclose(A.sum(-1), torch.ones(2, 2, 6)))
    print("Attention weight onto <pad> keys:", A[0, :, :, 4:].abs().max().item())
