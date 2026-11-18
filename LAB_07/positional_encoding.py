"""Sinusoidal positional encoding + Experiment 1 / permutation experiment (LAB 07).

PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
X' = X + PE

Chạy `python positional_encoding.py`:
  - in PE cho d=4, pos=0,1,2 (đối chiếu phần tính tay)
  - chạy permutation experiment (dog bites man / man bites dog) và "I love NLP" / "NLP love I"
  - vẽ positional_encoding.png
"""

import numpy as np
import torch

from transformer_block import MultiHeadSelfAttention, TransformerBlock


def sinusoidal_pe(max_len, d):
    """[max_len, d]. d phải chẵn."""
    assert d % 2 == 0
    pos = np.arange(max_len)[:, None]                         # [max_len, 1]
    two_i = np.arange(0, d, 2)[None, :]                       # [1, d/2] = 2i
    angle = pos / np.power(10000.0, two_i / d)
    pe = np.zeros((max_len, d))
    pe[:, 0::2] = np.sin(angle)
    pe[:, 1::2] = np.cos(angle)
    return pe


def cosine(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def permutation_experiment(tokens_1, tokens_2, d_model=16, n_heads=2, seed=0, use_block=False):
    """So sánh biểu diễn của 2 câu cùng tập token, khác thứ tự (tokens_2 là hoán vị của tokens_1).

    Embedding cố định theo từ (cùng từ -> cùng vector, bất kể vị trí); trọng số attention ngẫu nhiên
    cố định (seed). Experiment A: không PE. Experiment B: có PE.
    Trả về dict các độ đo cho A và B.
    """
    torch.manual_seed(seed)
    vocab = sorted(set(tokens_1) | set(tokens_2))
    emb = {w: torch.randn(d_model) for w in vocab}
    layer = (TransformerBlock(d_model, n_heads, 2 * d_model, dropout=0.0) if use_block
             else MultiHeadSelfAttention(d_model, n_heads)).eval()
    pe = torch.tensor(sinusoidal_pe(max(len(tokens_1), len(tokens_2)), d_model), dtype=torch.float32)

    def run(tokens, with_pe):
        x = torch.stack([emb[w] for w in tokens])[None]                  # [1, T, d]
        if with_pe:
            x = x + pe[: len(tokens)][None]
        with torch.no_grad():
            out = layer(x)
        return out[0].numpy()                                            # [T, d]

    res = {}
    for name, with_pe in (("A (no PE)", False), ("B (with PE)", True)):
        o1, o2 = run(tokens_1, with_pe), run(tokens_2, with_pe)
        # vị trí của từng từ trong câu 2 để so cùng một từ giữa hai câu
        per_word = {w: float(np.abs(o1[tokens_1.index(w)] - o2[tokens_2.index(w)]).max()) for w in vocab}
        res[name] = {
            "pooled_cosine": cosine(o1.mean(0), o2.mean(0)),
            "pooled_max_abs_diff": float(np.abs(o1.mean(0) - o2.mean(0)).max()),
            "per_word_max_abs_diff": per_word,
            "out_1": o1, "out_2": o2,
        }
    return res


def plot_positional_encoding(path="positional_encoding.png", max_len=50, d=64):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pe = sinusoidal_pe(max_len, d)
    norm = pe / np.linalg.norm(pe, axis=1, keepdims=True)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    im = ax[0].imshow(pe, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
    ax[0].set(title=f"PE matrix (max_len={max_len}, d={d})", xlabel="dimension", ylabel="position")
    plt.colorbar(im, ax=ax[0])
    for dim in (0, 1, 8, 9, 32, 33):
        ax[1].plot(pe[:, dim], label=f"dim {dim} ({'sin' if dim % 2 == 0 else 'cos'})")
    ax[1].set(title="PE theo position cho vài dimension", xlabel="position", ylabel="value")
    ax[1].legend(fontsize=7, ncol=2)
    sim = norm @ norm.T
    im = ax[2].imshow(sim, cmap="viridis", vmin=-0.2, vmax=1)
    ax[2].set(title="Cosine similarity giữa các position", xlabel="position", ylabel="position")
    plt.colorbar(im, ax=ax[2])
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    print("PE (d=4), pos = 0, 1, 2:")
    print(sinusoidal_pe(3, 4))

    for t1, t2 in ((["dog", "bites", "man"], ["man", "bites", "dog"]),
                   (["I", "love", "NLP"], ["NLP", "love", "I"])):
        for use_block in (False, True):
            res = permutation_experiment(t1, t2, use_block=use_block)
            print(f"\n=== {' '.join(t1)}  vs  {' '.join(t2)}  ({'full TransformerBlock' if use_block else 'MH self-attention'}) ===")
            for k, r in res.items():
                print(f"{k:<12s} pooled cosine = {r['pooled_cosine']:.6f} | pooled max|diff| = {r['pooled_max_abs_diff']:.2e}"
                      f" | per-word max|diff| = { {w: f'{v:.2e}' for w, v in r['per_word_max_abs_diff'].items()} }")
    plot_positional_encoding()
    print("\nSaved positional_encoding.png")
