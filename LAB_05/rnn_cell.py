"""Simple RNN cell tự implement bằng NumPy — LAB 05 §14.

    a_t = W_xh x_t + W_hh h_{t-1} + b_h        (pre-activation)
    h_t = tanh(a_t)

Computation graph của một bước:

    x_t ──► W_xh x_t ──┐
                       ├──► (+) ──► + b_h ──► tanh ──► h_t
    h_{t-1} ─► W_hh h_{t-1} ┘

Cùng một bộ (W_xh, W_hh, b_h) được dùng lại ở MỌI timestep (weight sharing theo thời gian);
h_t là "bộ nhớ" tóm tắt x_1..x_t và được truyền sang bước t+1.

Quy ước shape giống PyTorch: W_xh (hidden, input), W_hh (hidden, hidden).
x_t có thể là một vector (input,) hoặc một batch (batch, input).

Chạy `python rnn_cell.py` để:
  1. test cell với một sequence nhỏ, in từng h_t;
  2. so sánh với torch.nn.RNNCell và torch.nn.RNN (copy cùng trọng số).
"""

import numpy as np


class SimpleRNNCell:
    def __init__(self, input_size, hidden_size, seed=None):
        # Khởi tạo giống PyTorch: U(-1/sqrt(hidden), 1/sqrt(hidden))
        rng = np.random.default_rng(seed)
        k = 1.0 / np.sqrt(hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_xh = rng.uniform(-k, k, size=(hidden_size, input_size))
        self.W_hh = rng.uniform(-k, k, size=(hidden_size, hidden_size))
        self.b_h = rng.uniform(-k, k, size=hidden_size)

    @classmethod
    def from_weights(cls, W_xh, W_hh, b_h=None):
        """Tạo cell từ trọng số cho trước (vd. copy từ PyTorch hoặc số liệu bài tập)."""
        W_xh, W_hh = np.asarray(W_xh, dtype=float), np.asarray(W_hh, dtype=float)
        cell = cls.__new__(cls)
        cell.hidden_size, cell.input_size = W_xh.shape
        cell.W_xh, cell.W_hh = W_xh, W_hh
        cell.b_h = np.zeros(cell.hidden_size) if b_h is None else np.asarray(b_h, dtype=float)
        return cell

    def init_hidden(self, batch_size=None):
        shape = (self.hidden_size,) if batch_size is None else (batch_size, self.hidden_size)
        return np.zeros(shape)

    def preactivation(self, x_t, h_prev):
        """a_t = W_xh x_t + W_hh h_{t-1} + b_h. Viết dạng x @ W.T để chạy cả vector lẫn batch."""
        return x_t @ self.W_xh.T + h_prev @ self.W_hh.T + self.b_h

    def forward(self, x_t, h_prev):
        """Một bước RNN: h_t = tanh(a_t)."""
        return np.tanh(self.preactivation(x_t, h_prev))

    def forward_sequence(self, xs, h0=None):
        """Unroll qua thời gian. xs: (T, input) hoặc (T, batch, input). Trả về mọi h_t: (T, ...)."""
        h = self.init_hidden(None if xs.ndim == 2 else xs.shape[1]) if h0 is None else h0
        hs = []
        for x_t in xs:
            h = self.forward(x_t, h)   # h_t phụ thuộc h_{t-1} -> phải tính tuần tự
            hs.append(h)
        return np.stack(hs)

    def num_parameters(self):
        return self.W_xh.size + self.W_hh.size + self.b_h.size


# ----------------------------------------------------------------------------
# Kiểm tra với PyTorch
# ----------------------------------------------------------------------------
def load_into_torch(cell, torch_module):
    """Copy (W_xh, W_hh, b_h) vào nn.RNNCell / nn.RNN.

    PyTorch có HAI bias: h_t = tanh(W_ih x_t + b_ih + W_hh h_{t-1} + b_hh),
    nên đặt b_ih = b_h và b_hh = 0 thì công thức trùng với b_h của mình.
    """
    import torch

    suffix = "_l0" if isinstance(torch_module, torch.nn.RNN) else ""
    with torch.no_grad():
        getattr(torch_module, f"weight_ih{suffix}").copy_(torch.from_numpy(cell.W_xh))
        getattr(torch_module, f"weight_hh{suffix}").copy_(torch.from_numpy(cell.W_hh))
        getattr(torch_module, f"bias_ih{suffix}").copy_(torch.from_numpy(cell.b_h))
        getattr(torch_module, f"bias_hh{suffix}").zero_()


def from_torch(torch_cell):
    """Chiều ngược lại: lấy trọng số của nn.RNNCell, gộp hai bias thành b_h = b_ih + b_hh."""
    W_xh = torch_cell.weight_ih.detach().double().numpy()
    W_hh = torch_cell.weight_hh.detach().double().numpy()
    b_h = (torch_cell.bias_ih + torch_cell.bias_hh).detach().double().numpy()
    return SimpleRNNCell.from_weights(W_xh, W_hh, b_h)


def compare_with_torch(input_size=3, hidden_size=2, seq_len=4, batch_size=2, seed=0, verbose=True):
    import torch

    rng = np.random.default_rng(seed)
    xs = rng.normal(size=(seq_len, batch_size, input_size))
    h0 = rng.normal(size=(batch_size, hidden_size)) * 0.5

    # 1) Cell của mình -> copy sang PyTorch
    cell = SimpleRNNCell(input_size, hidden_size, seed=seed)
    hs_ours = cell.forward_sequence(xs, h0)

    t_cell = torch.nn.RNNCell(input_size, hidden_size, nonlinearity="tanh").double()
    load_into_torch(cell, t_cell)
    h = torch.from_numpy(h0)
    hs_cell = []
    with torch.no_grad():
        for x_t in torch.from_numpy(xs):
            h = t_cell(x_t, h)
            hs_cell.append(h)
    hs_cell = torch.stack(hs_cell).numpy()

    t_rnn = torch.nn.RNN(input_size, hidden_size, nonlinearity="tanh").double()  # input (T, B, in)
    load_into_torch(cell, t_rnn)
    with torch.no_grad():
        out, h_last = t_rnn(torch.from_numpy(xs), torch.from_numpy(h0)[None])
    hs_rnn = out.numpy()

    # 2) PyTorch (khởi tạo ngẫu nhiên, 2 bias) -> copy sang cell của mình
    torch.manual_seed(seed)
    t_cell2 = torch.nn.RNNCell(input_size, hidden_size).double()
    cell2 = from_torch(t_cell2)
    x1, hp = torch.from_numpy(xs[0]), torch.from_numpy(h0)
    with torch.no_grad():
        diff_rev = np.abs(cell2.forward(xs[0], h0) - t_cell2(x1, hp).numpy()).max()

    diffs = {
        "SimpleRNNCell vs nn.RNNCell (mọi timestep)": np.abs(hs_ours - hs_cell).max(),
        "SimpleRNNCell vs nn.RNN (mọi timestep)": np.abs(hs_ours - hs_rnn).max(),
        "h_T của mình vs h_n của nn.RNN": np.abs(hs_ours[-1] - h_last[0].numpy()).max(),
        "nn.RNNCell (2 bias) -> SimpleRNNCell (b_h = b_ih + b_hh)": diff_rev,
    }
    if verbose:
        print(f"Input nhỏ: seq_len={seq_len}, batch={batch_size}, input_size={input_size}, "
              f"hidden_size={hidden_size}")
        print(f"Số tham số: W_xh {cell.W_xh.shape} + W_hh {cell.W_hh.shape} + b_h {cell.b_h.shape} "
              f"= {cell.num_parameters()}")
        for t in range(seq_len):
            print(f"  t={t + 1}: h_t[sample 0] = {np.round(hs_ours[t, 0], 6)}")
        print("Sai khác tuyệt đối lớn nhất:")
        for name, d in diffs.items():
            print(f"  {name:<58s} {d:.2e}")
    return diffs


if __name__ == "__main__":
    diffs = compare_with_torch()
    assert all(d < 1e-10 for d in diffs.values()), diffs
    print("OK — SimpleRNNCell cho kết quả trùng với PyTorch.")
