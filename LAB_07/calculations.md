# Calculations — LAB 07
> Tính tay từng bước (không dùng lớp Transformer/attention của thư viện). Kết quả được đối chiếu bằng
> `python calculations_check.py` (NumPy thuần) và `python transformer_block.py` (shape).
> Đề không in giá trị cụ thể của `X`, `W`, `W₂`, `b₂`, nên các ma trận dưới đây là **ví dụ tự chọn**.

## Bài 6 — Một attention head (`X ∈ ℝ³ˣ⁴`, `d_k = 2`)
`X` (3 token × 4 chiều), `W_Q, W_K, W_V ∈ ℝ⁴ˣ²`:

```
X   = [[1,0,1,0],      W_Q = [[1,0],     W_K = [[1,0],     W_V = [[1,1],
       [0,1,0,1],             [0,1],            [0,1],            [0,1],
       [1,1,0,0]]             [0,0],            [1,0],            [1,0],
                              [0,0]]            [0,1]]            [0,0]]
```

- `Q = X W_Q = [[1,0],[0,1],[1,1]]` (shape 3×2) — vì `W_Q` chỉ lấy 2 cột đầu của `X`
- `K = X W_K = [[2,0],[0,2],[1,1]]` (3×2) — ví dụ hàng 1: `[1,0,1,0]·W_K = [1+1, 0+0] = [2,0]`
- `V = X W_V = [[2,1],[0,1],[1,2]]` (3×2) — ví dụ hàng 1: `[1·1+0+1·1+0, 1·1+0+0+0] = [2,1]`
- `QKᵀ` (3×3):
  ```
  [[1·2+0·0, 1·0+0·2, 1·1+0·1],   = [[2, 0, 1],
   [0·2+1·0, 0·0+1·2, 0·1+1·1],      [0, 2, 1],
   [1·2+1·0, 1·0+1·2, 1·1+1·1]]      [2, 2, 2]]
  ```
- `S = QKᵀ/√2` (√2 ≈ 1.4142):
  ```
  [[1.4142, 0.0000, 0.7071],
   [0.0000, 1.4142, 0.7071],
   [1.4142, 1.4142, 1.4142]]
  ```
- `A = softmax(S)` theo từng hàng:
  - Hàng 1: `exp = [4.1133, 1, 2.0281]`, tổng 7.1414 ⇒ `[0.5760, 0.1400, 0.2840]`
  - Hàng 2 (đối xứng hàng 1): `[0.1400, 0.5760, 0.2840]`
  - Hàng 3: cả 3 score bằng nhau ⇒ `[1/3, 1/3, 1/3]`
- `Z = A V`:
  - Hàng 1: `0.576·[2,1] + 0.14·[0,1] + 0.284·[1,2] = [1.152+0.284, 0.576+0.14+0.568] ≈ [1.4359, 1.2840]`
  - Hàng 2: `0.14·[2,1] + 0.576·[0,1] + 0.284·[1,2] = [0.28+0.284, 0.14+0.576+0.568] ≈ [0.5641, 1.2840]`
  - Hàng 3: trung bình của 3 value `= [(2+0+1)/3, (1+1+2)/3] = [1, 1.3333]`

  `Z ∈ ℝ³ˣ²` (mỗi token một vector `d_k = 2`).
- **Nhận xét:** đây chính xác là attention của LAB 06 (`softmax(QKᵀ/√d_k)V`); điểm mới duy nhất là `Q, K, V` được tạo bằng các ma trận chiếu `W_Q, W_K, W_V` từ cùng một `X` (self-attention). Token 3 (`[1,1,0,0]`) có query `[1,1]` cân bằng giữa hai key nên attention đều.

## Bài 7 — Multi-head
2 head, mỗi head `d_k = 2`: `head₁ ∈ ℝᵀˣ²`, `head₂ ∈ ℝᵀˣ²`, `Concat(head₁, head₂) ∈ ℝᵀˣ⁴`.
(Ví dụ với `head₁ = Z` ở Bài 6 và `T = 3`, concat cho ma trận 3×4: hàng 1 = `[1.4359, 1.2840, h₂₁, h₂₂]`.)

- **Vì sao cần concatenation trước khi áp dụng `W_O`?**
  - Mỗi head chỉ nhìn vào một không gian con `d_k` chiều (ví dụ 2 chiều) và chạy **độc lập**, nên đầu ra của từng head chưa hề được trộn với nhau. Concat ghép các không gian con đó lại thành một vector `d_model = h·d_k` chiều cho mỗi token (`T×4`).
  - `W_O ∈ ℝ^{d_model×d_model}` (ở đây 4×4) sau đó mới **trộn thông tin giữa các head** và chiếu về `d_model` để (i) cộng được với input `X` ở residual (cùng shape `T×d_model`), (ii) đưa vào lớp kế tiếp. Nếu không concat thì không có chỗ để các head tương tác với nhau; về mặt toán học `Concat(head₁..head_h) W_O = Σᵢ headᵢ W_O^{(i)}` (mỗi head có một khối của `W_O`), nên `W_O` quyết định mỗi head đóng góp thế nào vào biểu diễn cuối.

## Bài 10 — Positional encoding, `d = 4`, `pos = 0, 1, 2`
Với `d = 4` có `i = 0, 1`: `PE(pos,0) = sin(pos/10000⁰) = sin(pos)`, `PE(pos,1) = cos(pos)`,
`PE(pos,2) = sin(pos/10000^{2/4}) = sin(pos/100)`, `PE(pos,3) = cos(pos/100)`.

| pos | dim 0 `sin(pos)` | dim 1 `cos(pos)` | dim 2 `sin(pos/100)` | dim 3 `cos(pos/100)` |
|---|---|---|---|---|
| 0 | 0 | 1 | 0 | 1 |
| 1 | 0.8415 | 0.5403 | 0.0100 | 0.9999 |
| 2 | 0.9093 | −0.4161 | 0.0200 | 0.9998 |

- `PE(0) = [0, 1, 0, 1]`, `PE(1) ≈ [0.8415, 0.5403, 0.0100, 1.0000]`, `PE(2) ≈ [0.9093, −0.4161, 0.0200, 0.9998]`.
- **Quan sát:** ba vị trí có ba vector khác nhau (khoảng cách Euclid `‖PE₀−PE₁‖ ≈ 0.959`, `‖PE₀−PE₂‖ ≈ 1.683`). Cặp dimension đầu (tần số cao, chu kỳ 2π) thay đổi nhanh theo `pos`, cặp sau (tần số thấp, chu kỳ 200π) thay đổi rất chậm — giống các "bánh răng" nhanh/chậm trong biểu diễn nhị phân, giúp mã hóa cả vị trí gần lẫn xa. Cộng `X' = X + PE` làm cùng một từ ở hai vị trí khác nhau có đầu vào khác nhau.

## Bài 14 — FFN
`x = [1, 2]ᵀ`, `W₁ = I₂`, `b₁ = [0, −1]ᵀ`.
- `W₁x + b₁ = [1, 2] + [0, −1] = [1, 1]`
- `h = ReLU([1, 1]) = [1, 1]` (không có phần tử âm nên không bị cắt về 0)
- Đề không cho `W₂, b₂` ⇒ chọn `W₂ = [[2,0],[1,1]]`, `b₂ = [0,1]ᵀ`: `W₂h = [2·1+0·1, 1·1+1·1] = [2, 2]`, `y = [2,2] + [0,1] = [2, 3]`.
- (Với GELU thay ReLU thì `h = [GELU(1), GELU(1)] ≈ [0.841, 0.841]`, `y ≈ [1.683, 2.683]`.)
- **FFN không phải attention:** FFN chỉ nhận vector của **một** vị trí và biến đổi nó (Linear → phi tuyến → Linear), không có bất kỳ tương tác nào giữa các token. Attention trộn thông tin *giữa* các token; FFN thêm biến đổi phi tuyến *tại* từng token (cùng một `W₁, W₂` cho mọi vị trí) và thường mở rộng chiều (`d_ff = 4·d_model` trong paper) để tăng sức biểu diễn. Nếu thiếu ReLU/GELU, hai Linear liên tiếp chỉ là một Linear.

## Residual connection — `y = x + F(x)`
**Nếu chỉ dùng `F(x)` thì khác gì `x + F(x)`?**
- `x + F(x)` giữ một **đường đi trực tiếp** từ input tới output: layer chỉ cần học *phần chỉnh* `F(x) = y − x` (residual); nếu `F(x) ≈ 0` thì layer là phép đồng nhất, nên thêm layer không làm mất thông tin đầu vào.
- Khi backprop, `∂y/∂x = I + ∂F/∂x`: thành phần `I` cho gradient chảy thẳng về các lớp dưới, giảm vanishing gradient ⇒ huấn luyện mạng sâu ổn định hơn.
- Nếu chỉ dùng `F(x)` thì toàn bộ thông tin của `x` phải đi qua `F`; thông tin có thể bị bóp méo/mất và gradient phải đi qua mọi phép biến đổi.
- Trong Transformer, residual còn yêu cầu `F(x)` cùng shape với `x` (`T×d_model`) — đó là lý do `W_O` chiếu về `d_model`.

## LayerNorm (ví dụ nhỏ)
LayerNorm chuẩn hóa **theo chiều feature của từng token**: `LN(v) = γ·(v − μ)/√(σ² + ε) + β`, với `μ, σ²` tính trên các feature của chính vector `v` đó (khác BatchNorm: không phụ thuộc batch và không phụ thuộc độ dài sequence).
- Ví dụ `x = [1,2]`, `F(x) = [0.5, 0.5]` ⇒ `v = x + F(x) = [1.5, 2.5]`, `μ = 2`, `σ² = 0.25`, `LN(v) = [−0.5, 0.5]/√0.25 = [−1, 1]` (γ=1, β=0).
- Vai trò: giữ scale của activation ổn định sau mỗi phép cộng residual → gradient ổn định, training ổn định hơn.

## Shape check — `B=8, T=100, d_model=128, n_heads=4`, `d_k = 128/4 = 32`
| Tensor | Shape | Ghi chú |
|---|---|---|
| `X` | `8×100×128` | `B×T×d_model` |
| `Q = XW_Q` (sau khi tách head) | `8×4×100×32` | Linear `128→128` ⇒ `8×100×128` ⇒ view `8×100×4×32` ⇒ transpose `8×4×100×32` |
| `K` | `8×4×100×32` | tương tự |
| `V` | `8×4×100×32` | tương tự |
| Attention scores `QKᵀ/√d_k` | `8×4×100×100` | `[8,4,100,32] @ [8,4,32,100]`; mỗi head có ma trận `T×T` |
| Attention weights `A` | `8×4×100×100` | softmax theo chiều cuối (key) |
| `A V` (mỗi head) | `8×4×100×32` | |
| Concat các head | `8×100×128` | `4·32 = 128` |
| Attention output (sau `W_O`) | `8×100×128` | cùng shape với `X` để cộng residual |
| FFN (hidden, `d_ff=512`) | `8×100×512` | |
| FFN output | `8×100×128` | |
| Output của block | `8×100×128` | giữ nguyên `B×T×d_model` nên xếp chồng được nhiều block |

Đã xác nhận bằng `python transformer_block.py` (in đúng các shape trên). Số tham số 1 block (`d_model=128, h=4, d_ff=512`): MHSA `4·(128·128+128) = 66,048`; FFN `128·512+512 + 512·128+128 = 131,712`; 2 LayerNorm `2·256 = 512`; tổng `198,272`.

- **Tại sao attention matrix có kích thước `T×T`?** Mỗi trong `T` token (query) phải tính một score với *mọi* `T` token (key), kể cả chính nó ⇒ `T×T` score cho mỗi head (và mỗi sample). `A[i,j]` = token i chú ý token j bao nhiêu.
- **`d_model = 128`, 8 heads ⇒ `d_k = 128/8 = 16`.**

## Số heads và `d_k` (`d_model = 128`)
| Heads | `d_k` | Params của MHSA (`4·(128²+128)`) |
|---|---|---|
| 1 | 128 | 66,048 |
| 2 | 64 | 66,048 |
| 4 | 32 | 66,048 |
| 8 | 16 | 66,048 |

Tăng số head **không** tăng chiều biểu diễn (`d_model` vẫn 128) và không tăng số tham số: nó chia `d_model` thành nhiều không gian con nhỏ hơn (`d_k` giảm tỉ lệ nghịch). Mỗi head có không gian query/key nhỏ hơn nhưng có nhiều ma trận attention độc lập hơn (nhiều "góc nhìn").

## Chi phí `O(T²)`
Ma trận attention có `T²` phần tử: khi `T` tăng từ 128 lên 256 thì `128² = 16,384 → 256² = 65,536` (gấp **4×**, vì `(2T)² = 4T²`). Với `B=64, h=4`, float32: `T=128` ⇒ `64·4·16,384·4 B = 16 MB`, `T=256` ⇒ `64 MB` mỗi block (chỉ riêng tensor attention weights, chưa tính scores/gradient).

| T | Phần tử/head | Bộ nhớ float32/head/sample |
|---|---|---|
| 32 | 1,024 | 4 KiB |
| 64 | 4,096 | 16 KiB |
| 128 | 16,384 | 64 KiB |
| 256 | 65,536 | 256 KiB |

**Tại sao Transformer không đơn giản giải quyết mọi vấn đề bằng cách tăng sequence length?**
1. Chi phí thời gian và bộ nhớ của self-attention tăng `O(T²)` (nhân thêm số head, số layer và batch size) — tăng 2× độ dài tốn 4× bộ nhớ attention, nên nhanh chóng vượt GPU.
2. Dữ liệu thật sự không phải lúc nào cũng có thông tin hữu ích ở phần cuối: như bảng "tokens retained" ở Exp 5, thêm độ dài mang lại lợi ích giảm dần trong khi chi phí tăng bậc hai.
3. Có một độ dài tối đa cố định của positional encoding/được huấn luyện, và attention phải chia xác suất cho nhiều token hơn (softmax trên `T` phần tử) nên tín hiệu quan trọng có thể bị "loãng"; model cũng khó học hơn khi dữ liệu train ngắn hơn lúc test.
