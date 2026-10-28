# Calculations — LAB 06
> ✍️ Tính tay (không dùng thư viện attention). Sau khi tính xong mới đối chiếu với output của
> `python attention_numpy.py` / mục 1 của notebook.

## Bài 1 — Attention scores
`q = [1, 2]ᵀ`, `K = [[1,0],[0,1],[1,1]]`, `sᵢ = qᵀkᵢ`

- Tính:
  - `s₁ = [1,2]·[1,0] = 1·1 + 2·0 = 1`
  - `s₂ = [1,2]·[0,1] = 1·0 + 2·1 = 2`
  - `s₃ = [1,2]·[1,1] = 1·1 + 2·1 = 3`
- Score vector `s = [1, 2, 3]`

## Bài 2 — Softmax
`s = [1, 2, 3]`

- `exp(sᵢ)`: `[e¹, e², e³] ≈ [2.7183, 7.3891, 20.0855]`, tổng ≈ 30.1929
- `α = [2.7183, 7.3891, 20.0855] / 30.1929 ≈ [0.0900, 0.2447, 0.6652]`
- Kiểm tra `Σαᵢ = 1`: `0.0900 + 0.2447 + 0.6652 = 0.9999 ≈ 1` (sai lệch do làm tròn; code cho đúng 1.0)
- **Vì sao phần tử có score lớn nhất nhận weight lớn nhất?**
  `exp` là hàm đơn điệu tăng nên score lớn hơn ⇒ `exp(s)` lớn hơn; mẫu số là hằng số chung cho mọi phần tử, nên thứ tự của `α` giữ nguyên thứ tự của `s`. Hơn nữa `exp` còn *khuếch đại* chênh lệch: score chỉ hơn 1 đơn vị đã được hơn ~2.7 lần weight, nên phần tử lớn nhất chiếm phần lớn xác suất (0.6652).

## Bài 3 — Context vector
`α = [0.1, 0.2, 0.7]`, `v₁ = [1,0]`, `v₂ = [0,1]`, `v₃ = [2,2]`

- `c = 0.1v₁ + 0.2v₂ + 0.7v₃ = [0.1, 0] + [0, 0.2] + [1.4, 1.4] = [1.5, 1.6]`
- **Value nào đóng góp nhiều nhất? Vì sao?**
  `v₃` đóng góp nhiều nhất: weight lớn nhất (0.7) và bản thân `v₃ = [2,2]` cũng có độ lớn lớn nhất. Nó đóng góp `[1.4, 1.4]` trong tổng `[1.5, 1.6]` (~93% chiều 1, ~88% chiều 2). Context là trung bình có trọng số nên bị kéo về phía value có weight cao.

## Bài 4 — Toàn bộ attention computation
`q = [1,0]`, keys `k₁=[1,0], k₂=[0,1], k₃=[1,1]`, values `v₁=[1,0], v₂=[0,2], v₃=[2,1]`

- scores: `s = [q·k₁, q·k₂, q·k₃] = [1, 0, 1]`
- softmax / attention weights: `exp = [2.7183, 1, 2.7183]`, tổng = 6.4366 ⇒ `α ≈ [0.4223, 0.1554, 0.4223]`
- weighted sum / context vector:
  `c = 0.4223·[1,0] + 0.1554·[0,2] + 0.4223·[2,1] = [0.4223 + 0.8446, 0.3108 + 0.4223] ≈ [1.267, 0.733]`
- (Tham khảo, nếu chia `√d_k = √2`: scores `[0.7071, 0, 0.7071]` ⇒ `α ≈ [0.4011, 0.1978, 0.4011]`, `c ≈ [1.2033, 0.7967]`.)

## Scaling
`d_k = 4`, `QKᵀ = [4, 2, 0]`

- scaled scores `QKᵀ/√4 = [2, 1, 0]`
- softmax có scaling: `exp = [7.3891, 2.7183, 1]`, tổng = 11.1074 ⇒ `[0.6652, 0.2447, 0.0900]`
- softmax không scaling: `exp = [54.598, 7.389, 1]`, tổng = 62.987 ⇒ `[0.8668, 0.1173, 0.0159]`
- **Scaling làm thay đổi độ tập trung của softmax như thế nào? Vì sao cần chia √d_k?**
  Chia `√d_k` thu nhỏ các score nên softmax bớt "nhọn": weight lớn nhất giảm từ 0.8668 xuống 0.6652, phân phối đều hơn. Lý do cần chia: nếu các thành phần của q, k độc lập, kỳ vọng 0, phương sai 1 thì `q·k` có phương sai `d_k` (độ lệch chuẩn `√d_k`). `d_k` lớn ⇒ score rất lớn ⇒ softmax bão hòa (gần one-hot), gradient qua softmax gần 0 và việc học bị chậm/kém ổn định. Chia `√d_k` đưa phương sai của score về ~1, giữ softmax ở vùng có gradient tốt.

## Temperature (`s = [4,2,0]`)
Công thức: `softmax(s/T)`.

| Temperature | Max weight | Entropy | Nhận xét |
|---|---|---|---|
| 0.5 | 0.9817 (`[0.9817, 0.0180, 0.0003]`) | 0.0930 | Rất nhọn, gần one-hot; tập trung gần như hoàn toàn vào một token |
| 1.0 | 0.8668 (`[0.8668, 0.1173, 0.0159]`) | 0.4411 | Mức mặc định, vẫn tập trung vào token lớn nhất |
| 2.0 | 0.6652 (`[0.6652, 0.2447, 0.0900]`) | 0.8324 | Phẳng hơn, phân phối đều hơn; entropy tăng (tối đa `ln 3 ≈ 1.0986`) |

Entropy tính theo nats. T nhỏ ⇒ max weight tăng, entropy giảm; T lớn thì ngược lại. (Lưu ý: T = 2 với `s` chính là softmax có scaling ở trên vì `√d_k = 2`.)

## Self-attention `["I","like","cats"]`
Đề không in ma trận cụ thể nên dùng ma trận ví dụ như trong notebook (mục 3), `d_k = 2`.

- `X`, `W_Q`, `W_K`, `W_V` (ma trận được cung cấp):
  - `X = [[1,0],[0,1],[1,1]]` (hàng: I, like, cats)
  - `W_Q = [[1,0],[0,1]]`, `W_K = [[1,1],[0,1]]`, `W_V = [[1,0],[1,1]]`
- `Q = XW_Q`, `K = XW_K`, `V = XW_V`:
  - `Q = [[1,0],[0,1],[1,1]]`
  - `K = [[1,1],[0,1],[1,2]]`
  - `V = [[1,0],[1,1],[2,1]]`
- `QKᵀ`:
  ```
  [[1, 0, 1],
   [1, 1, 2],
   [2, 1, 3]]
  ```
- `softmax(QKᵀ/√d_k)` (chia √2 ≈ 1.4142, softmax theo từng hàng):
  ```
  scores/√2 = [[0.7071, 0.0000, 0.7071],
               [0.7071, 0.7071, 1.4142],
               [1.4142, 0.7071, 2.1213]]
  A         = [[0.4011, 0.1978, 0.4011],
               [0.2483, 0.2483, 0.5035],
               [0.2840, 0.1400, 0.5760]]
  ```
  Ví dụ hàng 1: `exp = [2.028, 1, 2.028]`, tổng 5.056 ⇒ `[0.4011, 0.1978, 0.4011]`.
- `softmax(...)V`:
  ```
  C = [[1.4011, 0.5989],
       [1.5035, 0.7517],
       [1.5760, 0.7160]]
  ```
  Ví dụ hàng 1: `0.4011·[1,0] + 0.1978·[1,1] + 0.4011·[2,1] = [1.4011, 0.5989]`.
- **Một token tương tác với token khác trong cùng sequence thể hiện ở đâu?**
  Ở ma trận `QKᵀ` / ma trận attention `A` (T×T): phần tử `A[i,j]` là mức token i "chú ý" token j trong cùng câu. Ví dụ "cats" (hàng 3) dành 0.576 weight cho chính nó, 0.284 cho "I", 0.140 cho "like". Sau đó `C = AV` trộn thông tin value của các token khác vào biểu diễn mới của token i, nên mỗi hàng của `C` mang ngữ cảnh của cả câu.

## Q / K / V
- **Query:** vector biểu diễn "token đang hỏi / cần tìm thông tin gì"; dùng để so khớp với các key.
- **Key:** vector biểu diễn "token này chứa/đại diện cho thông tin gì" để được so khớp; `qᵀk` cho biết mức liên quan.
- **Value:** nội dung thực sự được lấy ra và cộng có trọng số (theo attention weights) để tạo context vector.
- **Self-attention khác attention thông thường ở đâu?**
  Ở attention thông thường (ví dụ seq2seq), query đến từ một chuỗi (decoder) còn key/value đến từ chuỗi khác (encoder). Ở self-attention, Q, K, V đều được chiếu từ cùng một chuỗi `X`, nên mỗi token tương tác trực tiếp với mọi token khác trong chính chuỗi đó (kể cả chính nó), không phụ thuộc khoảng cách và có thể tính song song.

## Chi phí O(T²)
Ma trận attention `QKᵀ` có kích thước T×T nên có T² phần tử.

| T | Attention matrix | Số phần tử |
|---|---|---|
| 10 | 10×10 | 100 |
| 100 | 100×100 | 10,000 |
| 500 | 500×500 | 250,000 |
| 1000 | 1000×1000 | 1,000,000 |

- **Vì sao self-attention có vấn đề computational cost khi sequence rất dài? Nếu T gấp đôi thì sao?**
  Mỗi token phải tính score với mọi token khác nên thời gian và bộ nhớ cho ma trận attention tăng theo O(T²) (nhân với số head và số layer). Khi T gấp đôi, số phần tử tăng gấp 4 lần `(2T)² = 4T²`; từ T=10 lên 1000 (tăng 100×) thì chi phí tăng 10,000×. Do đó sequence dài nhanh chóng tốn quá nhiều bộ nhớ/thời gian, nên cần các biến thể như sparse/local attention, linear attention, FlashAttention...
