# Calculations — LAB 08

> Tính tay bằng công thức `L = -Σ_{i∈M} log P(y_i | x)` (logarit tự nhiên, đơn vị nats). Đối chiếu bằng code ở `tokenization.ipynb` (mục 3).
> Phần Calculation 2 điền sau khi chạy tokenizer (đề không yêu cầu đoán trước).

## Calculation 1 — MLM loss

Câu `The cat [MASK] fish.` Chỉ có 1 vị trí bị mask nên `M = {t}` và `L = -log P(y_t | x)`.
Model dự đoán: `cat 0.10`, `dog 0.05`, `eats 0.70`, `fish 0.15` (tổng = 1.00, đúng là một phân phối xác suất). Token đúng: `eats`.

**Phép tính chính**

- `L = -log(0.70) = 0.3567` (`≈ 0.357`).
- Ý nghĩa: `exp(-0.3567) = 0.70`, nghĩa là loss nhỏ tương ứng với việc model gán xác suất khá cao cho đáp án đúng. Loss không phụ thuộc vào xác suất của `cat`, `dog`, `fish` một cách trực tiếp — chỉ có `P(eats)` xuất hiện trong công thức (các xác suất còn lại ảnh hưởng gián tiếp vì tổng phải bằng 1: tăng `P(eats)` nghĩa là giảm phần còn lại).

**Câu 1 — Nếu `P(eats) = 0.95`**

- `L = -log(0.95) = 0.0513`.
- So với 0.3567: loss giảm `0.3567 − 0.0513 = 0.3054`, tức giảm khoảng **86%** (gấp ~7 lần nhỏ hơn). Model tự tin hơn và đúng nên loss giảm mạnh, tiệm cận 0 khi `P → 1`.

**Câu 2 — Nếu `P(eats) = 0.01`**

- `L = -log(0.01) = 4.6052`.
- So với 0.3567: loss tăng `4.6052 − 0.3567 = 4.2485`, tức **gấp ~12.9 lần**. Model gần như "loại" đáp án đúng nên bị phạt rất nặng; loss tiến tới `+∞` khi `P → 0`.

**Bảng tổng hợp**

| `P(eats)` | `L = -ln P` | Nhận xét |
|---|---|---|
| 0.95 | 0.0513 | model đúng và tự tin → loss rất nhỏ |
| 0.70 | 0.3567 | đề bài |
| 0.25 (đều 4 từ) | 1.3863 (`= ln 4`) | model đoán mù trong 4 ứng viên |
| 0.01 | 4.6052 | model gần như loại bỏ đáp án đúng |
| `1/30522` (đoán mù trên toàn vocab BERT) | 10.326 (`= ln 30522`) | mức loss ban đầu của model chưa học |

Nhận xét: hàm `-log` **không đối xứng** — chênh lệch từ 0.70 lên 0.95 chỉ giảm 0.31, nhưng từ 0.70 xuống 0.01 tăng 4.25. Model bị phạt nặng hơn nhiều khi *tự tin sai* so với phần thưởng khi *tự tin đúng*.

**Câu 3 — Vì sao cross-entropy phù hợp với MLM?**

1. MLM thực chất là bài toán **phân loại nhiều lớp**: tại vị trí bị mask, chọn 1 trong `|V|` token (≈30,522 với BERT). Softmax + cross-entropy là cặp chuẩn cho phân loại một nhãn đúng trong nhiều lớp.
2. `-log P(y_i | x)` chính là **negative log-likelihood** của token thật, nên cực tiểu hóa nó là maximum likelihood estimation: ép model gán xác suất cao nhất có thể cho token đúng — đúng với mục tiêu `P(w_t | w_{≠t})`.
3. Phạt rất mạnh khi model tự tin sai (xem `L = 4.61` ở trên), và gradient theo logit có dạng đơn giản `softmax(z) − onehot(y)` — không bị triệt tiêu khi dự đoán sai (khác với MSE trên xác suất, gradient của MSE+softmax nhỏ khi bão hòa).
4. Cộng được theo nhiều vị trí mask (`Σ_{i∈M}`) và theo batch; mỗi câu chỉ cần tính loss ở các vị trí bị mask, không cần nhãn con người → học được từ văn bản thô (self-supervised).
5. Là *proper scoring rule*: xác suất được model học ra có xu hướng hiệu chỉnh (calibrated) về phân phối thật, thay vì chỉ học thứ hạng.

## Calculation 2 — Tokenization `"The cats are playing."`

Tokenizer: `bert-base-uncased` (WordPiece, vocab 30,522, lowercase). Kết quả lấy từ `tokenization.ipynb` mục 1.

1. **Token sequence:** `['[CLS]', 'the', 'cats', 'are', 'playing', '.', '[SEP]']`
2. **Token IDs:** `[101, 1996, 8870, 2024, 2652, 1012, 102]`
3. **Special tokens:** `[CLS]` (id 101, đầu chuỗi) và `[SEP]` (id 102, cuối chuỗi). Không có `[PAD]` vì chỉ có 1 câu; chữ `The` được hạ thành `the` (uncased).
4. **Attention mask:** `[1, 1, 1, 1, 1, 1, 1]` (toàn token hợp lệ). Nếu ghép batch với câu dài hơn (ví dụ đệm tới 10 token) thì 3 vị trí cuối là `[PAD]` với mask `0`: `[1,1,1,1,1,1,1,0,0,0]`.
5. **So sánh số lượng:**

| Đại lượng | Số lượng |
|---|---|
| words (tách theo khoảng trắng: `The`, `cats`, `are`, `playing.`) | 4 |
| words + dấu câu (pre-tokenizer của BERT tách `.` ra) | 5 |
| tokens kể cả special | 7 |
| tokens nội dung (không tính `[CLS]`, `[SEP]`) | 5 |
| subword pieces (token bắt đầu bằng `##`) | **0** |

Nhận xét: số token (7) lớn hơn số word (4) do (i) dấu chấm tách riêng, (ii) 2 special token. Trong câu này **không có subword nào** — `cats` và `playing` đều là token đầy đủ trong vocab. Subword xuất hiện ở từ hiếm/dài, ví dụ (từ notebook mục 1) `tokenization → token ##ization`, `unhappiness → un ##ha ##pp ##iness`; còn `unbelievable` vẫn là 1 token trong vocab này (khác với ví dụ minh họa `un + believe + able` trong đề — đề cũng ghi tokenizer cụ thể có thể cho subword khác). Điều này cho thấy "1 từ = 1 token" không đúng nói chung, nên độ dài chuỗi tính bằng token phải đo bằng tokenizer, không đếm từ.

## Bổ sung — chi phí attention theo độ dài (liên quan Exp 4)

Ma trận attention có `T × T` phần tử mỗi head mỗi layer.

| `T` | `T²` | so với `T = 64` |
|---|---|---|
| 64 | 4,096 | 1× |
| 128 | 16,384 | 4× |
| 256 | 65,536 | 16× |

Tăng `T` gấp đôi ⇒ phần attention tăng 4× (bậc 2), còn phần FFN/embedding chỉ tăng 2× (tuyến tính). Với BERT-base và `T ≤ 256`, FFN vẫn chiếm phần lớn FLOPs nên thời gian thực tế tăng ít hơn 4× — sẽ đối chiếu với số đo ở Exp 4.
