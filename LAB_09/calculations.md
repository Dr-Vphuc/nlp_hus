# Calculations — LAB 09

> Tính tay trước, sau đó đối chiếu với `tokenizer_alignment.ipynb` (BIO bằng `words_to_bio`, shape bằng DistilBERT thật,
> loss/IGNORE bằng `CrossEntropyLoss`) và `ner_utils.entity_metrics` (đã đối chiếu với `seqeval`).

## Bài 1 — BIO

Câu: `Elon Musk founded SpaceX in California.` với `Elon Musk = PER`, `SpaceX = ORG`, `California = LOC`.

Tokens (tách cả dấu câu):

| # | Token | BIO |
|---|---|---|
| 1 | Elon | B-PER |
| 2 | Musk | I-PER |
| 3 | founded | O |
| 4 | SpaceX | B-ORG |
| 5 | in | O |
| 6 | California | B-LOC |
| 7 | . | O |

Entity `Elon Musk` gồm 2 token: token đầu là `B-PER`, token sau là `I-PER`. `SpaceX` và `California` mỗi entity chỉ có 1 token nên chỉ có `B-`. Dấu chấm và các từ không thuộc entity là `O`. Notebook cho kết quả giống hệt.

## Bài 2 — Entity boundary

`University of California Berkeley` (toàn bộ cụm là một ORG):

| Token | BIO |
|---|---|
| University | B-ORG |
| of | I-ORG |
| California | I-ORG |
| Berkeley | I-ORG |

**Vì sao không gán tất cả token là `ORG` nếu muốn biểu diễn ranh giới entity?**
Nhãn chỉ có loại (`ORG ORG ORG ORG`) thì không cho biết entity bắt đầu và kết thúc ở đâu. Ví dụ `Google Microsoft` (hai tổ chức đứng sát nhau) và `University of California` (một tổ chức) đều thành `ORG ORG...`, không phân biệt được "một entity dài" với "nhiều entity liền kề". Nếu dự đoán sai chỉ một token thì cũng không xác định được span. `B-` đánh dấu *điểm bắt đầu* entity mới, `I-` đánh dấu *tiếp tục* entity đang mở, `O` đánh dấu *đóng* entity; nhờ vậy từ chuỗi nhãn khôi phục duy nhất danh sách span `(type, start, end)`. Notebook kiểm tra: `['B-ORG','B-ORG']` → 2 entity, còn `['I-ORG','I-ORG']` → 1 entity. NER vì thế không chỉ nhận diện *loại* mà còn phải nhận diện *ranh giới*.

## Bài 3 — Shape

`B = 8`, `T = 128`, `d = 768`, `C = 9`.

- hidden states: `H ∈ ℝ^{B×T×d}` = **8 × 128 × 768** (mỗi token một vector `d` chiều).
- classification head: `Z = H W + b` với `W ∈ ℝ^{d×C}` (768 × 9), `b ∈ ℝ^{C}`; số tham số head = `768·9 + 9 = 6,921`.
- logits: `Z ∈ ℝ^{B×T×C}` = **8 × 128 × 9** (mỗi token một vector 9 logit; tổng 8·128·9 = 9,216 logit).
- labels: **8 × 128** (mỗi token một id nhãn `0..8`, hoặc `-100` cho IGNORE).
- Loss: `reshape` logits thành `(B·T, C) = (1024, 9)` và labels thành `(1024,)` rồi `CrossEntropyLoss(ignore_index=-100)`.

Notebook (`tokenizer_alignment.ipynb`, mục 4) chạy DistilBERT thật cho đúng shape này: `(8,128,768)`, `(8,128,9)`, head `Linear(768→9)` với 6,921 tham số.

## Label alignment — ví dụ tính tay

`Nikolaus van der Pas visited Reykjavik .` với `B-PER I-PER I-PER I-PER O B-LOC O`. Tokenizer (WordPiece cased) tách:
`[CLS] Nikola ##us van der Pa ##s visited Rey ##k ##ja ##vik . [SEP]` (đã kiểm tra bằng notebook).
Số word = 7, số token (không tính đặc biệt) = 12 ⇒ split ratio = 12/7 ≈ 1.71, 3 word bị tách (Nikolaus, Pas, Reykjavik).

| Token | word_id | Strategy A (đã chọn) | Strategy B |
|---|---|---|---|
| [CLS] | None | -100 | -100 |
| Nikola | 0 | B-PER | B-PER |
| ##us | 0 | -100 | I-PER |
| van | 1 | I-PER | I-PER |
| der | 2 | I-PER | I-PER |
| Pa | 3 | I-PER | I-PER |
| ##s | 3 | -100 | I-PER |
| visited | 4 | O | O |
| Rey | 5 | B-LOC | B-LOC |
| ##k, ##ja, ##vik | 5 | -100, -100, -100 | I-LOC, I-LOC, I-LOC |
| . | 6 | O | O |
| [SEP] | None | -100 | -100 |

**Chiến lược chọn: Strategy A** (nhãn chỉ ở subword đầu; các subword còn lại và token đặc biệt = `-100`).
1. *Giải thích:* NER annotate theo word, nên mỗi word chỉ cần một dự đoán. Lúc suy luận lấy dự đoán tại subword đầu của từng word, nhất quán với lúc huấn luyện.
2. *Áp dụng nhất quán:* `align_labels_first_subtoken` dùng cho cả train/validation/test; dự đoán mức word (`predict`) lấy argmax tại subword đầu; mọi metric tính mức word (đúng với số word gốc của CoNLL).
3. *Ảnh hưởng:* (i) Với Strategy B, mảnh sau của `B-LOC` phải nhận `I-LOC`, tức model bị dạy rằng "`##k` là phần tiếp theo của entity" cho cả những word không phải entity mới — dễ đưa nhãn `I-` vào những mảnh không có ý nghĩa ngôn ngữ riêng; Strategy A tránh được điều này. (ii) Subword bị IGNORE vẫn là input và vẫn tham gia attention, nên thông tin của chúng *vẫn đến được* subword đầu; chỉ là không có tín hiệu loss trực tiếp tại đó. (iii) Cái giá của A là word dài có nhiều subword chỉ nhận một tín hiệu nhãn (ít gradient hơn B), và nếu vì truncation mà subword đầu bị cắt thì word đó không có dự đoán (code gán `O`; chỉ 2/3,453 câu test vượt 128 token). Lab không chạy so sánh A và B (không bắt buộc), nên kết luận về ảnh hưởng ở (iii) là suy luận, chưa kiểm chứng bằng thực nghiệm.
4. Trên test set, 17,026/63,461 subword (≈ 26.8%) có nhãn IGNORE (trừ [CLS]/[SEP]).

## IGNORE index trong loss

`L = -Σ_{t: yₜ ≠ -100} log p(yₜ | xₜ)` (chia cho số token hợp lệ). Ví dụ 4 token, token 2 có nhãn `-100`:
loss từng token = `[2.5892, (bỏ), 2.0086, 4.1514]` ⇒ mean = `(2.5892 + 2.0086 + 4.1514)/3 = 2.9164`, trùng với `CrossEntropyLoss(ignore_index=-100)`; gradient tại logits của token IGNORE bằng 0 (notebook, mục 3).
IGNORE **không** có nghĩa là token không tồn tại: token vẫn đi qua Transformer, vẫn tham gia self-attention của các token khác; chỉ là *nhãn* của nó không được dùng trực tiếp trong loss.

## Entity-level vs token-level — ví dụ tính tay

Gold: `Barack Obama visited New York` = `B-PER I-PER O B-LOC I-LOC`; Pred: `B-PER I-PER O B-LOC O`.
- Token accuracy = 4/5 = **0.80** (chỉ `York` sai).
- Entity-level: gold có 2 entity `{PER(0,2), LOC(3,5)}`, dự đoán có 2 entity `{PER(0,2), LOC(3,4)}`.
  - `PER(0,2)` trùng hoàn toàn ⇒ TP = 1.
  - `LOC(3,4)` ≠ `LOC(3,5)` (khác ranh giới) ⇒ vừa là FP (entity dự đoán sai) vừa làm gold `LOC(3,5)` là FN.
  - TP = 1, FP = 1, FN = 1 ⇒ P = 1/2, R = 1/2, F1 = **0.50**.
- Cùng một dự đoán nhưng token accuracy 0.80, entity F1 0.50: token accuracy bị thổi phồng bởi nhãn `O` (≈ 83% token CoNLL là `O`) và bởi việc đúng từng phần của entity.

Ví dụ thứ hai: Gold `{New York (LOC), Barack Obama (PER), Google (ORG)}`, Pred `{New (LOC), Barack Obama (PER), Paris (LOC)}` ⇒ TP = 1 (`Barack Obama`), FP = 2 (`New`, `Paris`), FN = 2 (`New York`, `Google`) ⇒ P = 1/3, R = 1/3, F1 = 1/3.

Công thức: `P = TP/(TP+FP)`, `R = TP/(TP+FN)`, `F1 = 2PR/(P+R)` với TP/FP/FN đếm theo **entity** (exact match cả type lẫn span).
