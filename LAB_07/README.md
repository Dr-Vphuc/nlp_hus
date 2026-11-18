# LAB 07 — Transformer Architecture (IMDb Sentiment)

**Môn:** Xử lý ngôn ngữ tự nhiên và ứng dụng — Học kỳ I 2026
**Chủ đề:** From Attention to Transformer
**Dataset:** [IMDb](https://huggingface.co/datasets/stanfordnlp/imdb), split và tokenizer giống LAB 05/06 (train 19,823 / val 4,956 / test 25,000, vocab 20,002, `max_length = 200` trừ Exp 5)

## Cấu trúc thư mục

| File | Nội dung |
|---|---|
| `transformer_block.py` | **Core:** `MultiHeadSelfAttention`, `FeedForward`, `TransformerBlock` tự ghép từ Linear/softmax/LayerNorm/Dropout (không dùng `nn.TransformerEncoderLayer`); in bảng shape; đối chiếu với `nn.TransformerEncoderLayer` |
| `positional_encoding.py` | PE sinusoidal, permutation experiment (A: không PE, B: có PE), vẽ `positional_encoding.png` |
| `transformer_model.py` | `TransformerClassifier` (mean / CLS pooling, bật/tắt PE), `LSTMClassifier`, vòng train |
| `preprocessing.py` | Copy từ LAB 05/06 (tokenizer, Vocabulary, padding, DataLoader) |
| `calculations_check.py` | NumPy thuần, đối chiếu các bài tính tay |
| `transformer_classification.ipynb` | Core check, Exp 1–6, Application, Error analysis, tổng hợp (đã chạy, có output) |
| `calculations.md`, `prediction.md`, `error_analysis.md`, `reflection.md` | Tính tay, prediction (+ kiểm chứng), error analysis, reflection |
| `results.csv` | Mọi run chính (LSTM, Transformer, 1/2/8 head, 4 độ dài, CLS pooling, bỏ PE); `results_seeds.csv`: 3 seed; `permutation_results.csv`; `benchmark_forward_backward.csv` |
| `positional_encoding.png`, `attention_heads.png`, `learning_curves.png`, `confusion_matrix.png`, `sequence_length.png` | Hình |
| `error_samples.csv` | 3 mẫu đúng + 3 mẫu sai kèm top token được attend |

## Cách chạy

```bash
pip install -r requirements.txt     # cần thêm nbformat/nbclient nếu muốn chạy notebook từ dòng lệnh
python transformer_block.py         # bảng shape B=8,T=100,d=128,h=4 + đối chiếu nn.TransformerEncoderLayer (sai lệch ~3.6e-07)
python positional_encoding.py       # PE d=4 + permutation experiment + positional_encoding.png
python calculations_check.py        # đối chiếu phần tính tay
jupyter notebook transformer_classification.ipynb   # toàn bộ experiment; run train được cache trong .cache/
```
Notebook train 15 model (≈ 1–1.5 giờ trên RTX 3050 dùng chung với các tiến trình khác). Thư mục `.cache/` đã bị xóa sau khi chạy; chạy lại sẽ train lại từ đầu.

## Model (Exp 4)

```
ids -> Embedding(128) -> + sinusoidal PE -> TransformerBlock (4 heads, d_k=32, d_ff=256, post-LN, dropout 0.1)
    -> mean pooling (bỏ <pad>) -> Dropout 0.3 -> Linear -> sigmoid
```
Cấu hình train giống LAB 05/06: Adam 1e-3, batch 64, 8 epochs, clip 5.0, seed 42, chọn epoch có val loss nhỏ nhất. Embedding dim = 128 (= `d_model`) cho **cả** LSTM và Transformer (khác 100 ở LAB 05/06 nên con số LSTM không trùng với LAB 06).

## Kết quả (test set 25,000 review, 1 seed = 42 trừ khi ghi khác)

**Exp 4 — Transformer vs LSTM**

| Model | Params | Accuracy | Precision | Recall | F1 | best epoch |
|---|---|---|---|---|---|---|
| LSTM | 2,692,481 | 0.8090 | 0.7844 | 0.8524 | 0.8170 | 6 |
| Transformer | 2,692,865 | 0.8394 | 0.8289 | 0.8554 | 0.8419 | 3 |

3 seed (42, 43, 44): LSTM accuracy 0.8129 ± 0.0034, F1 0.8134 ± 0.0053; Transformer accuracy 0.8392 ± 0.0050, F1 0.8388 ± 0.0101. Transformer overfit rất sớm (val loss tốt nhất ở epoch 3: 0.338, epoch 8: 0.733; train loss 0.047).

**Exp 2 — Số heads (`d_model = 128`)**

| Heads | d_k | Params | Accuracy | F1 |
|---|---|---|---|---|
| 1 | 128 | 2,692,865 | 0.8433 | 0.8436 |
| 2 | 64 | 2,692,865 | 0.8360 | 0.8421 |
| 4 | 32 | 2,692,865 | 0.8394 | 0.8419 |
| 8 | 16 | 2,692,865 | 0.8398 | 0.8406 |

Chênh lệch ≤ 0.007, bằng cỡ dao động giữa seed ⇒ không có bằng chứng nhiều head hơn thì tốt hơn.

**Exp 3 — Head diversity (`attention_heads.png`):** sau train, cosine trung bình giữa các cặp head = 0.14 (khởi tạo: 0.82); các head khác nhau nhưng có chồng lấn (head 1 và 3 cùng nhìn "terrible" ở câu ví dụ). Không gán nhãn ngôn ngữ cho head.

**Exp 5 — Sequence length (Transformer 4 head)**

| max_length | Tokens retained | Attention matrix | Attention memory (B=64, h=4, MB) | Accuracy | F1 |
|---|---|---|---|---|---|
| 32 | 13.7% | 1,024 | 1 | 0.7038 | 0.6968 |
| 64 | 27.1% | 4,096 | 4 | 0.7616 | 0.7641 |
| 128 | 51.2% | 16,384 | 16 | 0.8122 | 0.8106 |
| 256 | 76.7% | 65,536 | 64 | 0.8508 | 0.8537 |

Peak GPU memory đo thực (B=16, forward+backward, `benchmark_forward_backward.csv`): Transformer 3.6 MB (T=32) → 4,175 MB (T=2048), gần `∝ T²`; LSTM 12 → 184 MB.

**Exp 6 — Pooling:** Mean 0.8394 / F1 0.8419; CLS-style 0.8364 / 0.8380 (chênh nhỏ hơn nhiễu).
**Ablation PE trên IMDb:** có PE 0.8394 / 0.8419, không PE 0.8406 / 0.8402.

**Permutation experiment:** "dog bites man" vs "man bites dog" — không PE: cosine(pooled) = 1.000000, sai lệch ≈ 1e-8 (giống hệt); có PE: các biểu diễn token khác nhau (0.03–1.3). Trên IMDb, model không PE cho cùng P(pos) = 0.0988 với câu gốc và câu xáo trộn; model có PE đổi từ 0.0334 sang 0.0345 (gần như không dùng thứ tự).

**Application (P(pos) của Transformer):** "I really enjoyed this movie." 0.997; "This was a terrible film." 0.001; câu dài tích cực 0.947; "I did not like this movie at all." 0.007; "This movie is not bad." 0.001 (**sai**, nghĩa là tích cực); "bad, but ... excellent" 0.630; "great, but boring" 0.087; câu mơ hồ 0.308. LSTM cho kết quả tương tự về nhãn.

## Hạn chế / điều cần biết khi đọc kết quả

- **Thời gian huấn luyện và micro-benchmark không đáng tin:** GPU/CPU được dùng chung với các tiến trình khác trong lúc chạy (thời gian mỗi epoch của cùng một cấu hình dao động từ 17 đến 62 s). Vì vậy tôi không kết luận gì chắc chắn về tốc độ Transformer vs LSTM; các cột "Training time" trong `results.csv` chỉ mang tính tham khảo. Số đo **bộ nhớ** là xác định và đáng tin hơn.
- Mỗi cấu hình (heads, độ dài, pooling, ablation PE) chỉ 1 seed; chỉ Transformer vs LSTM có 3 seed. Chênh lệch nhỏ hơn ~0.005 accuracy không nên được diễn giải.
- Model chỉ có 1 block, train từ đầu, overfit sau ~3 epoch; kết luận không khái quát cho Transformer sâu hoặc pretrained.
- PE không cải thiện accuracy trên IMDb (xem reflection); permutation experiment là bằng chứng cho vai trò *khả năng* biết thứ tự, không phải cho accuracy.
- Ma trận ví dụ trong `calculations.md` (X, W, W₂, b₂) tự chọn vì PDF không in giá trị cụ thể.
- Attention visualization không phải giải thích; `error_analysis.md` đã tách quan sát khỏi giả thuyết, các giả thuyết chưa được kiểm chứng bằng ablation.
- Application: "This movie is not bad." bị dự đoán negative (P=0.001) cả ở Transformer lẫn bản không PE, chưa xác định nguyên nhân (có thể do câu quá ngắn so với review ~200 token lúc train hoặc model chủ yếu dựa vào từ "bad").

## AI policy

**AI assistance statement**
- **Tool:** Claude Code (Claude Sonnet 5.5)
- **Purpose:** viết code, chạy experiment, vẽ hình, soạn nội dung `calculations.md`, `prediction.md`, `error_analysis.md`, `reflection.md` và README
- **Generated:** toàn bộ code (`transformer_block.py`, `positional_encoding.py`, `transformer_model.py`, `calculations_check.py`, notebook) và các file `.md` — lưu ý đề bài quy định calculation, prediction, shape calculation, PE calculation, giải thích multi-head, error analysis và reflection phải do sinh viên tự làm; các file này **cần sinh viên đọc, tự kiểm tra và viết lại bằng lời của mình** trước khi nộp
- **Modified:** _(sinh viên điền)_
- **Verified:** `TransformerBlock` khớp `nn.TransformerEncoderLayer` (sai lệch 3.6e-07, có padding mask); mỗi hàng attention cộng bằng 1 và `<pad>` nhận weight 0; PE `d=4` khớp phần tính tay; các số trong `calculations.md` khớp `calculations_check.py`; prediction được viết trước khi chạy experiment và không sửa; _(sinh viên bổ sung phần đã tự kiểm tra)_
