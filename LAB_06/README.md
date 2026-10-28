# LAB 06 — Attention Mechanism (IMDb Sentiment)

**Môn:** Xử lý ngôn ngữ tự nhiên và ứng dụng — Học kỳ I 2026
**Chủ đề:** From Recurrent State to Context-Aware Representation
**Dataset:** [IMDb](https://huggingface.co/datasets/stanfordnlp/imdb), split và tokenizer giống LAB 05

## Cấu trúc thư mục

| File | Nội dung |
|---|---|
| `attention_numpy.py` | softmax ổn định số (có temperature), `scaled_dot_product_attention`, `self_attention`, `entropy` — chỉ dùng NumPy |
| `attention_model.py` | `LSTMClassifier` (LAB 05) và `AttentionClassifier` (PyTorch, để train trên IMDb) + vòng train |
| `preprocessing.py` | Copy từ LAB 05 (tokenizer, Vocabulary, padding, DataLoader) |
| `self_attention.ipynb` | Core check, Exp 1–5, Application, dữ liệu error analysis |
| `calculations.md`, `prediction.md`, `error_analysis.md`, `reflection.md` | Phần sinh viên tự viết (xem mục AI policy) |
| `results.csv` | LSTM vs Attention trên test set |
| `attention_heatmap.png`, `confusion_matrix.png`, `learning_curves.png` | Hình sinh từ notebook |
| `error_samples.csv` | 3 mẫu đúng + 3 mẫu sai kèm top token được attend |

## Cách chạy

```bash
pip install -r requirements.txt
python attention_numpy.py          # kiểm tra core NumPy (đối chiếu với phần tính tay)
jupyter notebook self_attention.ipynb
```

## Model (Exp 4)

```
ids -> Embedding(100) -> Self-Attention (Q=XW_Q, K=XW_K, V=XW_V, d=128, mask <pad>)
    -> pooling bằng learned query (context vector c = Σ αᵢhᵢ) -> Dropout 0.3 -> Linear -> sigmoid
```

Cấu hình giống LAB 05: max_length 200 (head), Adam 1e-3, batch 64, 8 epochs, clip 5.0, seed 42, chọn epoch có val loss nhỏ nhất.
**Không có positional encoding** (để dành LAB 07), nên model không biết thứ tự từ.

## Kết quả (test set 25,000 review)

| Model | Accuracy | F1 | Train time (s) | Params |
|---|---|---|---|---|
| LSTM | 0.8214 | 0.8233 | 74.3 | 2,118,089 |
| Attention | 0.8344 | 0.8296 | 21.9 | 2,055,241 |

Mỗi model chỉ chạy 1 seed (42), chênh lệch ~1.3 điểm accuracy chưa đủ để kết luận Attention tốt hơn LSTM.

## Hạn chế cần biết khi đọc kết quả

- **Pooling attention gần như uniform** (≈1/T): câu 4 token cho 0.24–0.26 mỗi token, câu 12 token ~0.08 mỗi token. Model hoạt động gần giống mean-pooling nên attention weights không phải bằng chứng rằng model "chọn" token quan trọng.
- **Application:** `"I like this movie."` và `"I do not like this movie."` đều bị dự đoán `negative` với P(pos) ≈ 0.000; câu thứ 3 (`...bad, but ... excellent`) cũng ra `negative` (P=0.107). Có thể do câu ngắn khác xa review ~200 token lúc train và do thiếu thông tin vị trí; đây mới chỉ là giả thuyết, chưa được kiểm chứng.
- **Exp 1/2** dùng đặc trưng 3 chiều `[sentiment, topic, intensity]` gán tay (không phải embedding học được).
- **Self-attention 3 token** dùng ma trận ví dụ tự đặt vì PDF không in ma trận W cụ thể.
- Heatmap (Exp 3) là attention pattern của model đã train, không phải bằng chứng về hiểu ngữ pháp.

## AI policy

Các file `calculations.md`, `prediction.md`, `error_analysis.md` (phần phân tích), `reflection.md` và mọi nhận xét "✍️" trong notebook do sinh viên tự viết.

**AI assistance statement**
- **Tool:** Claude Code (Claude Sonnet 5.5)
- **Purpose:** boilerplate code, plotting, chạy experiment, dựng template tài liệu
- **Generated:** `attention_numpy.py`, `attention_model.py`, `self_attention.ipynb` (code), `README.md`, template các file `.md`
- **Modified:** _(sinh viên điền)_
- **Verified:** `attention_numpy.py` khớp `torch.nn.functional.scaled_dot_product_attention` (sai lệch ~2e-16); mask `<pad>` nhận weight 0; _(sinh viên bổ sung phần đã tự kiểm tra)_
