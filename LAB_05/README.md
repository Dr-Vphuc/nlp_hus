# LAB 05 — Sequence Models: RNN, LSTM and GRU (IMDb Sentiment)

**Môn:** Xử lý ngôn ngữ tự nhiên và ứng dụng — Học kỳ I 2026  
**Chủ đề:** From Static Representations to Sequential Modeling  
**Bài toán:** Sentiment classification (`positive` / `negative`)  
**Dataset:** [IMDb](https://huggingface.co/datasets/stanfordnlp/imdb) — 25,000 train / 25,000 test review phim  
**Repo:** [https://github.com/Dr-Vphuc/nlp_hus](https://github.com/Dr-Vphuc/nlp_hus)

## Cấu trúc thư mục

| File | Nội dung |
|---|---|
| `README.md` | Tổng quan, cách chạy, AI assistance statement |
| `calculations.md` | Bài 1 (một bước RNN), Bài 2 (information flow), Bài 3 (word order), vanishing gradient, LSTM gates |
| `prediction.md` | Prediction 1–5 trước khi chạy experiment |
| `rnn_cell.py` | `SimpleRNNCell` tự implement bằng NumPy + so sánh với `torch.nn.RNNCell` / `torch.nn.RNN` |
| `preprocessing.py` | Tải dữ liệu, split, tokenizer (giống LAB 04), `Vocabulary`, padding / truncation, DataLoader |
| `sequence_classification.ipynb` | Experiment 1–6, evaluation, learning curves, error analysis, so sánh LAB 04, application |
| `results.csv` | Kết quả mọi model trên test set (sinh ra khi chạy notebook) |
| `learning_curves.png` | Train / validation loss theo epoch của RNN, LSTM, GRU |
| `confusion_matrix.png` | Confusion matrix của TF-IDF + LR, RNN, LSTM, GRU |
| `sequence_length.png` | max_length → F1 / validation loss / training time / GPU memory |
| `error_analysis.md` | 3 mẫu đúng + 3 mẫu sai và phân tích nguyên nhân |
| `reflection.md` | Reflection (RNN → Attention) |
| `requirements.txt` | Thư viện cần cài |

## Pipeline

```
Raw review ─► Tokenizer ─► Vocabulary (fit trên TRAIN) ─► Token IDs ─► Padding / truncation (max_length)
           ─► Embedding ─► RNN / LSTM / GRU ─► h_T ─► Dropout ─► Linear ─► sigmoid ─► Sentiment
```

- **Split:** giống hệt LAB 04 — test chuẩn của IMDb; train gốc bỏ duplicate và review trùng test,
  chia 80% train / 20% validation (stratified, `random_state=42`).
- **Tokenizer:** dùng lại của LAB 04 (lowercase, bỏ HTML, giữ contraction như `don't`).
- **Vocabulary:** top 20,000 token của train (tần suất ≥ 2) + `<pad>` (id 0) + `<unk>` (id 1).
- **Padding:** `<pad>` ở cuối; model dùng `pack_padded_sequence` nên $h_T$ là hidden state tại token thật cuối cùng
  (`<pad>` không ảnh hưởng tới prediction). **Truncation:** mặc định giữ `max_length` token đầu (`head`).
- **Cấu hình chung** (giữ cố định, chỉ đổi recurrent layer hoặc `max_length`): embedding 100, hidden 128,
  1 layer, dropout 0.3, Adam lr 1e-3, batch 64, 8 epochs, gradient clipping 5.0, seed 42.
  Đánh giá test bằng trọng số của epoch có validation loss nhỏ nhất.

| Experiment | Nội dung |
|---|---|
| Exp 1 (§11) | Dataset inspection: samples, classes, class distribution, sequence length (min/median/mean/max), histogram, duplicate, missing |
| Core (§14) | `SimpleRNNCell` + so sánh với PyTorch; minh họa word order (`dog bites man`) và vanishing/exploding gradient |
| Exp 3 (§15) | Baseline TF-IDF + LR vs Embedding → RNN → Linear; bằng chứng thứ tự từ (xáo trộn từ, cặp câu cùng bag-of-words) |
| Exp 4 (§19) | RNN vs LSTM vs GRU (`max_length = 200`): parameters, training time, metrics, learning curves |
| Exp 2 / 5 (§13, §20) | `max_length ∈ {50, 100, 200}` (+30) cho cả 3 architecture; truncation head vs tail |
| Exp 6 (§21) | Long-range dependency: từ quyết định ở đầu / giữa / cuối với filler dài dần; độ nhạy theo vị trí token |
| §22–26 | Evaluation, confusion matrix, error analysis, so sánh LAB 04, application `predict_sentiment()` |

Precision / Recall / F1 tính với positive class là `positive` (label 1).

## Cách chạy

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install torch --index-url https://download.pytorch.org/whl/cu126   # GPU NVIDIA; bỏ --index-url nếu chỉ có CPU
pip install -r requirements.txt
python rnn_cell.py              # test SimpleRNNCell với PyTorch
```

Mở `sequence_classification.ipynb`, chọn kernel từ `.venv`, chạy **Run All**. Lần chạy đầu tải IMDb (~84 MB)
từ Hugging Face. Notebook ghi `results.csv`, `learning_curves.png`, `confusion_matrix.png`, `sequence_length.png`
vào cùng thư mục.

## AI assistance statement

- **Tool:** Claude Code (Claude Opus 5.5)
- **Purpose:** viết code pipeline (tiền xử lý, vocabulary, padding/truncation, training loop, các experiment,
  evaluation), visualization, giải thích API PyTorch, viết tài liệu README và khung các file markdown.
- **Generated:** `preprocessing.py`, `rnn_cell.py`, `sequence_classification.ipynb` (code và mô tả các bước),
  `requirements.txt`, `README.md`, khung câu hỏi trong `calculations.md`, `prediction.md`, `reflection.md`;
  trong `error_analysis.md`: bảng tổng hợp và số liệu từng mẫu (P(pos), token occlusion, error rate theo độ dài),
  copy từ output notebook mục 11.
- **Modified:** _(điền phần đã tự chỉnh sửa)_
- **Verified:** chạy `python rnn_cell.py` (`SimpleRNNCell` khớp `torch.nn.RNNCell` / `torch.nn.RNN`, sai số ~1e-16);
  chạy lại toàn bộ notebook từ đầu đến cuối không lỗi; kết quả TF-IDF + LR trùng LAB 04 (cùng split, cùng test set).
  _(bổ sung: đối chiếu Bài 1 tính tay với mục 4.2, đọc và giải thích được từng bước code)_
- **Không dùng AI cho:** nội dung `calculations.md`, `prediction.md`, phần phân tích trong `error_analysis.md`,
  `reflection.md`, các ô ✍️ trong notebook (diễn giải learning curves, chọn nguyên nhân lỗi), individual learning check.
