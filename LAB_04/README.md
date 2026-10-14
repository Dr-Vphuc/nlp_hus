# LAB 04 — Text Classification (IMDb Sentiment)

**Môn:** Xử lý ngôn ngữ tự nhiên và ứng dụng — Học kỳ I 2026
**Chủ đề:** From Text Representation to Supervised NLP
**Bài toán:** Option A — Sentiment classification (`positive` / `negative`)
**Dataset:** [IMDb](https://huggingface.co/datasets/stanfordnlp/imdb) — 25,000 train / 25,000 test review phim

## Cấu trúc thư mục

| File | Nội dung |
|---|---|
| `README.md` | Tổng quan, cách chạy, AI assistance statement |
| `calculations.md` | Bài tập tính toán 1–4 (logit/sigmoid, threshold, confusion matrix, accuracy trap) |
| `prediction.md` | Prediction trước khi chạy experiment |
| `classification.ipynb` | Experiment 1–5, data leakage, evaluation, confusion matrix, error analysis, model weights |
| `preprocessing.py` | Tải dữ liệu, split, `clean_text` / `tokenize`, Average Word2Vec (`MeanEmbeddingVectorizer`) |
| `results.csv` | Kết quả mọi experiment trên test set (sinh ra khi chạy notebook) |
| `confusion_matrix.png` | Confusion matrix của 3 model A, B, C (sinh ra khi chạy notebook) |
| `error_analysis.md` | 3 mẫu đúng + 3 mẫu sai và phân tích nguyên nhân |
| `reflection.md` | Reflection |
| `requirements.txt` | Thư viện cần cài |

## Pipeline

```
Dataset ─► Train / Validation / Test split ─► fit representation trên TRAIN
        ─► transform Validation / Test ─► train classifier ─► chọn C trên Validation
        ─► đánh giá trên Test ─► confusion matrix ─► error analysis ─► inspect weights
```

- **Split:** test giữ nguyên bộ test chuẩn của IMDb. Train gốc được bỏ duplicate và bỏ các review trùng với test,
  sau đó chia 80% train / 20% validation (stratified, `random_state=42`).
- **Không leakage:** TF-IDF và Word2Vec đều chỉ `fit` trên train; validation/test chỉ `transform`.
  Notebook có mục so sánh trực tiếp pipeline đúng và pipeline sai (fit TF-IDF trên toàn bộ dataset).
- **Tokenizer dùng chung** cho TF-IDF và Word2Vec: lowercase, bỏ thẻ HTML (`<br />`), giữ contraction
  (`don't`, `isn't`) để không mất thông tin phủ định.
- **Chọn siêu tham số:** `C ∈ {0.01, 0.1, 1, 10, 100}` theo F1 trên validation.

| Experiment | Nội dung |
|---|---|
| Exp 1 | Dataset inspection: số samples, classes, class distribution, độ dài, duplicate, missing |
| Exp 2 | Baseline TF-IDF (unigram, `min_df=2`) + Logistic Regression |
| Exp 3 | A: TF-IDF + LR · B: Average Word2Vec (skip-gram, 100 chiều, train trên train split) + LR · C: TF-IDF + Linear SVM |
| Exp 4 | Word order: `"this movie is good"` vs `"this movie is not good"` và các cặp câu tương tự |
| Exp 5 | TF-IDF unigram vs unigram + bigram |
| Thêm | Kiểm chứng Prediction 3–5: vocabulary size, bỏ stopwords, accuracy vs F1 trên tập mất cân bằng |

Precision / Recall / F1 tính với positive class là `positive` (label 1).

## Cách chạy

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Mở `classification.ipynb`, chọn kernel từ `.venv`, chạy **Run All**. Lần chạy đầu tải IMDb (~84 MB) từ Hugging Face.
Notebook ghi `results.csv` và `confusion_matrix.png` vào cùng thư mục.

## AI assistance statement

- **Tool:** Claude Code (Claude Opus 5.5)
- **Purpose:** viết code pipeline (tiền xử lý, các experiment, evaluation), visualization, gợi ý cách dùng
  scikit-learn / gensim, viết tài liệu README.
- **Generated:** `preprocessing.py`, `classification.ipynb` (code và mô tả các bước), `requirements.txt`, `README.md`.
- **Modified:** _(điền phần đã tự chỉnh sửa)_
- **Verified:** _(điền cách đã kiểm tra, ví dụ: chạy lại toàn bộ notebook, đọc và giải thích được từng bước code)_
- **Không dùng AI cho:** `calculations.md`, `prediction.md`, `error_analysis.md`, `reflection.md`,
  các ô ✍️ trong notebook (diễn giải metrics, chọn nguyên nhân lỗi), individual learning check.
