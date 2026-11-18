# LAB 09 — Named Entity Recognition và Information Extraction (CoNLL-2003)

**Môn:** Xử lý ngôn ngữ tự nhiên và ứng dụng — Học kỳ I 2026
**Chủ đề:** From Sequence Classification to Token Classification
**Dataset:** CoNLL-2003 NER (bản parquet [`lhoestq/conll2003`](https://huggingface.co/datasets/lhoestq/conll2003) vì `conll2003` gốc dùng loading script, `datasets` mới không hỗ trợ). Nhãn BIO (IOB2): PER / ORG / LOC / MISC. 14,041 train / 3,250 validation / 3,453 test câu; test có 5,648 entity.

## Cấu trúc thư mục

| File | Nội dung |
|---|---|
| `ner_utils.py` | Load CoNLL, `words_to_bio`, `spans_from_tags` (BIO → span), **label alignment** (Strategy A: first-subtoken, Strategy B: propagate), metric entity-level tự cài (tổng / theo type / theo độ dài), phân loại lỗi |
| `baseline_ner.py` | Baseline: đặc trưng token (từ, hình dạng chữ, tiền/hậu tố, ±2 từ) → `DictVectorizer` → Logistic Regression |
| `ner_model.py`, `train_ner.py` | Token classification với DistilBERT-cased: tokenize + align, vòng huấn luyện tự viết (`ignore_index=-100`), suy luận mức word, cơ chế resume |
| `tokenizer_alignment.ipynb` | BIO (Calculation 1, 2), subword + `word_ids`, Strategy A/B, IGNORE index, shape (Calculation 3), **Exp 1** Tokenization Analysis |
| `ner_finetuning.ipynb` | Pipeline từng bước (tokens, input_ids, attention_mask, word_ids, labels, logits, predictions), đường cong huấn luyện, **Exp 2–6**, confusion matrix, context analysis |
| `information_extraction.py` | Mini IE (Option A — News): raw text → NER → JSON + đánh giá entity-level trên 30 câu tự gán gold |
| `calculations.md`, `prediction.md`, `error_analysis.md`, `reflection.md` | Phần tính tay / dự đoán / phân tích lỗi / reflection (xem mục AI policy) |
| `results.csv` | Baseline vs Transformer (P/R/F1 entity-level, token accuracy, số tham số, thời gian train) |
| `entity_metrics.csv` | P/R/F1 theo PER / ORG / LOC / MISC cho cả hai model |
| `entity_length_analysis.png` (+ `.csv`) | F1 theo độ dài entity (1 / 2 / 3 / 4+ token) |
| `confusion_matrix.png` | Confusion matrix nhãn BIO 9 lớp mức word (baseline và Transformer) |
| `error_samples.csv` | 24 mẫu lỗi (6 FP, 6 FN, 6 boundary, 6 wrong type) |
| `tokenization_stats.csv`, `tokenization_analysis.png` | Số liệu Exp 1 (30 câu) và thống kê split rate trên toàn test set |
| `seen_unseen_analysis.csv`, `training_curves.png`, `training_log.json`, `train_log.txt`, `train_runs.txt` | Phân tích entity hiếm/chưa thấy, đường cong train, log huấn luyện |
| `ie_results.csv`, `ie_examples.json`, `ie_errors.csv` | Kết quả mini IE (P/R/F1, JSON có cấu trúc cho 30 câu, lỗi) |
| `baseline_predictions.json`, `baseline_metrics.json`, `transformer_predictions.json` | Dự đoán trên test (dùng chung cho notebook) |
| `ner_checkpoint/` | Checkpoint DistilBERT đã fine-tune (~250 MB, đã đưa vào `.gitignore`) |

## Cách chạy

```bash
pip install -r requirements.txt
python baseline_ner.py                 # ~1 phút, tạo baseline_predictions.json / baseline_metrics.json
python train_ner.py 2                  # fine-tune DistilBERT 2 epoch (CPU, ~55 phút); tạo ner_checkpoint/, training_log.json, transformer_predictions.json
python information_extraction.py       # mini IE + đánh giá 30 câu
jupyter nbconvert --to notebook --execute --inplace tokenizer_alignment.ipynb
jupyter nbconvert --to notebook --execute --inplace ner_finetuning.ipynb   # cần chạy sau train_ner.py
```

(Thứ tự: `baseline_ner.py` và `train_ner.py` phải chạy trước `ner_finetuning.ipynb`.)

## Thiết lập

- **Label alignment: Strategy A.** `tokenizer(words, is_split_into_words=True)` → `word_ids()`; subword đầu của mỗi word nhận nhãn của word, các subword sau và `[CLS]/[SEP]` nhận `-100` (IGNORE, không vào loss). Khi suy luận lấy dự đoán tại subword đầu. Lý do/ảnh hưởng: `calculations.md`.
- **Model:** `distilbert-base-cased` (65.2M tham số) + `Linear(768→9)`; AdamW lr 5e-5, weight decay 0.01, batch 16, **2 epoch**, warmup 10% + linear decay, clip 1.0, seed 42, max_length 128 (2/3,453 câu test dài hơn). Chọn epoch có validation entity-F1 cao nhất (epoch 2: 0.9351; epoch 1: 0.9157). Chạy trên **CPU**.
- **Metric:** entity-level exact match (đúng cả type lẫn span), tự cài trong `ner_utils.entity_metrics` và **đối chiếu khớp với `seqeval`** (assert trong notebook). Cả hai model đánh giá trên **cùng test set**.

## Kết quả (test set, 3,453 câu / 5,648 entity)

| Model | Precision | Recall | F1 | Token accuracy | Params | Train time |
|---|---|---|---|---|---|---|
| Baseline (token features + LogReg) | 0.7405 | 0.7916 | 0.7652 | 0.9605 | ~1.0M | 31 s |
| DistilBERT-cased (fine-tuned, 2 epoch) | 0.8826 | 0.9026 | **0.8925** | 0.9788 | 65.2M | ~54 phút (CPU) |

Token accuracy luôn cao hơn nhiều so với entity F1 (0.9605 vs 0.7652; luôn đoán `O` đã có accuracy 0.825) nên không dùng làm metric chính.

**Theo loại entity (F1, P/R trong `entity_metrics.csv`)**

| Entity | Baseline | Transformer | #gold |
|---|---|---|---|
| PER | 0.800 | 0.949 | 1,617 |
| ORG | 0.666 | 0.861 | 1,661 |
| LOC | 0.837 | 0.914 | 1,668 |
| MISC | 0.753 | 0.790 | 702 |

**Theo độ dài entity (F1)**

| Độ dài | #gold | Baseline | Transformer |
|---|---|---|---|
| 1 token | 3,574 | 0.745 | 0.888 |
| 2 token | 1,776 | 0.832 | 0.910 |
| 3 token | 236 | 0.655 | 0.848 |
| 4+ token | 62 | 0.495 | 0.812 |

**Theo mức độ xuất hiện trong train (F1)**: Transformer — frequent (≥5 lần) 0.945, rare (1–4) 0.899, unseen 0.846; Baseline — 0.922 / 0.789 / 0.623.

**Tokenization (Exp 1, toàn test set)**: 46,435 word → 63,461 subword (split ratio 1.367); 17.8% word bị tách. Entity bị tách nhiều hơn non-entity (PER 59.9%, ORG 45.2%, LOC/MISC 31.2% vs `O` 12.0%); từ viết HOA 85.9%, từ viết thường 6.6%.

**Information Extraction (30 câu tin tức tự viết + gold tự gán, 81 entity)**: Precision 0.975, Recall 0.963, **F1 0.969** (PERSON 0.971, ORGANIZATION 0.957, LOCATION 0.982). Ví dụ: `Barack Obama visited Paris.` → `{"PERSON": ["Barack Obama"], "ORGANIZATION": [], "LOCATION": ["Paris"]}`.

Phân tích lỗi chi tiết, context analysis (cùng surface form, khác type), đối chiếu prediction: `error_analysis.md`.

## Hạn chế / lưu ý khi đọc kết quả

- **Mỗi model chỉ chạy 1 seed (42)**, không có khoảng tin cậy; chênh lệch Baseline–Transformer (+12.7 F1) đủ lớn để kết luận, nhưng các chênh lệch nhỏ (ví dụ MISC 0.753 vs 0.790, nhóm 4+ token chỉ có 62 entity) thì chưa chắc chắn.
- **Dùng DistilBERT và chỉ 2 epoch** (thay vì BERT-base/nhiều epoch) vì chạy trên CPU dùng chung với các tác vụ khác; F1 ~0.89 thấp hơn mức ~0.91–0.92 thường báo cáo cho BERT-base trên CoNLL-2003. Validation F1 vẫn tăng giữa epoch 1 → 2 (0.916 → 0.935), nên model chưa hội tụ hoàn toàn.
- **Quá trình huấn luyện bị gián đoạn**: lần chạy đầu bị kill (hết bộ nhớ hệ thống / tiến trình bị dừng) nên đã bỏ và chạy lại từ đầu với cơ chế resume (lưu trạng thái mỗi 100 step). Lần chạy cuối bị `DefaultCPUAllocator: not enough memory` ở khoảng step 500+ của epoch 1 và được tự động resume từ checkpoint step 500 (xem `train_runs.txt`, `train_log.txt`). Vì vậy kết quả có thể không tái lập chính xác từng bit khi chạy lại; `train_time_s` chỉ tính thời gian chạy thực (không tính thời gian chờ giữa hai lần chạy).
- **Baseline không dùng POS/chunk tag** của CoNLL và dự đoán từng token độc lập (không CRF), nên có 462 chuyển tiếp BIO không hợp lệ (Transformer: 72); đây là baseline đơn giản đúng yêu cầu đề, không phải baseline mạnh nhất có thể.
- **Strategy B (propagate) chỉ được cài đặt và minh hoạ**, không được huấn luyện để so sánh với Strategy A (đề không yêu cầu). Nhận định về ảnh hưởng alignment trong `calculations.md` là suy luận, trừ số liệu: entity chứa word bị tách có recall 0.900 so với 0.906 khi không bị tách (Transformer).
- **Mini IE chỉ dựa trên 30 câu tự viết**, gồm nhiều tên nổi tiếng/dễ nên kết quả F1 0.969 **lạc quan** hơn so với văn bản thật; gold do tự gán (theo quy ước CoNLL: CLB/đội = ORG, quốc gia = LOC) và không được dùng để chỉnh model. MISC được bỏ khỏi đầu ra IE (Option A chỉ lấy PERSON/ORGANIZATION/LOCATION). Lỗi còn lại: `Cupertino` bị đoán ORG, `Elon Musk` bị đoán ORG, `Twitter` bị bỏ sót.
- **Context analysis** dựa nhiều vào bảng thể thao của CoNLL (định dạng dòng đã gợi type), nên chưa phải phép thử thuần ngữ nghĩa; các câu tự viết (`Washington` / `Jordan` / `Leeds`) chỉ là minh hoạ định tính.
- CoNLL-2003 có **nhiễu annotation** (ví dụ `Mantua`, `van Boxmeer`, `AFC`) nên một phần "lỗi" không phải lỗi thật của model.

## AI policy

Đề yêu cầu sinh viên tự làm BIO labeling exercise, label alignment reasoning, prediction, diễn giải metric, error analysis và reflection. Các file `calculations.md`, `prediction.md`, `error_analysis.md`, `reflection.md` và các nhận xét trong notebook trong thư mục này là **bản nháp do AI soạn** từ kết quả thực nghiệm (`prediction.md` được viết trước khi chạy bất kỳ experiment nào của lab); sinh viên cần đọc lại, kiểm tra và viết lại bằng hiểu biết của mình trước khi nộp.

**AI assistance statement**
- **Tool:** Claude Code (Claude Sonnet 5.5)
- **Purpose:** boilerplate code, chạy experiment, vẽ hình, dựng nháp tài liệu
- **Generated:** `ner_utils.py`, `baseline_ner.py`, `ner_model.py`, `train_ner.py`, `information_extraction.py` (kể cả 30 câu + gold), hai notebook, `README.md`, nháp các file `.md`
- **Modified:** _(sinh viên điền)_
- **Verified:** `entity_metrics` khớp `seqeval` (precision/recall/F1 trên dự đoán thật và trên dự đoán nhiễu giả lập); cơ chế `-100` cho loss/gradient = 0 tại token IGNORE và khớp tính tay; shape `(8,128,768)`, `(8,128,9)` khớp DistilBERT thật; `spans_from_tags` / `words_to_bio` khớp phần tính tay; không có chuyển tiếp BIO không hợp lệ trong nhãn gold train; _(sinh viên bổ sung phần đã tự kiểm tra)_
