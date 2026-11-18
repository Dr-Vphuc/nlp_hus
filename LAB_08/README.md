# LAB 08 — Pretrained Transformers and BERT Fine-Tuning (IMDb Sentiment)

**Môn:** Xử lý ngôn ngữ tự nhiên và ứng dụng — Học kỳ I 2026
**Chủ đề:** From Training a Transformer to Using a Pretrained Language Model
**Dataset:** [IMDb](https://huggingface.co/datasets/stanfordnlp/imdb) (split train/val như LAB 05/06, xem "Thiết lập"); domain shift: [Amazon polarity](https://huggingface.co/datasets/fancyzhx/amazon_polarity), [Rotten Tomatoes](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes)
**Model pretrained:** `distilbert-base-uncased` (BERT-style, 6 layer, 66.4M tham số) — **không phải BERT-base**, xem mục "Điều chỉnh so với đề".

## Cấu trúc thư mục

| File | Nội dung |
|---|---|
| `tokenization.ipynb` | Calculation 2 (tokenize `"The cats are playing."`), `tokens.keys()`, special tokens, attention mask (có kiểm chứng ảnh hưởng của mask), kiểm chứng MLM loss bằng BERT thật, contextual embedding, phân bố độ dài token IMDb |
| `pretrained_classification.py` | Toàn bộ code nặng: data split, `CLSClassifier` ([CLS]→Dropout→Linear), Transformer nhỏ from scratch, vòng train (AMP, gradient accumulation, chọn epoch theo val loss), registry các run, benchmark theo độ dài, gộp kết quả |
| `fine_tuning.ipynb` | Reproducibility, pipeline + head, Exp 1–5, Application (metrics + confusion matrix), confidence analysis, dữ liệu error analysis. Đọc kết quả từ `outputs/` (không train lại trong notebook) |
| `calculations.md`, `prediction.md`, `error_analysis.md`, `reflection.md`, `learning_check.md` | Tính tay MLM loss + tokenization, prediction trước experiment, phân tích lỗi + 6 câu hỏi bắt buộc, reflection (~480 từ), trả lời 12 câu Individual Learning Check |
| `results.csv` | Mỗi run một dòng (acc/precision/recall/F1, thời gian, tham số, bộ nhớ, siêu tham số, hardware) |
| `domain_shift.csv` | Metrics của từng model trên IMDb / Amazon / Rotten Tomatoes |
| `data_efficiency.png`, `confusion_matrix.png`, `confidence_analysis.png` | Hình bắt buộc theo đề |
| `domain_shift.png`, `sequence_length_cost.png`, `token_length_distribution.png` | Hình bổ sung (Exp 5, Exp 4, độ dài token) |
| `error_samples.csv` | 8 mẫu lỗi (3 FP, 3 FN, 2 lỗi rất tự tin) với đầy đủ text, xác suất, độ dài token |
| `outputs/` | `<run>.json` (metrics, lịch sử train, demo), `<run>_probs.npz` (xác suất dự đoán), `seqlen_benchmark.csv`, `logs/` (log chạy, gồm các lần crash/OOM) |
| `requirements.txt` | Phiên bản thư viện đã dùng |

## Cách chạy

```bash
pip install -r requirements.txt
# toàn bộ thí nghiệm (cần GPU; đã chạy xong, kết quả nằm ở outputs/ - run đã có sẽ được bỏ qua)
python pretrained_classification.py --list
python pretrained_classification.py --runs all
python pretrained_classification.py --collect        # gộp outputs/*.json -> results.csv, domain_shift.csv
jupyter notebook tokenization.ipynb                  # tokenization + MLM
jupyter notebook fine_tuning.ipynb                   # đọc outputs/, vẽ hình, confidence, error analysis
```

Lần đầu chạy cần mạng để tải `distilbert-base-uncased`, IMDb, Amazon polarity (chỉ split test), Rotten Tomatoes (test).

## Thiết lập (Reproducibility)

| Mục | Giá trị |
|---|---|
| Model / Tokenizer | `distilbert-base-uncased` (66,364,418 tham số, WordPiece vocab 30,522 — trùng vocab `bert-base-uncased`); `tokenization.ipynb` dùng `bert-base-uncased` (cùng vocab) |
| Head | `last_hidden_state[:,0]` ([CLS]) → Dropout(0.1) → Linear(768→2) → softmax |
| Dataset | IMDb; split gốc như LAB 05/06 (train 20,000 / val 5,000 stratified seed 42, test 25,000); **sau đó lấy subset cố định**: train pool **10,000**, val **2,000**, test **5,000** (stratified, seed 42) |
| Domain shift | Amazon polarity test (2,000 mẫu cân bằng, `title. content`), Rotten Tomatoes test (1,066) |
| Max sequence length | 128 (mặc định), truncation giữ phần đầu; Exp 4: 64 / 128 / 256 |
| Batch size | 32 hiệu dụng (micro-batch 4–16 × gradient accumulation vì GPU chung); Transformer nhỏ: 64 |
| Learning rate | Fine-tune 2e-5 (warmup 10% + linear decay); frozen head 1e-3; random-init DistilBERT 1e-4; Transformer nhỏ 5e-4 |
| Epochs | 2 (DistilBERT: fine-tune/frozen/random-init); 6 (Transformer nhỏ); control 10% data: 10; chọn epoch có val loss nhỏ nhất |
| Optimizer | AdamW (weight decay 0.01), grad-clip 1.0, fp16 autocast |
| Random seed | 42 (một seed duy nhất cho mọi run) |
| Hardware | NVIDIA GeForce RTX 3050 6GB Laptop GPU, torch 2.14.1+cu126, transformers 5.20.2. **GPU bị dùng chung với các tiến trình khác trong lúc chạy** |
| Training time | Fine-tune T=128, 10k review, 2 epoch: ~1,118 s nhưng **không đáng tin để so sánh** (xem Hạn chế) |

## Điều chỉnh so với đề (cần đọc)

1. **DistilBERT thay cho BERT-base** và **subset nhỏ hơn** (train 10,000 / val 2,000 / test 5,000) vì GPU 6 GB bị các tiến trình khác dùng chung (lần chạy đầu bằng BERT-base bị OOM rồi chậm ~3–50 mẫu/giây). Do đó accuracy tuyệt đối (~87%) thấp hơn BERT-base trên toàn bộ dữ liệu (~92–93%); **các con số này không so sánh trực tiếp được với LAB 05/06** (khác tập test).
2. **Exp 4** dùng 50% training pool (5,000 review) thay vì 100% để tiết kiệm GPU; T=128 chính là run `ft_len128_50`.
3. Gradient accumulation (micro-batch nhỏ) để tránh OOM; kết quả toán học tương đương batch 32 (khác nhau chút ít về số học fp16), nhưng **peak memory trong `results.csv` phụ thuộc micro-batch của từng run** nên không so sánh được giữa các run; bộ nhớ theo T lấy từ benchmark riêng (`outputs/seqlen_benchmark.csv`).
4. "From scratch" gồm 2 cấu hình: (a) Transformer nhỏ 4 layer, d=256 (~11.1M) giống tinh thần LAB 07; (b) kiến trúc DistilBERT khởi tạo ngẫu nhiên (control cùng kiến trúc với pretrained). Hyper-parameter của chúng chỉ thử một giá trị (không tune).

## Kết quả chính (IMDb test subset 5,000, 1 seed)

### Exp 1 — Pretrained vs From Scratch

| Model | Accuracy | F1 | Train time (s)* | # Params |
|---|---|---|---|---|
| Transformer from scratch (4L, d=256) | 0.7484 | 0.7328 | 387 (6 epoch) | 11,104,770 |
| DistilBERT random init (control) | 0.8000 | 0.7976 | 338 | 66,364,418 |
| **Pretrained DistilBERT + fine-tuning** | **0.8704** | **0.8701** | 1,118 | 66,364,418 |

\* Thời gian chịu ảnh hưởng nặng của việc GPU dùng chung (run random-init nhiều khả năng chạy lúc ít tranh chấp hơn): chỉ để tham khảo, **không dùng để kết luận chi phí**.

### Exp 2 — Frozen vs Fine-tuning

| Mode | Encoder trainable? | Trainable params | Accuracy | F1 | Peak GPU mem (MB) |
|---|---|---|---|---|---|
| Frozen | No | 1,538 | 0.7918 | 0.7889 | 440 |
| Fine-tuning | Yes | 66,364,418 | 0.8704 | 0.8701 | 1,354 |

### Exp 3 — Data efficiency (xem `data_efficiency.png`)

| Training data | Fine-tune Acc / F1 | Frozen Acc / F1 | Small scratch Acc / F1 |
|---|---|---|---|
| 10% (1,000) | 0.7968 / 0.7979 | 0.7332 / 0.7385 | 0.5948 / 0.5728 |
| 25% (2,500) | 0.8448 / 0.8416 | 0.7658 / 0.7571 | 0.6742 / 0.7086 |
| 50% (5,000) | 0.8584 / 0.8602 | 0.7778 / 0.7674 | 0.7186 / 0.7315 |
| 100% (10,000) | 0.8704 / 0.8701 | 0.7918 / 0.7889 | 0.7484 / 0.7328 |

Mức giảm 100% → 10%: fine-tune −7.4, frozen −5.9, scratch **−15.4** điểm accuracy. Control: 10% data với 10 epoch (320 bước thay vì 64) cho 0.8180 / 0.8277, tức một phần "tụt" ở 10% là do ít bước cập nhật.

### Exp 4 — Max sequence length (50% training pool; xem `sequence_length_cost.png`)

| Max length | Accuracy | F1 | Time/step, benchmark (s)† | Peak mem benchmark (MB)† | Attention T² share of FLOPs |
|---|---|---|---|---|---|
| 64 | 0.7932 | 0.7951 | 0.152 | 1,291 | 1.4% |
| 128 | 0.8584 | 0.8602 | 0.165 | 1,293 | 2.7% |
| 256 | 0.8970 | 0.8982 | 0.212 | 1,349 | 5.3% |
| 512 (chỉ benchmark) | — | — | 0.306 | 1,560 | 10.0% |

† Benchmark có kiểm soát: forward+backward DistilBERT, micro-batch 4, token giả. Chi phí đo thực tế tăng chậm hơn tuyến tính (T×8 → thời gian ×2.0) và rất xa `O(T²)` (×64) vì ở T ≤ 512 phần attention bậc hai chiếm ≤10% FLOPs; số đo ở T nhỏ còn bị overhead và tranh chấp GPU chi phối. Thời gian train từng run (`results.csv`) không dùng để suy luận (micro-batch khác nhau, GPU chung).

### Exp 5 — Domain shift (train IMDb → test; xem `domain_shift.png`)

| Model | IMDb (in-domain) | Rotten Tomatoes | Amazon |
|---|---|---|---|
| DistilBERT fine-tuned | 0.8704 | 0.7908 (−8.0) | 0.9135 (+4.3) |
| DistilBERT frozen | 0.7918 | 0.7411 (−5.1) | 0.8240 (+3.2) |
| Small scratch | 0.7484 | 0.6098 (−13.9) | 0.7520 (+0.4) |
| DistilBERT random init | 0.8000 | 0.7008 (−9.9) | 0.7730 (−2.7) |

Amazon **không** giảm (thậm chí cao hơn) vì review ngắn hơn (median 94 so với 220 token) và ít bị cắt, nên không so sánh được độ khó giữa các tập. Trên Rotten Tomatoes model fine-tuned lệch positive: dự đoán positive 58.7% (nhãn 50%), precision 0.748, recall 0.878, ECE 0.019 → 0.042.

### Application — Sentiment classification (fine-tuned, 5,000 review)

Accuracy 0.8704, Precision 0.8719, Recall 0.8684, F1 0.8701; TN/FP/FN/TP = 2181 / 319 / 329 / 2171 (`confusion_matrix.png`).

### Confidence analysis (`confidence_analysis.png`)

| Mức | n | Accuracy |
|---|---|---|
| Low (0.5–0.7) | 803 | 0.620 |
| Medium (0.7–0.9) | 1,366 | 0.809 |
| High (0.9–1.0) | 2,831 | 0.971 |

ECE = 0.019. Confidence **có** liên hệ với đúng/sai nhưng không đảm bảo: bin High vẫn có 82 lỗi, 23 lỗi có confidence ≥ 0.95 (hai lỗi tự tin nhất có vẻ là nhãn nhiễu/mơ hồ, xem `error_analysis.md`).

## Prediction vs kết quả

| Prediction (viết trước khi chạy) | Kết quả |
|---|---|
| P1: pretrained > scratch, kỳ vọng BERT ≈ 90–93%, scratch ≈ 82–87% | **Chiều đúng**; độ lớn sai: 87.0% vs 74.8% (random-init cùng kiến trúc 80.0%) — thấp hơn dự đoán vì DistilBERT, 10k mẫu, cắt 128 token |
| P2: frozen **kém hơn** fine-tuning (C), kỳ vọng 80–87% | **Đúng (C)**; 79.2% — sát đáy khoảng dự đoán, kém fine-tuning 7.9 điểm |
| P3: scratch giảm mạnh hơn khi còn 10% dữ liệu | **Đúng**: −15.4 so với −7.4 (pretrained) và −5.9 (frozen); số tuyệt đối ở 10% (pretrained 79.7%, scratch 59.5%) thấp hơn dự đoán 86–89% / 65–75% |
| Exp 4: lợi ích giảm dần, chi phí < ×4 mỗi lần gấp đôi T | Một phần đúng: chi phí tăng ít (đúng); lợi ích vẫn lớn từ 128→256 (+3.9) |
| Exp 5: Amazon giảm nhiều hơn RT; scratch giảm nhiều hơn pretrained | **Sai** với Amazon (không giảm), đúng với RT (−8.0 vs scratch −13.9) |

## Hạn chế cần biết khi đọc kết quả

- **Một seed duy nhất** cho mọi run; test 5,000 mẫu cho sai số chuẩn ≈ 0.5 điểm (95% CI ≈ ±1) nên chênh lệch vài điểm (ví dụ 50% vs 100% dữ liệu) chưa đủ chắc chắn.
- **Thời gian train không so sánh được** (GPU dùng chung, throughput dao động 3–60 mẫu/giây trong lúc chạy; micro-batch khác nhau giữa các run). Chi phí tính toán được suy từ số tham số, FLOPs lý thuyết và benchmark có kiểm soát, không từ cột `train_time_s`.
- Một số run có crash/OOM giữa chừng và được chạy lại từ đầu (log trong `outputs/logs/`); kết quả cuối chỉ lấy từ các run hoàn tất.
- Exp 3 ở 10% dữ liệu bị lẫn giữa "ít dữ liệu" và "ít bước cập nhật" (đã có control 10 epoch cho fine-tune nhưng chưa có cho scratch/frozen).
- Exp 5 thiếu baseline in-domain (train trên chính Amazon/RT) nên không tách được shift khỏi độ khó/độ dài của từng tập; mọi nguyên nhân giải thích cho suy giảm trên RT là giả thuyết.
- Hyper-parameter của hai cấu hình from-scratch không được tune (lr một giá trị), nên khoảng cách với pretrained có thể bị ước lớn.
- Model đã train không được lưu (nặng); notebook dùng xác suất đã lưu trong `outputs/*.npz`. Do đó không chạy lại được thí nghiệm ablation trên từng mẫu lỗi.
- Phần ✍️ trong notebook và các file `.md` là phân tích dựa trên đúng các số ở trên; các giải thích cơ chế (ví dụ vì sao frozen kém hơn) là **giả thuyết**, chưa kiểm chứng bằng ablation.

## AI policy

Đề không cho phép giao cho AI làm: tính MLM, prediction trước experiment, giải thích pretraining/fine-tuning, phân tích kết quả, error analysis, reflection. **Trong lần làm này tất cả các file trên (kể cả `calculations.md`, `prediction.md`, `error_analysis.md`, `reflection.md`, `learning_check.md` và các nhận xét ✍️) do Claude (agent) soạn theo yêu cầu "làm trọn vẹn lab"**. Sinh viên cần tự đọc, hiểu, sửa lại bằng lời của mình (đặc biệt `prediction.md` — tuy được viết trước khi có kết quả nhưng không phải là dự đoán của sinh viên — và `reflection.md`) trước khi nộp.

**AI assistance statement**
- **Tool:** Claude Code (Claude Sonnet 5.5)
- **Purpose:** viết code, chạy thí nghiệm, vẽ hình, soạn tài liệu/phân tích
- **Generated:** `pretrained_classification.py`, `tokenization.ipynb`, `fine_tuning.ipynb`, toàn bộ file `.md` ở trên, `README.md`
- **Modified:** _(sinh viên điền)_
- **Verified:** `calculations.md` được đối chiếu bằng code (`-ln 0.70 = 0.3567`, `-ln 0.95 = 0.0513`, `-ln 0.01 = 4.6052`; tokenization khớp output notebook); pipeline train được smoke-test trước khi chạy đầy đủ; số liệu trong README/ghi chú được lấy từ `results.csv`, `domain_shift.csv` và output notebook; _(sinh viên bổ sung phần đã tự kiểm tra)_
