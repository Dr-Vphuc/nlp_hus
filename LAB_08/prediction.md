# Prediction trước experiment — LAB 08

> Viết **trước khi** có bất kỳ kết quả nào của `pretrained_classification.py` (file này được viết khi các run đầu tiên chưa hoàn thành; chưa nhìn accuracy nào).
> Mỗi mục theo mẫu của đề: **Prediction / Reason / Expected evidence** (+ Confidence).
> Các con số kỳ vọng dựa trên kiến thức chung về BERT/IMDb, không dựa trên kết quả của lab này.

## Prediction 1 — Pretrained Transformer fine-tuned có tốt hơn Transformer nhỏ train from scratch trên cùng dữ liệu không?

- **Prediction:** Có. BERT-base fine-tuned trên IMDb sẽ tốt hơn rõ rệt Transformer nhỏ khởi tạo ngẫu nhiên.
- **Reason:** Transformer from scratch phải học cả *biểu diễn ngôn ngữ* (nghĩa của từ, cú pháp, phủ định, ngữ cảnh) lẫn *bài toán sentiment* chỉ từ 20,000 review có nhãn. BERT đã học biểu diễn đó trong pretraining (MLM + NSP) trên khoảng 3.3 tỷ từ (BooksCorpus + Wikipedia), nên fine-tuning chỉ cần điều chỉnh để phân biệt positive/negative: `θ_pretrained → θ_task` thay vì bắt đầu từ trọng số ngẫu nhiên.
- **Expected evidence:** test accuracy của BERT fine-tuned khoảng 90–93%, Transformer nhỏ from scratch khoảng 82–87% (chênh vài điểm đến ~10 điểm). Nếu cho cả BERT-base random init (cùng kiến trúc) thì nó phải kém BERT pretrained, khẳng định lợi ích đến từ *trọng số pretrained* chứ không phải kiến trúc. Đổi lại BERT đắt hơn: số tham số ~110M so với ~10M của Transformer nhỏ, và thời gian train mỗi epoch lớn hơn.
- **Confidence:** 90% về chiều của kết luận (pretrained > scratch); 60% về độ lớn của khoảng cách.

## Prediction 2 — Freeze toàn bộ pretrained encoder, chỉ train classification head

- **Prediction:** **C — kém hơn** fine-tuning toàn bộ (nhưng không kém quá nhiều và vẫn tốt hơn mức đoán ngẫu nhiên rất nhiều).
- **Reason:** Encoder BERT được pretrain bằng MLM/NSP, **không** phải để phân biệt sentiment, nên vector `[CLS]` chưa được tối ưu cho bài toán này (`[CLS]` ở last layer chưa được huấn luyện trực tiếp cho classification — ở BERT gốc nó chỉ qua NSP). Head tuyến tính chỉ cắt một siêu phẳng trong không gian 768 chiều cố định; còn fine-tuning điều chỉnh được cả 110M tham số nên có thể "xoay" không gian biểu diễn về hướng có ích cho sentiment (ví dụ tách các câu có phủ định). Mặt khác sentiment là thông tin tương đối "dễ" được mã hóa trong biểu diễn ngữ cảnh, nên frozen vẫn dùng được.
- **Expected evidence:** frozen khoảng 80–87% accuracy, fine-tuning 90–93% (kém hơn khoảng 5–10 điểm). Training time của frozen nhỏ hơn (không có backward qua encoder; chỉ 768×2+2 = 1,538 tham số trainable so với ~109M).
- **Confidence:** 80% (C), 10% (B), 5% (A), 5% (không đoán được).

## Prediction 3 — Chỉ có 10% training data: pretrained hay from-scratch giảm performance mạnh hơn?

- **Prediction:** Model **train from scratch** sẽ giảm mạnh hơn khi đi từ 100% xuống 10% dữ liệu; pretrained model giảm ít.
- **Reason (dựa trên vai trò của pretraining):** Pretraining đã "trả trước" phần lớn chi phí học ngôn ngữ bằng dữ liệu không nhãn khổng lồ. Fine-tuning chỉ cần dữ liệu có nhãn để học *ánh xạ nhỏ* từ biểu diễn có sẵn sang nhãn, nên ít phụ thuộc vào kích thước dữ liệu downstream. Model from scratch phải học cả embedding của hàng nghìn từ từ chính tập huấn luyện; với 2,000 review (10% của 20k) nhiều từ chỉ xuất hiện vài lần nên embedding không học được và model dễ overfit. Do đó đường cong data-efficiency của from scratch dốc hơn.
- **Expected evidence:**
  - Pretrained: 100% ≈ 91–92%, 50% ≈ 90–91%, 25% ≈ 89–90%, 10% ≈ 86–89% (giảm khoảng 3–5 điểm).
  - From scratch: 100% ≈ 84–86%, 10% ≈ 65–75% (giảm khoảng 10–20 điểm).
  - Phải dựa vào toàn bộ đường cong 10/25/50/100% chứ không chỉ một điểm.
- **Confidence:** 85%.

## Prediction bổ sung (không bắt buộc — để đối chiếu với Exp 4, 5)

### Exp 4 — Max sequence length `T ∈ {64, 128, 256}`
- **Prediction:** Accuracy tăng theo T nhưng giảm dần (diminishing returns): `64 < 128 < 256`, bước 64→128 lớn hơn bước 128→256. Thời gian train và bộ nhớ tăng nhưng *ít hơn 4×* mỗi lần gấp đôi T.
- **Reason:** IMDb review dài (trung bình ~300 token WordPiece) nên T nhỏ cắt mất nội dung; nhưng sentiment thường lộ ra ngay từ vài câu đầu và nhiều từ cảm xúc lặp lại. Chi phí attention `O(T²)` nhưng FFN và embedding tuyến tính theo T nên tổng thời gian tăng ít hơn bậc 2.
- **Expected evidence:** khoảng 87% / 90% / 92% cho T = 64 / 128 / 256; thời gian ~0.5× / 1× / ~2.2×.
- **Confidence:** 65%.

### Exp 5 — Domain shift (train IMDb → test Amazon product reviews, Rotten Tomatoes)
- **Prediction:** Performance giảm so với in-domain; giảm nhiều hơn với tập khác domain hơn (Amazon), ít hơn với tập cùng domain phim nhưng khác style (Rotten Tomatoes, câu rất ngắn) — hoặc ngược lại nếu style ngắn gây khó hơn. Model from scratch giảm nhiều hơn pretrained.
- **Reason:** Pretraining trên Wikipedia/sách giúp biểu diễn tổng quát (từ cảm xúc "great", "terrible" dùng chung giữa các domain) nhưng head và lớp trên được fine-tune trên review phim nên mang theo từ vựng và phong cách riêng (tên phim, "plot", "acting"); pretrained model không "hiểu mọi domain".
- **Expected evidence:** pretrained BERT: IMDb ≈ 92%, Amazon ≈ 85–90%, RT ≈ 80–85%; Transformer nhỏ: giảm 10 điểm trở lên.
- **Confidence:** 60%.
