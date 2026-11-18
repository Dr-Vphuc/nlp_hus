# Prediction trước experiment — LAB 07

> Viết **trước khi** chạy bất kỳ experiment nào (phần "Prediction / Reason / Confidence" bên dưới **không được sửa** sau khi xem kết quả).
> Phần "Kiểm chứng prediction" ở cuối file được điền sau khi có kết quả.

## Prediction 1 — Transformer có thể xử lý các token song song hơn RNN không?
- **Prediction:** Có. Transformer xử lý được tất cả các token của một sequence cùng lúc, còn RNN phải xử lý tuần tự từng timestep.
- **Reason:** Trong RNN, hidden state `h_t` phụ thuộc `h_{t-1}` nên không thể tính bước t trước khi xong bước t-1. Trong self-attention, `QKᵀ` và `AV` là các phép nhân ma trận trên toàn bộ T token, không có phụ thuộc giữa các vị trí trong cùng một lớp.
- **Confidence:** 90%

## Prediction 2 — Nếu bỏ positional encoding, Transformer có còn biết thứ tự token không?
- **Prediction:** Không. Nó chỉ thấy một tập (bag) token, không biết token nào đứng trước/sau.
- **Reason:** Self-attention chỉ dùng dot product giữa các vector embedding, không có chỗ nào đưa chỉ số vị trí vào; nếu hoán vị đầu vào thì đầu ra bị hoán vị tương ứng (permutation-equivariant), nên không có thông tin thứ tự.
- **Confidence:** 85%

## Prediction 3 — Hai câu cùng tập token nhưng khác thứ tự, Transformer không có positional information có phân biệt được không?
- **Prediction:** Không phân biệt được (biểu diễn của cả câu sau pooling sẽ giống hệt nhau; biểu diễn của từng token cũng giống nhau khi so sánh cùng một từ giữa hai câu).
- **Reason:** Cùng lý do như Prediction 2: output của từng token chỉ phụ thuộc vào embedding của nó và *tập* các token còn lại, không phụ thuộc thứ tự. Mean pooling lại là hàm đối xứng theo vị trí.
- **Confidence:** 90%

## Prediction 4 — Hai attention heads có nhất thiết học cùng một attention pattern không?
- **Prediction:** Không nhất thiết. Có thể khác nhau, nhưng cũng có thể một số head gần giống nhau (redundant).
- **Reason:** Mỗi head có ma trận chiếu `W_Q^i, W_K^i, W_V^i` khởi tạo ngẫu nhiên khác nhau nên pattern ban đầu khác nhau, nhưng không có loss nào ép các head phải khác nhau, nên sau khi train có thể có head trùng lặp.
- **Confidence:** 75%

## Prediction 5 — Tăng số attention heads có chắc chắn làm accuracy tăng không?
- **Prediction:** Không chắc chắn. Accuracy có thể gần như không đổi, tăng hoặc giảm nhẹ.
- **Reason:** Khi `d_model` cố định, tăng số head chỉ làm `d_k = d_model / h` nhỏ đi và số tham số gần như không đổi, tức là không tăng chiều biểu diễn. Với bài toán sentiment đơn giản như IMDb, sự khác biệt có thể nằm trong nhiễu giữa các lần chạy (mỗi cấu hình chỉ 1 seed).
- **Confidence:** 80%

## Prediction 6 — Transformer có luôn nhanh hơn RNN trên mọi sequence length không?
- **Prediction:** Không. Với sequence ngắn thì Transformer có thể nhanh hơn nhờ song song hóa, nhưng khi sequence dài thì chi phí `O(T²)` của attention có thể làm nó chậm/tốn bộ nhớ hơn.
- **Reason:** RNN có chi phí `O(T)` (tuần tự) còn self-attention `O(T²)` thời gian và bộ nhớ vì ma trận attention T×T. Trên GPU song song hóa bù được một phần, nhưng khi T rất lớn thì T² thắng. Trên CPU (máy chạy lab này không có GPU build của torch) lợi thế song song hóa càng ít.
- **Confidence:** 70%

---

# Kiểm chứng prediction (điền SAU khi có kết quả; các prediction phía trên không bị sửa)

| # | Dự đoán | Kết quả | Đúng/Sai | Giải thích |
|---|---|---|---|---|
| 1 | Transformer song song hơn RNN | Về cấu trúc: đúng, mỗi layer chỉ gồm các phép nhân ma trận trên cả `T` token, còn LSTM phải lặp `T` bước. Về đo đạc: thời gian không kết luận được, vì GPU/CPU dùng chung với các tiến trình khác khi chạy lab này nên timing rất nhiễu (xem Exp 5). Min epoch time (s) trên GPU: LSTM 17.8, Transformer 17.6–26.9 (tuỳ cấu hình); tổng 3 seed: LSTM 326 ± 39 s, Transformer 207 ± 28 s | Đúng (phần lý thuyết); phần đo đạc chỉ gợi ý | "Song song hơn" là tính chất của phụ thuộc dữ liệu (không có `h_{t-1}`), không đảm bảo luôn nhanh hơn trong thực tế. Số đo của tôi không đủ sạch để chứng minh. |
| 2 | Bỏ PE thì mất thứ tự | `permutation_results.csv`: không PE, cùng một từ ở hai câu hoán vị cho output khác nhau cỡ `1e-7` (sai số float), có PE cỡ `1e-2` đến `1`. Trên IMDb, model không PE cho P(pos) **y hệt** 0.0988 trên câu gốc và câu bị xáo trộn | Đúng | Self-attention + mean pooling là hàm của *tập* token. |
| 3 | Hai câu cùng token khác thứ tự không phân biệt được nếu thiếu PE | Không PE: cosine(pooled S1, pooled S2) = 1.000000, `max|diff|` ≈ `3e-8` (dog bites man vs man bites dog). Có PE: cosine ≈ 0.9993 (MHSA) và 0.9992 (block đầy đủ), `max|diff|` pooled 3e-2 và 1.2e-1; biểu diễn từng token "dog"/"man" chênh tới 0.97/1.31 trong block đầy đủ | Đúng | Chú ý: với trọng số ngẫu nhiên và mean pooling, độ khác biệt *của câu* nhỏ (cosine 0.9993, gần 1) nhưng khác không rõ rệt, còn không PE thì giống hệt tới sai số float. Biểu diễn từng token mới khác rõ. |
| 4 | Hai head không nhất thiết cùng pattern | Sau khi train, cosine trung bình giữa các cặp head = 0.14 (0.09–0.17); lúc khởi tạo là 0.82 (attention gần uniform). Trên câu ví dụ, head 1 và head 3 cùng dồn vào "terrible" (cosine 0.53) nên có chồng lấn một phần | Đúng | Các head khác nhau rõ rệt về trung bình, nhưng không loại trừ việc một số cặp có phần redundant (cosine 0.53 ở câu ví dụ). |
| 5 | Tăng số head không chắc tăng accuracy | 1/2/4/8 head: accuracy 0.8433 / 0.8360 / 0.8394 / 0.8398, F1 0.8436 / 0.8421 / 0.8419 / 0.8406. Số tham số giữ nguyên 2,692,865. Biên độ ≤ 0.007, tương đương độ lệch chuẩn giữa các seed (≈0.005) | Đúng | `d_model` cố định nên tăng head chỉ giảm `d_k`, không tăng chiều biểu diễn hay tham số; tốt nhất lại là 1 head. |
| 6 | Transformer không luôn nhanh hơn RNN | Bộ nhớ: peak GPU cho forward+backward tăng `~T²` (Transformer 3.6 MB → 4,175 MB khi T = 32 → 2048) so với LSTM tăng chậm hơn (12 → 184 MB); ở `T ≥ 128` Transformer đã tốn bộ nhớ hơn LSTM. Thời gian: có lần đo Transformer chậm hơn LSTM (CPU, T=1024: 1.76×; GPU, T=2048), nhưng nhiều điểm đo khác nhiễu nặng (vd GPU T=512 nhanh bất thường) | Đúng (nghiêng về bằng chứng bộ nhớ) | Chi phí `O(T²)` thể hiện rõ trong bộ nhớ (deterministic); thời gian đo được không đủ tin cậy để kết luận điểm giao nhau. |

Không có prediction nào bị sai, nhưng prediction 1 và 6 chỉ được hỗ trợ một phần bởi số đo vì máy chạy lab dùng chung tài nguyên. Ghi chú: câu "máy này không có GPU build của torch" trong Reason của prediction 6 viết trước khi tôi chạy; thực tế các run huấn luyện được thực hiện trên GPU (RTX 3050, CUDA build của torch), còn trên CPU Transformer vẫn nhanh hơn LSTM ở T ≤ 512 trong lần đo micro-benchmark, tức lý do "CPU ít lợi thế song song" không được kiểm chứng.
