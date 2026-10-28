# Prediction trước experiment — LAB 06

> ✍️ Viết **trước khi** xem kết quả notebook. Mỗi mục: Prediction / Reason / Confidence (0–100%).
> AI policy: phần này sinh viên tự viết, không để AI thay thế.

## Prediction 1 — Một key có dot product lớn hơn các key khác thì attention weight của nó thế nào?
- **Prediction:** Attention weight sẽ cao hơn
- **Reason:** Dot product càng lớn thì sự tương đồng càng cao, dẫn đến attention weight càng lớn
- **Confidence:** 80%

## Prediction 2 — Tất cả scores giống nhau, `s = [1, 1, 1, 1]`
- **Prediction:** Attention weights sẽ giống nhau
- **Reason:** Nếu tất cả các scores đều bằng nhau, thì sau khi áp dụng softmax, các attention weights cũng sẽ bằng nhau
- **Confidence:** 90%

## Prediction 3 — Một score lớn hơn rất nhiều, `s = [1, 1, 10]`
- **Prediction:** Attention weight của token có score lớn nhất sẽ cao hơn nhiều
- **Reason:** Score lớn hơn nhiều sẽ dẫn đến attention weight cao hơn khi áp dụng softmax
- **Confidence:** 95%

## Prediction 4 — Sequence tăng từ 10 lên 1000 tokens, chi phí attention thay đổi thế nào?
- **Prediction:** Chi phí attention sẽ tăng
- **Reason:** Khi số lượng tokens tăng, số lượng phép tính attention cũng tăng theo, dẫn đến chi phí tính toán tăng
- **Confidence:** 85%

## Prediction 5 — Attention có luôn chọn đúng token mà con người cho là quan trọng nhất không?
- **Prediction:** Không nhất thiết
- **Reason:** Attention dựa trên các tính toán và mô hình, có thể không luôn chọn đúng token mà con người cho là quan trọng nhất
- **Confidence:** 70%
