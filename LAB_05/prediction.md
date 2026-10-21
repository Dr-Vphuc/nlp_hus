# Prediction trước experiment — LAB 05 (§10)

> Ghi **trước khi** chạy notebook / xem `results.csv` (AI policy §27: tự làm, không dùng AI).
> Cột "Kiểm chứng" điền **sau** khi có kết quả.

## Prediction 1
Giữa **TF-IDF + Logistic Regression** và **RNN + Embedding**, model nào có khả năng xử lý word order trực tiếp hơn? Giải thích.

- **Prediction:** RNN + Embedding
- **Reason:** Vì vector embedd đầu ra của RNN dựa mạnh mẽ vào thứ tự của các từ trong sequence, trong khi TF-IDF chỉ dựa trên tần suất xuất hiện của từ mà không quan tâm đến vị trí hay thứ tự.
- **Confidence:** 90%

## Prediction 2
Giữa **RNN** và **LSTM**, model nào có khả năng xử lý dependency dài tốt hơn?
Giải thích bằng cơ chế model, không chỉ nói "LSTM hiện đại hơn".

- **Prediction:** LSTM
- **Reason:** LSTM sử dụng các gate (input, forget, output) để kiểm soát việc lưu trữ và truy xuất thông tin từ các bước thời gian trước, giúp model có thể học được các dependency dài hơn so với RNN đơn giản.
- **Confidence:** 95%

## Prediction 3
Nếu sequence length tăng từ 50 → 200, training của RNN sẽ thay đổi như thế nào?

- **Prediction:** RNN sẽ gặp khó khăn hơn.
- **Reason:** Khi sequence length tăng, RNN sẽ gặp phải vấn đề vanishing gradient, khiến việc học các dependency dài trở nên khó khăn hơn.
- **Confidence:** 90%

## Prediction 4
Nếu truncate toàn bộ review về 30 tokens, thông tin classification có thể bị ảnh hưởng không?

- **Prediction:** Có, thông tin classification có thể bị ảnh hưởng.
- **Reason:** Việc truncate sẽ làm mất mát thông tin quan trọng từ các từ ở cuối sequence, đặc biệt là trong các bài toán classification mà vị trí của các từ có thể ảnh hưởng đến kết quả.
- **Confidence:** 90%

## Prediction 5
LSTM có chắc chắn cho accuracy cao hơn RNN trên mọi dataset không?

- **Prediction:** Có, LSTM có thể có accuracy cao hơn RNN trên mọi dataset.
- **Reason:** LSTM có khả năng học được các dependency dài tốt hơn nhờ vào cơ chế gate, giúp model giữ được thông tin từ các bước thời gian trước.
- **Confidence:** 90%

---

## Kiểm chứng (điền sau khi chạy notebook)

| Prediction | Kiểm chứng ở | Đúng / sai / một phần | Ghi chú |
|---|---|---|---|
| 1 | Notebook mục 6 (shuffle test, cặp câu cùng bag-of-words) | | |
| 2 | Mục 7 (Exp 4), mục 9 (Exp 6) | | |
| 3 | Mục 8 (Exp 2/5 — RNN với max_length 50 → 200) | | |
| 4 | Mục 8 (max_length = 30) | | |
| 5 | Mục 7, mục 8 | | |
