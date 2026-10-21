# Reflection LAB 05

## 1. Sequence model khác static representation ở điểm nào?
So với TF-IDF / Average Word2Vec ở LAB 04, thêm hidden state $h_t = f(x_t, h_{t-1})$ thay đổi điều gì?
Bằng chứng nào trong notebook (mục 6, 7) cho thấy điều đó?

TF-IDF và Average Word2Vec biến cả review thành một vector cố định, không quan tâm từ nào đứng trước từ nào. Với RNN, $h_t$ phụ thuộc vào $h_{t-1}$, nên cùng một từ nhưng đứng ở vị trí khác hoặc sau những từ khác sẽ cho hidden state khác. Vì vậy model nhạy với thứ tự từ.

Bằng chứng trong notebook:
- Khi xáo trộn thứ tự từ trong test set, TF-IDF + LR không đổi prediction nào (0%), còn RNN đổi 32.0%, LSTM 11.7%, GRU 12.6%.
- "dog bites man" và "man bites dog" có cùng P(pos) với TF-IDF (0.3635), nhưng RNN cho hai giá trị khác nhau (0.44 và 0.79).

Tuy nhiên, nhạy với thứ tự không có nghĩa là hiểu đúng thứ tự. Ví dụ "not bad, actually quite good" vẫn bị cả RNN, LSTM, GRU đoán negative.

## 2. Prediction nào của mình sai? Vì sao?
(đối chiếu `prediction.md` với kết quả)

- **Prediction 5 sai.** Mình nghĩ LSTM luôn tốt hơn RNN trên mọi dataset. Trên IMDb thì LSTM đúng là tốt hơn RNN ở mọi max_length, nhưng lab chỉ thử một dataset nên không thể kết luận cho mọi dataset. GRU ít tham số hơn LSTM mà còn tốt hơn LSTM, và LSTM bị overfit sau vài epoch. Mình đã đánh đồng "cơ chế tốt hơn" với "kết quả luôn tốt hơn".
- **Prediction 3 đúng một phần.** Mình nghĩ tăng max_length từ 50 lên 200 sẽ làm RNN học khó hơn. Thực tế training tốn hơn (thời gian mỗi epoch từ 2.4s lên 10.1s, GPU memory từ 84 lên 211 MB) nhưng F1 lại tăng (0.580 lên 0.709) vì có thêm thông tin. Vanishing gradient không làm accuracy giảm mà thể hiện ở chỗ RNN gần như bỏ qua phần đầu review (98.7% độ nhạy nằm ở 50 token cuối).
- Prediction 1, 2, 4 đúng.

## 3. Mô hình tốt hơn có phải luôn là mô hình phức tạp hơn?
(accuracy / F1 so với số tham số, training time, memory, mục 7, 8, 10)

Không. TF-IDF + LR chỉ có khoảng 43 nghìn tham số, train mất 6 giây nhưng có F1 cao nhất (0.879). RNN, LSTM, GRU có khoảng 2 triệu tham số, train lâu hơn 7 đến 14 lần mà F1 chỉ 0.709 đến 0.829. Giữa các sequence model, GRU ít tham số hơn LSTM, train nhanh hơn (42.5s so với 74.7s), dùng ít memory hơn (276 so với 319 MB) mà F1 vẫn cao hơn (0.829 so với 0.823).

Với IMDb, sentiment phần lớn nằm ở các từ khen chê, nên đếm từ đã rất hiệu quả. Model phức tạp hơn chỉ có lợi khi bài toán thật sự cần thứ tự từ và có đủ dữ liệu để học, nếu không thì dễ overfit và tốn tài nguyên hơn.

## 4. Giới hạn của recurrent models
`x1 → x2 → x3 → ... → x100`: muốn dùng thông tin ở $x_1$ khi xử lý $x_{100}$, thông tin phải đi qua bao nhiêu bước?
Điều đó gây ra vấn đề gì về gradient, về tốc độ (có song song hóa được không)?

Thông tin từ $x_1$ phải đi qua 99 bước cập nhật hidden state mới tới $x_{100}$.

- **Gradient:** gradient từ $h_{100}$ về $h_1$ là tích của 99 Jacobian. Nếu các Jacobian nhỏ hơn 1 thì gradient tắt dần (vanishing), lớn hơn 1 thì bùng nổ (exploding). Trong notebook, RNN dồn 98.7% độ nhạy vào 50 token cuối, và gradient norm có lúc lên tới 115. LSTM và GRU đỡ hơn nhờ gate nhưng vẫn không giải quyết hết.
- **Tốc độ:** $h_t$ phải chờ $h_{t-1}$ tính xong nên không thể song song hóa theo chiều thời gian. Sequence càng dài thì càng chậm, ví dụ thời gian mỗi epoch của RNN tăng từ 2.4s ở max_length 50 lên 10.1s ở max_length 200.
- Toàn bộ review bị nén vào một vector $h_T$ có kích thước cố định, nên review càng dài càng dễ mất thông tin.

## 5. RNN → Attention
Có cách nào để một token trực tiếp "nhìn" những token quan trọng khác trong sequence mà không phải truyền toàn bộ thông tin qua một chuỗi recurrent dài? (§32)

Có, đó là attention. Mỗi token tính một điểm tương đồng với tất cả các token khác, rồi lấy tổng có trọng số của chúng. Như vậy $x_{100}$ có thể lấy thông tin trực tiếp từ $x_1$ chỉ trong một bước, không phải đi qua 99 bước như RNN. Gradient cũng đi thẳng giữa hai token nên không bị tắt dần theo khoảng cách. Ngoài ra, attention của các token có thể tính cùng lúc bằng phép nhân ma trận nên song song hóa được trên GPU. Đây là ý tưởng chính của self-attention và Transformer.

## 6. Nếu làm lại lab này, mình sẽ thay đổi gì?

- Dùng truncation tail hoặc giữ cả đầu và cuối review, vì ở max_length 100, truncation tail cho F1 0.833 so với 0.772 khi cắt head. Nhiều lỗi của GRU nằm ở review dài bị cắt mất phần đánh giá ở cuối.
- Dùng early stopping và train ít epoch hơn, vì GRU và LSTM bắt đầu overfit từ khoảng epoch 5.
- Thử pretrained embedding (Word2Vec ở LAB 04) thay vì học embedding từ đầu.
- Thay vì chỉ dùng $h_T$, thử lấy trung bình hoặc max pooling trên mọi $h_t$, hoặc dùng bidirectional RNN, để model không quá phụ thuộc vào các token cuối.
- Chạy mỗi cấu hình với vài seed khác nhau, vì RNN khá không ổn định (ví dụ max_length 50 còn kém hơn max_length 30).
