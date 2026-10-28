# Reflection — LAB 06

> ✍️ Sinh viên tự viết (AI policy).

## 1. Attention giải quyết vấn đề gì của recurrent processing?

RNN/LSTM đọc tuần tự và nén toàn bộ câu vào một hidden state cuối, nên thông tin ở đầu câu dễ bị mất (bottleneck, long-range dependency, vanishing gradient). Ngoài ra vì phụ thuộc vào bước trước nên không tính song song theo chiều thời gian được. Attention cho mỗi vị trí truy cập trực tiếp mọi token bằng weighted combination, không phụ thuộc khoảng cách và tính song song được. Trong lab, Attention train 21.9s so với 74.3s của LSTM (cùng 8 epoch, 1 seed), nhanh hơn khoảng 3.4 lần.

## 2. Attention tạo ra vấn đề mới nào?

- Chi phí O(T²) về thời gian và bộ nhớ vì phải tính ma trận T×T (phần tính tay: T=1000 là 1,000,000 phần tử). Peak GPU memory của Attention cũng cao hơn LSTM (255 MB so với 178 MB, ở T=200).
- Không có khái niệm thứ tự: self-attention là permutation-equivariant nên phải thêm positional information.
- Attention weights dễ bị hiểu sai là lời giải thích của model (xem mục 4).

## 3. Representation của LSTM (final hidden state) và Attention (weighted combination) khác nhau thế nào? Kết quả thực nghiệm nói gì?

LSTM dùng final hidden state: một vector được cập nhật tuần tự, token cuối ảnh hưởng mạnh nhất và thông tin sớm phải "sống sót" qua nhiều bước. Attention dùng context `c = Σ αᵢhᵢ`, tổ hợp có trọng số của biểu diễn tất cả token, nên mọi vị trí đều đóng góp trực tiếp. Kết quả test: Attention accuracy 0.8344 / F1 0.8296, LSTM 0.8214 / 0.8233, chênh khoảng 1.3 điểm accuracy. Mỗi model chỉ chạy 1 seed nên chưa đủ kết luận Attention tốt hơn về chất lượng; điều rõ ràng hơn là nó huấn luyện nhanh hơn. Ngoài ra pooling attention của model gần uniform (≈1/T), nên model hoạt động gần giống mean-pooling, và sự khác biệt có thể đến từ việc không phải nén qua chuỗi tuần tự chứ không hẳn vì "chọn đúng token".

## 4. Attention weights có phải là giải thích (reasoning) của model không? Vì sao?

Không nhất thiết. Weights chỉ cho biết mức đóng góp của từng vector vào context ở một lớp, còn dự đoán còn phụ thuộc embedding, các phép chiếu V, các tầng Linear phía sau; các phân phối attention khác nhau vẫn có thể cho cùng dự đoán. Trong lab, weights gần uniform (top-5 token đều ~0.01) và các token top như "should", "donald", "nazi" không phải lý do rõ ràng cho nhãn dự đoán. Vì vậy chỉ nên xem attention như một hypothesis rồi đối chiếu với hành vi model (ví dụ error analysis), không kết luận "model sai vì nhìn sai từ".

## 5. Vì sao "Attention tốt hơn RNN nên luôn thay thế RNN" là quan niệm sai?

- Kết quả của lab chỉ cho thấy chênh lệch nhỏ (~1.3 điểm accuracy, 1 seed) trên một bài toán và một cấu hình, không khái quát được.
- Attention tốn O(T²), với chuỗi rất dài hoặc tài nguyên hạn chế, RNN có chi phí O(T) và bộ nhớ trạng thái cố định, phù hợp streaming / thiết bị nhỏ.
- Model attention trong lab thiếu thông tin vị trí, nên với bài toán phụ thuộc thứ tự thì RNN lại có lợi thế vì thứ tự đã nằm sẵn trong cơ chế tuần tự.
- Lựa chọn nên dựa vào dữ liệu, độ dài chuỗi, ràng buộc tính toán và kết quả thực nghiệm nhiều seed, không phải theo "cái mới luôn hơn".

## 6. Model trong lab chưa có positional encoding — điều đó ảnh hưởng gì (ví dụ negation)? Còn thiếu gì để thành Transformer (LAB 07)?

Không có positional encoding thì model không biết thứ tự từ, tức là coi câu như một "bag of tokens" có trọng số. "I like this movie" và "I do not like this movie" chỉ khác nhau ở token "do not" nên model khó phân biệt phủ định gắn với từ nào; "not good" và "good not" cũng giống nhau. Trong phần Application, cả hai câu đều bị dự đoán negative với P(pos) ≈ 0.000, phù hợp với việc thiếu vị trí, nhưng đây mới là giả thuyết chưa kiểm chứng (còn có thể do câu ngắn khác xa review ~200 token lúc train). Trong error analysis, mẫu sai #18371 chứa nhiều từ phủ định/tiêu cực ("didn't", "don't", "violated") nhưng nhãn là positive, và #21895 có "terrible", "nazi" trong một review khen diễn xuất; hai mẫu này gợi ý model phụ thuộc vào từ cảm xúc rời rạc, nhưng chưa phải bằng chứng.

Để thành Transformer còn thiếu: positional encoding, multi-head attention, residual connection + layer normalization, feed-forward network theo từng vị trí, xếp chồng nhiều block, (decoder) causal mask và cross-attention nếu là seq2seq. Model hiện tại chỉ có một lớp self-attention một head rồi pooling.
