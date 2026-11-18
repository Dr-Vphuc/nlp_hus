# Reflection — LAB 07 (Transformer → BERT)

## 1. Từ self-attention (LAB 06) đến Transformer, phần nào là mới?
Self-attention ở LAB 06 là `softmax(QKᵀ/√d_k)V` với một head và không có thông tin vị trí. Transformer Encoder block thêm: (1) **multi-head** (nhiều bộ `W_Q, W_K, W_V` song song, concat rồi `W_O`), (2) **positional encoding** cộng vào embedding, (3) **FFN** theo từng vị trí, (4) **residual + LayerNorm** sau mỗi sub-layer. Bài tính tay Bài 6 cho thấy một head chỉ là attention của LAB 06 với các ma trận chiếu, không có gì thần bí; phần khó là giữ đúng shape (`B×h×T×d_k`, scores `B×h×T×T`).
Implementation `transformer_block.py` được kiểm tra bằng cách chép trọng số sang `nn.TransformerEncoderLayer`: output chênh nhau tối đa `3.6e-7`, và token `<pad>` nhận attention weight đúng bằng 0.

## 2. Vì sao cần positional encoding? Experiment nói gì?
Self-attention là hàm của *tập* token (permutation-equivariant). Permutation experiment: "dog bites man" vs "man bites dog" — không PE thì cùng một từ ở hai câu cho output giống nhau tới `1e-7` và pooled representation có cosine = 1.000000; có PE thì output theo từng token khác nhau (tới ~1 trong block đầy đủ, 3–5e-2 với MHSA đơn). Trên IMDb: model không PE cho P(pos) = 0.0988 **y hệt** trên câu gốc và câu bị xáo trộn.
Tuy nhiên có một điều tôi không dự đoán được: **trên IMDb, PE gần như không giúp ích**: accuracy 0.8394 (có PE) vs 0.8406 (không PE), F1 0.8419 vs 0.8402, đều nằm trong dao động giữa các seed (≈0.005). Model có PE cũng gần như bỏ qua thứ tự: câu xáo trộn chỉ đổi P(pos) từ 0.0334 sang 0.0345. Giải thích có thể (chưa kiểm chứng thêm): sentiment trên IMDb phần lớn được quyết định bởi các từ cảm xúc rời rạc, nên một model kiểu bag-of-words đã đủ tốt; một block, mean pooling và 8 epoch không đủ để học cách dùng thứ tự (ví dụ "not good"). Kết luận đúng là: *PE là điều kiện cần để model **có thể** biết thứ tự*, không phải bảo đảm model **sẽ** dùng thứ tự để tăng accuracy trên mọi bài toán.

## 3. Multi-head: thực sự mang lại gì?
- Khi `d_model = 128` cố định, tăng số head (1, 2, 4, 8) làm `d_k` giảm (128, 64, 32, 16), **số tham số không đổi** (2,692,865) vì `W_Q, W_K, W_V, W_O` vẫn là `128×128`. Tăng số head **không** tăng chiều biểu diễn.
- Accuracy 0.8433 / 0.8360 / 0.8394 / 0.8398 (F1 0.8436 / 0.8421 / 0.8419 / 0.8406): chênh ≤ 0.007, tức cỡ nhiễu giữa seed. Prediction 5 đúng: không có bằng chứng rằng nhiều head hơn thì tốt hơn trên bài toán này; 1 head thậm chí cao nhất (nhưng chênh lệch không đáng tin).
- Head diversity (`attention_heads.png`): sau khi train, các head khác nhau rõ rệt (cosine trung bình giữa các cặp attention matrix = 0.14, so với 0.82 lúc khởi tạo khi attention gần uniform). Trên câu ví dụ, head 1 và 3 dồn vào "terrible", head 2 dồn vào "not"/"but", head 4 vào "acting"/"I"/"did". Tôi **không** đặt tên head ("head 1 = negation") vì chưa có bằng chứng can thiệp. Có một số chồng lấn (head 1 và 3, cosine 0.53 trên câu đó) nên không thể nói các head hoàn toàn độc lập. Entropy chuẩn hóa 0.68–0.80 (1 = uniform): attention vẫn khá phân tán, khoảng cách `|i−j|` trung bình ≈ 33 (gần với giá trị 33 của attention đều trên 100 token), tức các head không có thiên lệch "nhìn gần" rõ ràng.

## 4. Vai trò FFN, residual, LayerNorm (và những gì tôi chỉ giải thích, chưa ablation)
- **FFN:** biến đổi phi tuyến *trong từng vị trí* (`W₂ σ(W₁x+b₁)+b₂`), không trộn token; attention trộn token, FFN xử lý thông tin sau khi trộn. Chiếm 131,712 trong 198,272 tham số của block (`d_ff = 512`), tức nhiều tham số hơn attention (66,048).
- **Residual:** `x + F(x)` giữ đường truyền trực tiếp, layer chỉ cần học phần chỉnh, gradient chảy thẳng về lớp dưới (`∂y/∂x = I + ∂F/∂x`).
- **LayerNorm:** chuẩn hóa theo chiều feature của từng token (khác BatchNorm), giữ scale ổn định.
- Lab này **không** ablation riêng residual/LayerNorm/FFN (chỉ ablate PE), nên các vai trò trên là lý thuyết + kiểm tra thực thi (block khớp với `nn.TransformerEncoderLayer`), chưa phải bằng chứng thực nghiệm.

## 5. Transformer vs LSTM: kết quả và thành phần nào gây khác biệt
| | LSTM | Transformer (1 block, 4 head, mean pooling) |
|---|---|---|
| Params | 2,692,481 | 2,692,865 |
| Seed 42: accuracy / F1 | 0.8090 / 0.8170 | 0.8394 / 0.8419 |
| 3 seed (42, 43, 44): accuracy | 0.8129 ± 0.0034 | 0.8392 ± 0.0050 |
| 3 seed: F1 | 0.8134 ± 0.0053 | 0.8388 ± 0.0101 |
| best epoch (val loss) | 6 | 3 |

- Trên 3 seed, Transformer cao hơn LSTM ở cả 3 seed (hơn 1.9 đến 3.0 điểm accuracy, trung bình ≈ 2.6) (0.8394/0.8341/0.8441 vs 0.8090/0.8155/0.8143), lớn hơn độ lệch chuẩn nên khá nhất quán, nhưng 3 seed vẫn ít và đây chỉ là một dataset/cấu hình. Không có mục tiêu "Transformer phải thắng"; kết quả này chỉ nói rằng ở cấu hình này nó tốt hơn.
- Khác biệt có thể đến từ: (a) mean pooling của mọi vị trí (mọi token đóng góp trực tiếp) so với LSTM chỉ dùng final hidden state sau 200 bước (token đầu phải "sống sót" qua nhiều bước), (b) token tương tác trực tiếp qua attention, (c) residual/LayerNorm giúp tối ưu dễ hơn, (d) lưu ý model LSTM không có pooling nên khác nhau không chỉ ở "attention". Lab không ablate từng yếu tố, nên không kết luận yếu tố nào quan trọng nhất; thử mean pooling trên chuỗi hidden state của LSTM sẽ tách được (b) khỏi (a).
- **Learning curves (`learning_curves.png`):** cả hai model đều **overfit**. Transformer hội tụ rất nhanh nhưng overfit sớm: val loss tốt nhất ở epoch 3 (0.338), sau đó tăng tới 0.733 ở epoch 8 trong khi train loss giảm xuống 0.047. LSTM chậm hơn và có val loss dao động (epoch 3: 0.590; best epoch 6: 0.414; epoch 8: 0.524). Vì chọn checkpoint theo val loss nên test không bị ảnh hưởng nhiều, nhưng với Transformer, 8 epoch là dư; accuracy validation vẫn giữ ≈0.85 dù loss tăng, nghĩa là model trở nên quá tự tin ở các mẫu sai chứ không hẳn tệ đi về nhãn.
- Chi phí huấn luyện: min epoch time trên GPU là 17.8 s (LSTM) và 17.6–26.9 s (Transformer), tổng 3 seed LSTM 326 ± 39 s và Transformer 207 ± 28 s. Nhưng GPU được dùng chung với các tiến trình khác (thời gian mỗi epoch dao động 17–62 s cho cùng một cấu hình), nên số đo thời gian **chỉ mang tính tham khảo**, tôi không kết luận Transformer nhanh hơn từ các số này.

## 6. Chi phí `O(T²)` và giới hạn
- Ma trận attention `T×T`; T gấp đôi ⇒ 4× số phần tử. Theo thực nghiệm (peak GPU memory, `B=16`, forward+backward, `sequence_length.png`): Transformer 3.6 → 10.3 → 26.0 → 73.3 → 275 → 1063 → 4175 MB khi T = 32 → 2048 (xấp xỉ ×4 mỗi lần gấp đôi T ở T lớn), trong khi LSTM chỉ 12 → 184 MB. Tức attention memory là thứ nổi lên đầu tiên khi tăng T. Thời gian đo được quá nhiễu để kết luận.
- Ở Exp 5, `max_length` 32/64/128/256 cho F1 0.697 / 0.764 / 0.811 / 0.854 (giữ lại 13.7% / 27.1% / 51.2% / 76.7% số token) và attention memory (lý thuyết, B=64, 4 head) 1 / 4 / 16 / 64 MB. F1 tăng đều theo độ dài, nên ở đây thêm ngữ cảnh *có ích* và đánh đổi là bộ nhớ tăng gấp 4 lần mỗi lần gấp đôi độ dài. Vậy câu trả lời cho "tại sao không giải quyết mọi vấn đề bằng cách tăng độ dài" là: chi phí tăng bậc hai, trong khi lợi ích có thể bão hòa (không đo được ở T > 256 trên IMDb trong lab này, nên đừng suy diễn thêm).

## 7. Attention ≠ explanation
Attention visualization chỉ cho thấy phân phối trọng số của một block, sau đó còn `W_O`, residual, LayerNorm, FFN, mean pooling và Linear. Trong error analysis, 6 mẫu: có mẫu từ cảm xúc không nằm trong top-5 attention mà dự đoán vẫn đúng (#16510), và có mẫu từ cảm xúc nằm trong top-5 mà dự đoán vẫn sai (#16701: `lacks`, `great`). Vì vậy tôi chỉ dùng attention làm *giả thuyết* để đối chiếu với hành vi model, không dùng nó làm lời giải thích (muốn kiểm chứng phải làm thí nghiệm can thiệp, ví dụ xóa/thay token và đo lại xác suất, chưa làm trong lab này).

## 8. Đường đi LAB 01 → LAB 07: mỗi bước giải quyết limitation nào?
```
TF-IDF      → Limitation: sparse, độ dài = |vocab|, không có ý nghĩa/độ tương đồng ngữ nghĩa giữa từ ("good" ≠ "great"), mất thứ tự từ.
Word2Vec    → Limitation: dense, có ngữ nghĩa, nhưng mỗi từ chỉ có MỘT vector tĩnh (không phụ thuộc ngữ cảnh), và câu thường chỉ được biểu diễn bằng trung bình (mất thứ tự).
LSTM        → Limitation: biểu diễn theo ngữ cảnh và có thứ tự, nhưng xử lý tuần tự (không song song theo T), thông tin xa phải đi qua nhiều bước, final hidden state là điểm nghẽn thông tin.
Attention   → Limitation: token tương tác trực tiếp (không bị bottleneck, song song), nhưng (LAB 06) không có thông tin vị trí và chỉ một "góc nhìn" attention; tốn O(T²).
Transformer → Remaining limitation: vẫn O(T²) về thời gian/bộ nhớ theo độ dài; học từ đầu trên dữ liệu nhỏ (bài này overfit sau 3 epoch); cần positional encoding cố định tối đa độ dài; mỗi bài toán lại phải train lại từ đầu, không tận dụng được kiến thức ngôn ngữ từ lượng văn bản khổng lồ.
```

## 9. Chuẩn bị cho LAB 08 (BERT)
Từ lab này tôi đã biết: `[CLS]`-style pooling là một lựa chọn, nhưng kết quả cho thấy khác biệt với mean pooling không đáng kể (accuracy 0.8364 vs 0.8394; F1 0.8380 vs 0.8419, nhỏ hơn dao động giữa seed; CLS cũng cần thêm một token và nhiều epoch hơn để học vector `[CLS]` từ đầu); positional information nằm ở đầu vào; mỗi block gồm MHSA, Add&Norm, FFN, Add&Norm. Các hạn chế quan sát được (overfit nhanh sau epoch 3, bỏ qua thứ tự, OOV ví dụ `lumpen`, `kafkaesque`, dữ liệu train chỉ 19.8k review) chính là động lực để dùng một Transformer **đã được pretrain** trên văn bản lớn rồi fine-tune: ngoài vocabulary subword (giảm OOV), model đã biết trước thứ tự từ và ngữ cảnh trước khi thấy nhãn IMDb.
