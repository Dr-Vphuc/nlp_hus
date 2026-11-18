# Individual Learning Check — LAB 08

> 12 câu hỏi ngắn của đề (mục 27), trả lời bằng lời của mình. Các câu có liên quan đến số đo thì trích từ `results.csv` / `fine_tuning.ipynb`.

**1. Pretraining khác fine-tuning ở đâu?**
Pretraining: huấn luyện encoder trên lượng văn bản rất lớn, không cần nhãn của bài toán đích, với mục tiêu tổng quát (ở BERT là Masked Language Modeling [+ NSP]); kết quả là một bộ trọng số `θ_pretrained` biểu diễn ngôn ngữ nói chung. Fine-tuning: lấy `θ_pretrained` làm điểm bắt đầu, thêm head cho task, rồi tiếp tục huấn luyện trên tập có nhãn nhỏ của bài toán cụ thể (`θ_pretrained → θ_task`) với learning rate nhỏ. Khác nhau về dữ liệu (lớn, không nhãn vs nhỏ, có nhãn), mục tiêu (dự đoán token bị che vs nhãn sentiment) và chi phí.

**2. MLM là gì?**
Che ngẫu nhiên một phần token (BERT: ~15%) bằng `[MASK]` rồi bắt model dự đoán token gốc từ ngữ cảnh *hai phía*: `P(w_t | w_{≠t})`. Loss là cross-entropy `-Σ_{i∈M} log P(y_i|x)` chỉ tính tại các vị trí bị che. Vì nhãn chính là văn bản gốc nên không cần gán nhãn thủ công (self-supervised).

**3. Tại sao cần tokenizer?**
Model chỉ nhận số. Tokenizer chuyển văn bản thành chuỗi ID cố định với vocab đã dùng lúc pretrain (nếu đổi tokenizer thì embedding pretrained không còn ý nghĩa), đồng thời thêm special token, cắt/đệm về độ dài và tạo attention mask.

**4. Subword token giải quyết vấn đề gì?**
Vấn đề từ ngoài từ điển (OOV) và vocab quá lớn: từ hiếm/mới được tách thành mảnh đã biết (`tokenization → token ##ization`), nên gần như không bao giờ phải dùng `[UNK]` cho từ thông thường, vocab giữ ở cỡ 30k, các từ cùng gốc chia sẻ mảnh. Cái giá là chuỗi dài hơn và từ hiếm bị vỡ vụn (ví dụ tên tiếng Việt trong notebook tokenization).

**5. `[CLS]` thường được sử dụng như thế nào?**
`[CLS]` là token đặc biệt được thêm vào đầu chuỗi; qua các lớp self-attention vector của nó tổng hợp thông tin toàn câu, nên với bài toán phân loại ta lấy hidden state của `[CLS]` ở lớp cuối (`h[:,0]`) → Dropout → Linear → softmax. (Nó không "tự nhiên" đại diện cho câu nếu chưa fine-tune; chính fine-tuning dạy head và encoder dùng nó cho task.)

**6. Attention mask dùng để làm gì?**
Báo cho attention biết token nào là thật (1) và token nào là `[PAD]` (0) để không cho padding tham gia làm key/value, tránh biểu diễn bị nhiễu bởi padding. Trong notebook tokenization: cosine giữa `[CLS]` của câu có padding và không padding = 1.0 khi mask đúng, nhưng chỉ 0.60 khi bỏ qua mask.

**7. Frozen encoder nghĩa là gì?**
Trọng số encoder cố định (`requires_grad=False`, không cập nhật, thường chạy ở eval/no_grad); chỉ head (Linear 768×2+2) được học. Khi đó encoder làm *feature extractor* và biểu diễn không đổi qua training.

**8. Fine-tuning thay đổi những tham số nào?**
Tất cả tham số có `requires_grad=True`: embedding, mọi lớp Transformer (attention Q/K/V/O, FFN, LayerNorm) *và* head mới. Với DistilBERT trong lab: 66.4M tham số trainable so với 1,538 của frozen. (Có phương án chỉ fine-tune một phần lớp trên / adapter, nhưng lab này fine-tune toàn bộ.)

**9. Vì sao pretrained model có thể hữu ích khi downstream dataset nhỏ?**
Phần lớn kiến thức ngôn ngữ (nghĩa từ, cú pháp, phủ định, ngữ cảnh) đã được học từ dữ liệu không nhãn khổng lồ, nên downstream chỉ cần học ánh xạ nhỏ từ biểu diễn sang nhãn; số tham số "phải học từ đầu" rất ít (head) và phần còn lại chỉ điều chỉnh nhẹ. Model from scratch phải học cả embedding lẫn cấu trúc từ vài nghìn mẫu nên dễ overfit. (Đối chiếu kết quả data-efficiency trong README.)

**10. Pretrained model có đảm bảo không bị domain shift không?**
Không. Pretraining chỉ làm biểu diễn tổng quát hơn nhưng không bao phủ mọi domain/style; head và phân phối nhãn học từ domain huấn luyện nên khi chuyển domain (review phim → product review, câu rất ngắn) performance vẫn đổi (xem Exp 5 trong README).

**11. Vì sao tăng sequence length làm chi phí attention tăng?**
Mỗi token attend tới mọi token khác nên ma trận score là `T × T` mỗi head: số phép tính và bộ nhớ của attention tỉ lệ `O(T²)`. Gấp đôi T → ×4 cho phần attention (phần FFN và embedding chỉ ×2), vì vậy tổng thời gian tăng giữa tuyến tính và bậc hai tùy T so với `d_model`.

**12. Confidence cao có đảm bảo prediction đúng không?**
Không. `confidence = max softmax` chỉ là độ chắc chắn của model, không phải xác suất đúng đã được kiểm chứng; model có thể sai với confidence rất cao (ví dụ nhãn nhiễu, mỉa mai, review nói về phim khác). Xem `confidence_analysis.png` và `error_analysis.md` để biết số liệu cụ thể của lab này.
