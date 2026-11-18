# Reflection — LAB 08

> Câu hỏi của đề: *Sau LAB 07 ta đã biết xây dựng Transformer. Sau LAB 08, tại sao trong thực tế người ta thường không bắt đầu bằng việc train Transformer từ đầu?*

Sau LAB 07, mình biết kiến trúc Transformer chỉ là *khung*; kiến thức ngôn ngữ nằm trong trọng số. Lab này cho số liệu để thấy vì sao người ta hiếm khi khởi tạo ngẫu nhiên.

**Dữ liệu.** Transformer from scratch phải học cả từ vựng, cú pháp lẫn bài toán sentiment chỉ từ dữ liệu có nhãn. Với 10,000 review, Transformer nhỏ from scratch đạt 0.748 accuracy, DistilBERT cùng kiến trúc nhưng random init đạt 0.800, còn bản pretrained + fine-tuning đạt 0.870. Với 10% dữ liệu (1,000 review), scratch tụt còn 0.595 trong khi pretrained vẫn giữ 0.797. Nhãn thì đắt, văn bản thô thì rẻ, nên pretraining trên văn bản không nhãn "trả trước" phần lớn việc học.

**Pretrained representation.** Điều bất ngờ nhất là frozen: chỉ train một lớp Linear 1,538 tham số trên vector `[CLS]` cố định vẫn đạt 0.792, cao hơn Transformer nhỏ train toàn bộ (0.748), tức là encoder đã mã hóa sẵn thông tin hữu ích cho sentiment. Fine-tuning toàn bộ còn cao hơn 7.9 điểm, nên biểu diễn pretrained là điểm xuất phát tốt chứ không phải đích.

**Chi phí tính toán.** Pretraining rất đắt nên ta tái sử dụng thay vì làm lại; fine-tuning chỉ tốn vài chục phút trên GPU 6 GB. Mình cũng thấy chi phí `O(T²)` của attention chưa chiếm ưu thế ở T ≤ 512: phần attention chỉ chiếm khoảng 10% FLOPs, thời gian đo tăng khoảng 2× khi T tăng 8×. Do GPU bị dùng chung, mình phải thu nhỏ quy mô (DistilBERT, 10,000 review train, 5,000 review test).

**Downstream task và fine-tuning.** Head Linear là phần riêng của task, encoder dùng chung; sang NER (LAB 09) chỉ cần đổi head rồi fine-tune. Cái giá là phải giữ một bản model cho mỗi task.

**Hạn chế và domain shift.** Pretrained không có nghĩa là "hiểu mọi thứ". Model fine-tune trên IMDb giảm từ 0.870 xuống 0.791 trên Rotten Tomatoes (câu ngắn), dự đoán positive 58.7% trong khi nhãn là 50%, và confidence kém hiệu chỉnh hơn (ECE 0.019 → 0.042). Model vẫn sai với mỉa mai, phủ định, nhãn mơ hồ và review dài bị cắt (accuracy 0.912 với review ≤128 token so với 0.795 với review >512 token). Trên Amazon model đạt 0.914, nhưng đó có thể do review ngắn và rõ ràng hơn chứ chưa chứng minh được độ bền với shift.

**Lưu ý độ tin cậy.** Mọi so sánh chỉ có 1 seed và test 5,000 mẫu (sai số khoảng ±1 điểm); bảng thời gian train bị nhiễu do GPU dùng chung. Mình chỉ kết luận "pretrained có lợi rõ ràng trong thí nghiệm này", không phải "luôn tốt hơn".

Tóm lại, người ta không train từ đầu vì (1) nhãn khan hiếm, (2) biểu diễn pretrained đã chứa nhiều kiến thức, (3) tái sử dụng rẻ hơn pretraining rất nhiều, (4) một encoder phục vụ nhiều task. Bù lại ta chấp nhận domain shift, thiên lệch và chi phí lưu trữ/suy luận của model lớn.
