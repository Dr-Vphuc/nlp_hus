# Prediction trước experiment — LAB 09

> Viết **trước khi** chạy bất kỳ baseline / fine-tuning nào (chưa có số liệu nào của CoNLL-2003 tại thời điểm viết).
> Mỗi mục: Prediction / Reason / Confidence (0–100%).
> AI policy: phần này là suy luận dự đoán, cần sinh viên đọc lại và chỉnh theo ý hiểu của mình.

## Prediction 1 — Pretrained Transformer thay cho classifier đơn giản dựa trên word features thì NER thay đổi thế nào?
- **Prediction:** Entity-level F1 tăng rõ rệt (dự đoán baseline LR khoảng 0.60–0.75 F1, Transformer khoảng 0.88–0.92 F1). Phần tăng chủ yếu đến từ ORG/MISC/LOC mơ hồ và entity chưa từng thấy.
- **Reason:** Baseline chỉ nhìn một cửa sổ từ cố định (word hiện tại, từ trước/sau, hình dạng chữ) và phải "nhớ" từng từ qua bảng trọng số, nên gặp từ mới (OOV) chỉ dựa được vào đặc trưng bề mặt (viết hoa, hậu tố). Pretrained Transformer có (i) biểu diễn contextual — cùng một từ ở context khác nhau có vector khác nhau, (ii) kiến thức về thế giới học từ pretraining (biết "Paris" là địa danh dù chưa thấy trong train), (iii) nhìn toàn bộ câu qua self-attention thay vì cửa sổ cố định.
- **Confidence:** 85%

## Prediction 2 — Entity dài nhiều token dễ hay khó hơn entity một token? (`Microsoft` vs `United Nations Security Council`)
- **Prediction:** Entity dài **khó hơn** theo entity-level F1 (F1 của entity 1 token cao nhất; entity 3 và 4+ token thấp hơn).
- **Reason:** Entity-level match đòi hỏi **mọi** token trong span đúng cả type lẫn B/I; chỉ cần sai một token (ví dụ dừng sớm, thêm một token thừa, hoặc đổi I-ORG thành I-LOC) thì cả entity tính sai. Xác suất đúng cả span xấp xỉ tích xác suất đúng từng token nên giảm khi độ dài tăng. Ngoài ra entity dài thường có từ chức năng (`of`, `the`) hoặc từ viết thường ở giữa, và có thể bị chia thành nhiều subword. Mặt khác, entity 1 token như `Microsoft` cũng có thể sai type (ORG hay MISC/LOC), nên mức chênh có thể không lớn; có thể entity dài (tên tổ chức nhiều từ) lại có nhiều ngữ cảnh hỗ trợ nên ngoại lệ vẫn có thể xảy ra ở 2 token (tên người "First Last" thường rất dễ).
- **Confidence:** 65% (xu hướng chung giảm theo độ dài; nhóm 2 token có thể cao nhất vì chủ yếu là tên người)

## Prediction 3 — Entity hiếm xuất hiện trong training data có gây lỗi nhiều hơn không?
- **Prediction:** Có, entity hiếm / chưa thấy trong train bị lỗi nhiều hơn rõ rệt, cả ở baseline lẫn Transformer; khoảng cách giữa hai model sẽ lớn nhất ở nhóm entity chưa thấy.
- **Reason:** Baseline dựa trên đặc trưng của từ cụ thể (word identity) nên entity chưa thấy chỉ còn lại đặc trưng chung (viết hoa, hậu tố, ngữ cảnh), dễ nhầm type. Transformer có kiến thức pretraining và context nên bù được phần nào, nhưng với entity hiếm, đặc biệt MISC và ORG mơ hồ (tên đội thể thao trùng tên thành phố), vẫn dễ sai type; tên riêng bị chia thành nhiều subword lạ cũng làm biểu diễn kém ổn định hơn.
- **Confidence:** 80%
