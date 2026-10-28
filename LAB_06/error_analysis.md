# Error analysis — LAB 06 (Attention model)

> Dữ liệu (text, nhãn, xác suất, top-5 token được pooling-attention) do notebook sinh ra (`error_samples.csv`).
> ✍️ Phần **Phân loại lỗi / Dự đoán token quan trọng / Đối chiếu attention / Giải thích** do bạn tự viết.
> Phân loại gợi ý: negation, long-range dependency, ambiguous wording, sarcasm, rare word, domain shift, truncation, noise.
> Attention chỉ được trình bày như **hypothesis** rồi đối chiếu với prediction; không kết luận "attention nhìn sai từ nên model sai".
> Lưu ý: pooling attention của model gần uniform (≈1/T), nên top-5 token chênh nhau rất ít.

## Mẫu #16074 — ĐÚNG
- **True / Pred:** positive / positive (P(pos) = 0.906, 170 tokens)
- **Top-5 attended:** should(0.01), should(0.01), disability(0.01), getting(0.01), only(0.01)
- **Text (300 ký tự đầu):** This is an incredible piece of drama and powerful which hits you. I found the film was great and getting to grips with the two main characters disability, this was represented in a great performance by both two Michael and Rory. Whether the story is based around a true story I feel the story was try
- **Loại lỗi / lý do đúng:** Đúng vì từ cảm xúc tích cực rõ ràng ('incredible', 'powerful', 'great', 'great performance'); không có phủ định hay mỉa mai trong đoạn đầu. Loại: không phải lỗi (câu dễ).
- **Token con người cho là quan trọng:** incredible, powerful, great, performance
- **Attention có tập trung vào các token đó không?** Không. Top-5 là should, should, disability, getting, only (đều 0.01 ≈ 1/T với T=170), không có từ cảm xúc nào. Weights gần uniform nên không phân biệt được.
- **Hypothesis giải thích prediction (và đối chiếu với behavior của model):** Hypothesis: nhiều từ tích cực xuất hiện lặp lại nên biểu diễn trung bình (gần mean-pooling) nghiêng về positive (P=0.906). Điều này khớp hành vi model: dự đoán dựa trên tổng hợp cả câu hơn là trên vài token được attend; attention weights không giải thích được quyết định.

## Mẫu #2153 — ĐÚNG
- **True / Pred:** negative / negative (P(pos) = 0.002, 160 tokens)
- **Top-5 attended:** mouth(0.01), vanessa(0.01), gay(0.01), gay(0.01), boring(0.01)
- **Text (300 ký tự đầu):** The plot for a movie such of this is a giveaway. How can you go wrong with a gay plot line and all the colors and music of India - a story like this writes itself. I'll watch most anything, but this was unwatchable. The sad thing is, the white folks are the most colorful in the film. Vanessa was a r
- **Loại lỗi / lý do đúng:** Đúng vì có tín hiệu tiêu cực rõ ('unwatchable', 'boring', 'sad thing'). Có chi tiết gây nhiễu ('How can you go wrong...' nghe như khen, 'I'll watch most anything, but...') nhưng không làm model lệch. Loại: không phải lỗi, có chút ambiguous/contrast.
- **Token con người cho là quan trọng:** unwatchable, boring, sad, but
- **Attention có tập trung vào các token đó không?** Chỉ một phần: 'boring' nằm trong top-5, còn 'mouth', 'vanessa', 'gay' là từ không liên quan; 'unwatchable' không xuất hiện trong top-5. Weights đều 0.01 nên không đáng kể.
- **Hypothesis giải thích prediction (và đối chiếu với behavior của model):** Hypothesis: các từ tiêu cực mạnh ('unwatchable', 'boring') chiếm ưu thế trong biểu diễn nên P(pos) = 0.002 rất thấp. Hành vi này phù hợp với cách model dựa vào từ cảm xúc mạnh. Chưa chứng minh được rằng attention (gần uniform) là nguyên nhân.

## Mẫu #19171 — ĐÚNG
- **True / Pred:** positive / positive (P(pos) = 0.995, 106 tokens)
- **Top-5 attended:** didn't(0.01), dread(0.01), dread(0.01), permission(0.01), permission(0.01)
- **Text (300 ký tự đầu):** of Adam's apple and With Your Permission. I didn't know until looking it up just now. Also the errrrrrmmmmm 'Wilbur Wants to Kill himself' I prefer this very focused deadpan ...drama over Adam's Apple's over the top comedic zeal. But With Your Permission is much more layered and subtle, but that's a
- **Loại lỗi / lý do đúng:** Đúng (P(pos)=0.995) nhưng văn bản có phần khó: 'didn't know', 'Wilbur Wants to Kill himself' (từ tiêu cực 'kill', 'dread' không phải cảm xúc về phim) và câu 'But ... much more layered and subtle'. Loại: negation / từ tiêu cực không liên quan đến cảm xúc, nhưng model vẫn đúng.
- **Token con người cho là quan trọng:** prefer, layered, subtle, focused
- **Attention có tập trung vào các token đó không?** Không. Top-5 là didn't, dread, dread, permission, permission (đều 0.01); các từ khen như 'prefer', 'layered', 'subtle' không có trong top-5.
- **Hypothesis giải thích prediction (và đối chiếu với behavior của model):** Hypothesis: các từ so sánh tích cực ('prefer', 'much more layered and subtle') đủ để đẩy xác suất lên cao dù có 'didn't', 'kill'. Hoặc model đoán đúng dựa vào dấu hiệu khác (độ dài ngắn hơn, chủ đề) mà chưa kiểm chứng. Độ tự tin rất cao (0.995) không đồng nghĩa với lý do đúng.

## Mẫu #2571 — SAI
- **True / Pred:** negative / positive (P(pos) = 0.783, 198 tokens)
- **Top-5 attended:** bad(0.01), should(0.01), donald(0.01), donald(0.01), cult(0.01)
- **Text (300 ký tự đầu):** "Curse of Michael Myers" is a very frustrating piece to deal with for a fan of the Halloween series. After a very disappointing letdown in Halloween 5, the series reached a near ebb in plot lines, with a silly devolution into witchcraft and a teenage cult dedicated to the worship of Michael Myers. "
- **Loại lỗi / lý do đúng:** Sai (negative → positive, P(pos)=0.783). Nội dung phần đầu rõ ràng tiêu cực ('frustrating', 'disappointing letdown', 'silly devolution'). Loại có thể: truncation (198 token, chỉ thấy phần đầu; phần sau chưa biết) / domain-specific (review fan của series, tên riêng 'Michael Myers', 'Halloween'). Chưa chắc là negation.
- **Token con người cho là quan trọng:** frustrating, disappointing, letdown, silly
- **Attention có tập trung vào các token đó không?** Không. Top-5 là bad, should, donald, donald, cult (đều 0.01); 'bad' có tính tiêu cực nhưng phần lớn là từ không liên quan, và các từ tiêu cực mạnh ('frustrating', 'disappointing') không có trong top-5.
- **Hypothesis giải thích prediction (và đối chiếu với behavior của model):** Hypothesis: phần còn lại của review (không thấy ở 300 ký tự đầu) có thể chứa từ khen (ví dụ nói về fan, điểm tốt của series) hoặc các từ như 'fan', 'series' bị gắn với positive trong dữ liệu train, làm kéo trung bình sang positive. Vì pooling gần uniform và không có thông tin vị trí nên model không nhấn mạnh được chỗ kết luận. Đây là giả thuyết, cần xem toàn văn và thử bỏ từng đoạn để kiểm chứng.

## Mẫu #18371 — SAI
- **True / Pred:** positive / negative (P(pos) = 0.381, 190 tokens)
- **Top-5 attended:** blame(0.01), detention(0.01), couldn't(0.01), violated(0.01), however(0.01)
- **Text (300 ký tự đầu):** I saw this movie, when it first came out. Patty didn't want to be separated from her siblings. However the Juvenile Justice system don't care. When Patty looked at the case file, the judge sent her to a Juvenile detention facility. He claimed she violated the rules of privacy, by looking at private 
- **Loại lỗi / lý do đúng:** Sai (positive → negative, P(pos)=0.381, gần ngưỡng). Đoạn đầu là tóm tắt cốt truyện với nhiều từ tiêu cực về mặt nội dung ('didn't want', 'don't care', 'detention', 'violated'), trong khi cảm xúc về phim chưa được nói rõ. Loại: ambiguous wording / plot summary (từ tiêu cực mô tả cốt truyện, không phải đánh giá), kèm negation.
- **Token con người cho là quan trọng:** chưa xác định được từ đánh giá nào trong 300 ký tự đầu (đoạn này chủ yếu tóm tắt cốt truyện); cần xem phần sau của review
- **Attention có tập trung vào các token đó không?** Có trùng nhưng vô nghĩa: top-5 là blame, detention, couldn't, violated, however, toàn là từ tiêu cực của cốt truyện nhưng chúng không phải ý kiến về phim. Weights đều 0.01 nên không có kết luận về việc 'nhìn sai từ'.
- **Hypothesis giải thích prediction (và đối chiếu với behavior của model):** Hypothesis: model gắn các từ tiêu cực của cốt truyện (detention, violated, don't) với nhãn negative vì không phân biệt được nội dung được kể và đánh giá về phim; thiếu positional/ngữ cảnh nên không hiểu 'didn't/don't' đang phủ định điều gì. Xác suất 0.381 gần 0.5 nên model không chắc chắn, một phần lỗi có thể do ranh giới quyết định chứ không chỉ do từ cụ thể.

## Mẫu #21895 — SAI
- **True / Pred:** positive / negative (P(pos) = 0.219, 181 tokens)
- **Top-5 attended:** anti(0.01), terrible(0.01), nazi(0.01), any(0.01), nemesis(0.01)
- **Text (300 ký tự đầu):** It's one of the paradoxes of Basil Rathbone's wartime anti-Nazi Sherlock Holmes films (Voice of Terror, SH in Washington, and this one) that while the plots and settings are mostly terrible, he is so good in them. Despite a bizarre wind-swept hairstyle meant to make him look younger, he blazes throu
- **Loại lỗi / lý do đúng:** Sai (positive → negative, P(pos)=0.219). Review khen diễn xuất ('he is so good in them') nhưng chê cốt truyện ('mostly terrible'), nên cảm xúc hỗn hợp. Loại: ambiguous wording / contrast ('while ... terrible, he is so good') và domain/rare words ('anti-Nazi', 'Sherlock Holmes', 'nemesis').
- **Token con người cho là quan trọng:** so good, blazes through (khen diễn xuất); 'terrible' chỉ về cốt truyện
- **Attention có tập trung vào các token đó không?** Chỉ nhìn thấy 'terrible' trong top-5 (cùng anti, nazi, any, nemesis); 'good' không có. Nhưng tất cả đều 0.01 nên việc 'terrible' có mặt không chứng minh nó dẫn đến dự đoán.
- **Hypothesis giải thích prediction (và đối chiếu với behavior của model):** Hypothesis: 'terrible', 'bizarre', 'anti', 'nazi' tạo tín hiệu tiêu cực cộng dồn, át từ khen 'so good' xuất hiện sau liên từ 'while'; model không mô hình hóa đối lập (cấu trúc 'while X, Y') vì không có thông tin thứ tự. Hành vi này nhất quán với Application (câu 'bad, but ... excellent' cũng ra negative), nhưng vẫn là giả thuyết.
