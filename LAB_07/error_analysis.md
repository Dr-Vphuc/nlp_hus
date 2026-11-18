# Error analysis — LAB 07 (Transformer 1 block, 4 heads, mean pooling, `max_length=200`)

> Dữ liệu (text, nhãn, xác suất, token nhận nhiều attention nhất) do notebook sinh ra (`error_samples.csv`, mục 9 của notebook); 3 mẫu đúng + 3 mẫu sai được chọn ngẫu nhiên (seed 42) từ test set 25,000 review.
> "Top-5 attended" = token nhận nhiều attention nhất, tính **trung bình theo 4 head và theo mọi query** (attention nhận được của mỗi token) trong block duy nhất của model. Giá trị uniform là `1/T` (≈0.005 với T=200), nên 0.03–0.06 là gấp 6–12 lần uniform.
> **Quan trọng:** đây là mô tả *phân phối attention weights*, KHÔNG phải lời giải thích của model. Phần "Giả thuyết" luôn được tách riêng khỏi quan sát; chưa có giả thuyết nào ở đây được kiểm chứng bằng ablation.
> Lưu ý chung: model chỉ thấy 200 token đầu (`head` truncation) và dùng mean pooling.

Kết quả test: Transformer accuracy 0.8394, F1 0.8419 (seed 42). Confusion matrix: `confusion_matrix.png`.

---

## Mẫu #16510 — ĐÚNG
- **True / Pred:** positive / positive (P(pos) = 0.849, 308 token, **108 token cuối bị cắt**)
- **Top-5 attended:** builds(0.064), town(0.038), nose(0.023), town(0.023), attempt(0.020)
- **Text (đầu):** "As with so many modern US films, there has to be a supernatural element to the plot, but if you just let that go, this is a tale with heaps of charm and a kindly heart cased in a crab shell..."
- **Loại:** không phải lỗi; review nặng tóm tắt cốt truyện (plot summary), có `contrast` ("..., but if you just let that go, ...") và **truncation** (phần cuối chứa "beautiful brooding mood", "excelled", "i like signs of life for its simpler less contrived story" bị cắt).
- **Token con người cho là quan trọng:** charm, kindly, heart; (trong phần bị cắt: beautiful, excelled, i like).
- **Attention có tập trung vào các token đó không?** Không. Top-5 toàn từ của cốt truyện (builds, town, nose, attempt); không có từ cảm xúc nào. Attention không phân tán đều (0.064 ≈ 12×uniform) nhưng cũng không nằm trên từ cảm xúc.
- **Giả thuyết:** dự đoán đúng có thể nhờ các từ tích cực rải rác trong 200 token đầu ("charm", "kindly", "happy"), được tổng hợp qua mean pooling; còn việc attention dồn vào "builds/town" không cho thấy nó là lý do. Với P = 0.849 chứ không phải 0.99, việc mất phần kết (nơi tác giả nói rõ "I like...") có thể làm model kém chắc chắn hơn, nhưng đây là suy đoán (chưa thử bỏ/giữ phần cuối).

## Mẫu #2285 — ĐÚNG
- **True / Pred:** negative / negative (P(pos) = 0.206, 632 token, **432 token cuối bị cắt**)
- **Top-5 attended:** nothing(0.046), cut(0.046), stupid(0.036), remember(0.031), mediocre(0.018)
- **Text (đầu):** "The Net is a movie I never saw upon release, I remember giving it a pass upon the mediocre reviews... they look stupid and hey it even stars sandra bullock... the net is just an unimaginative plain and totally routine hitchockian cat and mouse thriller nothing archaic about that..."
- **Loại:** không phải lỗi, nhưng khó: mở đầu kể lể/hoài niệm ("retro heaven", "the best part of this movie experience"), **sarcasm/ambiguous** ("that was the best part of this movie experience for me" thật ra là chê), **long sequence** (632 token) và **truncation**.
- **Token con người cho là quan trọng:** mediocre, stupid, unimaginative, plain, routine, "nothing archaic".
- **Attention có tập trung vào các token đó không?** Một phần: `stupid`, `mediocre`, `nothing` nằm trong top-5; `cut` (trong "cut down on fat") và `remember` không liên quan đến sentiment; `unimaginative`, `routine` không nằm trong top-5.
- **Giả thuyết:** các từ tiêu cực rõ ràng (mediocre, stupid, unimaginative) đủ để kéo P(pos) xuống 0.206, mặc dù mức tự tin thấp hơn nhiều so với một review chê rõ ràng vì đoạn đầu có giọng hoài niệm/mỉa mai. Việc `stupid`/`mediocre` vừa nhận attention vừa là từ tiêu cực *phù hợp* với giả thuyết này nhưng không chứng minh attention là nguyên nhân (cùng review `cut` cũng được attend mà vô nghĩa với sentiment).

## Mẫu #19447 — ĐÚNG
- **True / Pred:** positive / positive (P(pos) = 0.711, 573 token, **373 token cuối bị cắt**)
- **Top-5 attended:** great(0.030), no(0.030), transport(0.026), through(0.023), great(0.023)
- **Text (đầu):** "I must say that I didn't expect much sitting down to watch 'Pitch Black,' but I got a lot back... This isn't a deep film by any means. Everything that it offers is either recycled or ripped off... but when I watch a rip off I want it to be good and this rip off is great..."
- **Loại:** không phải lỗi, nhưng có **negation** ("didn't expect much", "isn't a deep film") và **contrast** ("but I got a lot back", "...this rip off is great"), nghĩa là bài kiểm tra tốt cho thứ tự từ.
- **Token con người cho là quan trọng:** great (×2), pure fun, excitement, "got a lot back", entertained.
- **Attention có tập trung vào các token đó không?** Có một phần: `great` xuất hiện 2 lần trong top-5; `no`, `transport`, `through` không liên quan sentiment (nằm trong cốt truyện).
- **Giả thuyết:** P chỉ 0.711 dù review khen rõ ràng; có thể các cụm phủ định/chê ("recycled", "ripped off", "isn't a deep film", "nasty") kéo xác suất xuống. Model không cho thấy bằng chứng nó xử lý "didn't expect much ... but" như một cấu trúc; chỉ biết là tổng thể vẫn nghiêng positive. Attention vào "great" **phù hợp** với dự đoán nhưng không đủ để nói "model đúng vì nhìn vào great".

---

## Mẫu #1817 — SAI
- **True / Pred:** negative / positive (P(pos) = 0.733, 144 token, không bị cắt)
- **Top-5 attended:** jungle(0.043), cop(0.043), era(0.037), bizarre(0.027), otherwise(0.026)
- **Text:** "Yes, this bizarre feature was written by John Sayles... you have a recipe for some rather unexciting action sequences. Davi is the standout amongst the cast and cinematographer Rene Verzier does a pretty good job otherwise, this is a rather lumpen action pic that won't satisfy action fans and will leaves Sayles admirers slack-jawed."
- **Loại:** **contrast / ambiguous sentiment** (khen diễn viên và quay phim: "standout", "pretty good job"; chê toàn bộ phim: "unexciting", "lumpen", "won't satisfy"), **rare word** (`lumpen` chỉ xuất hiện 1 lần trong train ⇒ `<unk>`), tên riêng (Sayles, Davi, Verzier), và **domain-specific** (review chuyên môn về đạo diễn/cast).
- **Token con người cho là quan trọng:** unexciting, lumpen, won't satisfy, slack-jawed (chê); standout, good (khen, chỉ về một phần).
- **Attention có tập trung vào các token đó không?** Không: top-5 là jungle, cop, era, bizarre, otherwise (từ trung tính của cốt truyện, trừ `bizarre`). `unexciting` và các cụm chê ở cuối câu không nằm trong top-5.
- **Giả thuyết:** hai từ khen ("standout", "pretty good") là các từ cảm xúc rõ nhất mà model biết, trong khi từ chê mạnh nhất (`lumpen`) là OOV và `won't satisfy` là phủ định (model có thể không hiểu "won't + satisfy"). Do mean pooling tính mọi token như nhau nên một vài từ khen có thể thắng. Giả thuyết cần thử thay `lumpen` bằng từ đã biết hoặc bỏ câu khen để kiểm tra.

## Mẫu #16701 — SAI
- **True / Pred:** positive / negative (P(pos) = 0.363, 270 token, **70 token cuối bị cắt**)
- **Top-5 attended:** noir(0.043), great(0.041), lacks(0.035), tries(0.031), tries(0.027)
- **Text:** "This interesting film noir features three very good performances... Demming, as the detective, is unfortunately not nearly as good. He lacks the intelligence, strength, and cynical world view of a Bogart... there are a few plot holes... What bothered me the most, however, was that the chance for a great and unexpected conclusion was wasted..."
- **Loại:** **contrast / mixed sentiment** (nhãn positive nhưng nội dung nhiều chỗ chê: "unfortunately", "lacks", "plot holes", "bothered me the most", "wasted"), **negation** ("not nearly as good"), **truncation** (phần cuối còn chê thêm), và có thể là **label noise / nhãn đơn giản hóa** (review 3–4 sao bị gán positive).
- **Token con người cho là quan trọng:** very good, outstanding (khen); unfortunately, lacks, plot holes, wasted (chê). Một người đọc cũng khó kết luận chắc chắn.
- **Attention có tập trung vào các token đó không?** Có một phần: `lacks` (chê) và `great` (trong "a great and unexpected conclusion was wasted") nằm trong top-5, cùng `noir`, `tries` ×2. Chú ý `great` được attend nhưng nằm trong ngữ cảnh phủ định ("was wasted"), model nhìn thấy `great` nhưng dự đoán vẫn là negative.
- **Giả thuyết:** dự đoán negative là hợp lý theo nội dung (nhiều ý chê), nên có thể xem đây là lỗi "model đúng theo văn bản, nhãn mơ hồ" hơn là lỗi thuần túy của model. Đây là suy đoán: P = 0.363 gần ngưỡng 0.5 nên model không chắc chắn.

## Mẫu #21110 — SAI
- **True / Pred:** positive / negative (P(pos) = 0.293, 405 token, **205 token cuối bị cắt**)
- **Top-5 attended:** obvious(0.044), sinister(0.031), reason(0.019), cannot(0.017), weather(0.016)
- **Text:** "I believe there are two angles to the story, first, it's the world satyric view, presented in an obvious Kafkaesque manner, underlining the sordid spectacle behind some of life's most common... second, the most interesting aspect of this film I believe is the fact that it attacks head on this natural compulsion to dissect the world..."
- **Loại:** **domain-specific / abstract vocabulary** (`kafkaesque`, `satyric` là OOV), không có từ khen trực tiếp (chỉ "most interesting aspect"), từ có sắc thái tiêu cực nhưng không phải đánh giá phim ("sordid", "sinister", "condemning", "uncomfortable", "cannot"), **noise** (viết sai `weather` thay `whether`), và **truncation** (205 token cuối bị cắt, nơi có thể chứa kết luận).
- **Token con người cho là quan trọng:** "most interesting aspect", "interesting" (hiếm hoi mang sắc thái khen); phần còn lại là phân tích triết lý của người viết.
- **Attention có tập trung vào các token đó không?** Không: `interesting` không nằm trong top-5. Top-5 gồm `sinister`, `cannot` (từ tiêu cực về nghĩa, nhưng nói về chủ đề của phim chứ không phải đánh giá phim) cùng `obvious`, `reason`, `weather`.
- **Giả thuyết:** model bị đánh lừa bởi các từ tiêu cực về chủ đề (sordid, sinister, cannot) vì chúng thường đi cùng review tiêu cực trong dữ liệu train, trong khi ý kiến thật của người viết ("interesting") là tín hiệu yếu và ẩn. Trường hợp này cho thấy lỗi do **khác biệt giữa sentiment của nội dung phim và sentiment của người đánh giá**; cần thử bỏ riêng các từ này để kiểm chứng.

---

## Tổng hợp
| Mẫu | Kết quả | Loại chính | Token quan trọng (người) có trong top-5 attention? |
|---|---|---|---|
| 16510 | đúng | plot summary, truncation | không |
| 2285 | đúng | sarcasm, long, truncation | một phần (stupid, mediocre) |
| 19447 | đúng | negation, contrast | một phần (great) |
| 1817 | sai | contrast, rare word (`lumpen`), domain | không |
| 16701 | sai | contrast/mixed, negation, truncation, nhãn mơ hồ | một phần (lacks, great) |
| 21110 | sai | domain-specific, noise, truncation | không |

- **Quan sát:** cả 3 mẫu sai đều có sentiment hỗn hợp hoặc được diễn đạt gián tiếp, 5/6 mẫu bị cắt ở 200 token (chỉ #1817, 144 token, không bị cắt), và trong cả 6 mẫu, token được attend nhiều nhất nhiều khi là từ cốt truyện (builds, jungle, noir, transport) thay vì từ cảm xúc.
- **Attention ≠ explanation:** (1) có mẫu mà từ cảm xúc xuất hiện trong top-5 nhưng dự đoán sai (#16701 có `lacks`, `great`), (2) có mẫu mà top-5 không có từ cảm xúc nào mà vẫn đúng (#16510). Attention ở đây là trung bình của một block, một phần của đường tính toán; sau đó còn residual, FFN, LayerNorm, mean pooling và lớp Linear, nên attention weights chỉ mô tả *model đã trộn thông tin từ token nào*, không nói *vì sao* ra nhãn đó. Để kiểm chứng cần thí nghiệm can thiệp (xóa/thay token và xem xác suất thay đổi), vốn chưa làm trong lab này.
- **Cỡ mẫu:** chỉ 6 mẫu nên không kết luận thống kê.
