# Error analysis — LAB 08 (DistilBERT fine-tuned, IMDb test subset 5,000)

> Mẫu được chọn bởi `fine_tuning.ipynb` (mục 9) và lưu đầy đủ ở `error_samples.csv` (cột `test_idx` là chỉ số trong IMDb test chính thức 25,000).
> Cách chọn (cố định, không chọn theo ý thích): 3 FP + 3 FN lấy ngẫu nhiên (`random_state=42`) từ các lỗi **trừ** hai lỗi tự tin nhất; 2 lỗi tự tin nhất (1 FP, 1 FN).
> Model: `ft_len128_100` (max_len=128, truncation giữ phần đầu). Tổng 648 lỗi / 5,000 (FP=319, FN=329).
> **Tất cả "Hypothesis" dưới đây là giả thuyết đọc từ văn bản, chưa được kiểm chứng bằng thí nghiệm ablation** (ví dụ chưa chạy lại model trên bản review đã sửa/đầy đủ). Model không được lưu nên không thể chạy lại thêm để kiểm tra.

## Bảng tóm tắt

| # (test_idx) | Text (rút gọn) | True | Pred | Confidence | Error type | Hypothesis |
|---|---|---|---|---|---|---|
| 2766 (FP) | "This movie is a great movie ONLY if you need something to ... laugh at the stupidity of it ... The director does a good job of adding more comic relief" | neg | pos | 0.750 (P(pos)=0.750) | sarcasm / ambiguous sentiment | Từ khen (*great, interesting, funny*) đứng sớm trong review; ý chê nằm ở nghĩa mỉa mai ("ONLY if...", "stupidity", "facts wrong") nên model dựa vào từ khóa (các câu khen *entertaining, good job* ở cuối có thể đã bị cắt vì review dài 189 token). |
| 10459 (FP) | "This sickly sweet and laboriously paced 5-reeler is definitely not among Harold Lloyd's better films ... Ugh. 4/10." | neg | pos | 0.711 | negation + domain-specific expression + mixed cues | "not among ... better films" (phủ định) và "sparse/uninspired" bị lấn át bởi từ vựng tích cực ở phần tóm tắt cốt truyện ("kindly", "sunshine", "bringing a little sunshine into the lives of children"); từ chuyên ngành ("5-reeler", "gags") và điểm số "4/10" khó khai thác. |
| 1219 (FP) | "I loved this movie 10 years ago ... My biggest mistake was to watch it again ... it has so many flaws ... makes this film better than the average" | neg | pos | 0.628 | context (đổi ý theo thời gian) + negation + mixed sentiment | Câu mở đầu "I loved" tích cực nhưng về sau mới phủ nhận ("biggest mistake", "so many flaws"); lại có nhượng bộ tích cực ("better than average", "not the worst"). Confidence thấp (0.628) phản ánh đúng sự mơ hồ. |
| 21540 (FN) | "The new voices scare me! ... It's the same thing over and over ... one of the most retarded shows ever! The first movie was so damn better! ..." | pos | neg | 0.763 (P(pos)=0.237) | label ambiguity / likely label noise | Review này đang chê series TV (so với phim gốc); nhãn "positive" có vẻ không khớp nội dung, model đoán negative là hợp lý theo văn bản. Chỉ có thể nói "có vẻ"; không có nhãn thứ hai để xác nhận. |
| 16711 (FN) | "I don't see that much wrong with this movie ... Some people claim ... ghastly ... plastic; I disagree ... may not be everyone's cup of tea, ... I give it ten points" | pos | neg | 0.751 (P(pos)=0.249) | negation + long-range dependency (truncation) | Review dài 283 token, bị cắt ở 128; phần còn lại trong 128 token chứa nhiều từ tiêu cực được nhắc đến để phản bác ("wrong", "ghastly", "plastic", "not ... Pavarotti"); câu chốt "give it ten points out of ten" nằm sau vị trí cắt nên model không thấy. |
| 22033 (FN) | "...apparently Bernard Cribbins ad libbed nearly all of his lines. If you can sit through the 'Daddy! Oh my daddy' bit without blubbing then you really need to get in touch with your inner child" | pos | neg | 0.769 (P(pos)=0.231) | implicit sentiment + negation ("without blubbing") + thiếu ngữ cảnh | Đây là đoạn trích rất ngắn (65 token), bắt đầu bằng "..." nên thiếu ngữ cảnh; khen gián tiếp qua cảm xúc ("blubbing", "inner child") chứ không có từ khen trực tiếp; các cụm "really need to get in touch with" nghe như phê bình khi tách rời ngữ cảnh. |
| 4118 (FP, rất tự tin) | "This is definitely one of the best Kung fu movies in the history of Cinema ... a masterpiece!" | neg | pos | **0.986** | label ambiguity / likely label noise | Văn bản hoàn toàn tích cực, không có phủ định hay mỉa mai nào đáng kể ("which is not often the case" phủ định một nhận xét phụ). Model dự đoán hợp lý theo nội dung; nhiều khả năng nhãn gốc sai/nhiễu. |
| 23120 (FN, rất tự tin) | "There's not much anyone can say about this flick ... the writing was just horrible ... Women: B ... Sex: B+ ... Story: C ... Overall: B-" | pos | neg | **0.986** (P(pos)=0.014) | ambiguous sentiment + domain-specific expression + long-range dependency (truncation) | Phần đầu (trong 128 token) toàn chê ("not much anyone can say", "writing was just horrible"), còn các điểm số kiểu chữ cái và kết luận "Overall: B-" nằm cuối review (382 token) nên bị cắt; bản thân nhãn "positive" cho một review pha trộn có điểm B- cũng khá tùy tiện. |

## Chi tiết từng mẫu

### FP #2766 — true negative, pred positive (P(pos)=0.750, 189 token)
- **Phân loại:** sarcasm / ambiguous sentiment.
- **Quan sát:** review dùng từ khen ("great movie", "interesting", "funny", và ở cuối "very entertaining", "does a good job") nhưng điều kiện đi kèm ("ONLY if you need something to ... laugh at the stupidity of it") và nội dung ("gets most of the important facts wrong") cho thấy ý chê. Đây là kiểu "so bad it's funny".
- **Hypothesis:** model nhạy với từ vựng cảm xúc bề mặt nên bị kéo về positive; mỉa mai cần suy luận về ý định người viết vượt ra ngoài từ khóa. Confidence chỉ 0.75, tức model không chắc chắn.
- **Kiểm chứng đề xuất (chưa làm):** thay *great/funny/entertaining* bằng từ trung tính và xem P(pos) giảm bao nhiêu.

### FP #10459 — true negative, pred positive (P(pos)=0.711, 102 token, không bị cắt)
- **Phân loại:** negation + domain-specific expression + mixed cues.
- **Quan sát:** ý chê rõ ở đầu ("definitely not among ... better films", "Gags are sparse and mostly uninspired", "Ugh. 4/10"), nhưng phần giữa tả cốt truyện bằng từ vựng ngọt ngào ("kindly", "sunshine", "children ... puppies") và có "sickly sweet", "Saccharine" mang nghĩa chê nhưng là từ hiếm với nghĩa ẩn dụ.
- **Hypothesis:** từ chê ẩn dụ (*saccharine*, *sickly sweet*) hiếm trong dữ liệu fine-tune 10,000 mẫu; từ tích cực mô tả cốt truyện (không phải đánh giá) đánh lừa model. Điểm số "4/10" là tín hiệu rõ nhưng chữ số ít khi được dùng làm tín hiệu sentiment bởi model.

### FP #1219 — true negative, pred positive (P(pos)=0.628, 109 token)
- **Phân loại:** context (đổi ý theo thời gian) + negation + mixed sentiment.
- **Quan sát:** "I loved this movie 10 years ago" → "My biggest mistake was to watch it again" → "not the worst ... but it has so many flaws" → "makes this film better than the average patriotic nonsense". Có nhượng bộ tích cực và điểm cuối `[****------]` (4/10 sao).
- **Hypothesis:** các mệnh đề tích cực đầu và cuối cạnh tranh với phủ định ở giữa; model phải theo dõi mối quan hệ "trước kia thích / giờ thì không" (phụ thuộc ngữ cảnh). Confidence thấp (0.628) cho thấy model phát hiện sự mơ hồ; đây là mẫu "khó cho cả người đọc".

### FN #21540 — true positive, pred negative (P(pos)=0.237, 202 token)
- **Phân loại:** label ambiguity (có vẻ là nhãn nhiễu).
- **Quan sát:** toàn bộ nội dung phàn nàn về series ("same thing over and over", "most retarded shows ever", "The first movie was so damn better!"). Người viết là fan của phim gốc; nhãn gốc là positive.
- **Hypothesis:** nhãn không phản ánh nội dung văn bản (hoặc nhãn gán theo phim gốc), nên model "sai" theo nhãn nhưng đúng theo ngữ nghĩa văn bản. Mình chưa thể xác nhận bằng nguồn thứ hai.

### FN #16711 — true positive, pred negative (P(pos)=0.249, 283 token)
- **Phân loại:** negation + long-range dependency (truncation).
- **Quan sát:** nhiều câu phản bác ý kiến tiêu cực ("I don't see that much wrong", "Some people claim ... ghastly ... plastic; I disagree on both counts"), kết luận tích cực ("redeeming value", "I give it ten points out of ten") nằm ở cuối.
- **Hypothesis:** với max_len=128, model chỉ thấy khoảng nửa đầu: toàn từ phủ định/chê được trích dẫn để phản bác; phán quyết cuối bị cắt. Dự đoán negative với 0.75 hợp lý trên nội dung nhìn thấy. Nếu đúng, tăng max_len lên 256–512 sẽ sửa được mẫu này (Exp 4: T=256 tăng +3.9 điểm accuracy tổng thể so với T=128, nhất quán nhưng chưa kiểm tra trực tiếp mẫu này).

### FN #22033 — true positive, pred negative (P(pos)=0.231, 65 token)
- **Phân loại:** implicit sentiment + negation ("without blubbing") + context bị thiếu.
- **Quan sát:** đây là đoạn trích ngắn (bắt đầu bằng "..."): khen bằng cách mô tả cảm xúc của người xem ("Daddy! Oh my daddy", "blubbing", "inner child"); không có từ khen trực tiếp.
- **Hypothesis:** không đủ tín hiệu từ vựng tích cực rõ ràng; "without blubbing" và "you really need to ..." đọc như điều kiện/phê bình. Model cần kiến thức về văn cảnh (đây là phim gia đình gây xúc động) mà đoạn văn không cung cấp.

### FP #4118 — true negative, pred positive, **very confident** (0.986, 73 token)
- **Phân loại:** label ambiguity / nhãn nhiều khả năng sai.
- **Quan sát:** "definitely one of the best Kung fu movies in the history of Cinema ... a great actor ... a masterpiece!" — không có mỉa mai hay phủ định có nghĩa.
- **Hypothesis:** model dự đoán đúng theo nội dung; nhãn "negative" là nhiễu của dataset (hoặc người đánh giá gán nhầm). Đây là ví dụ rằng *lỗi có confidence cao không nhất thiết là lỗi của model*; nó có thể là lỗi của nhãn. Mình không có nguồn độc lập để khẳng định nhãn sai.

### FN #23120 — true positive, pred negative, **very confident** (P(pos)=0.014, 382 token)
- **Phân loại:** ambiguous sentiment + domain-specific expression (điểm chữ cái) + long-range dependency (truncation).
- **Quan sát:** review pha trộn: chê kịch bản ("writing was just horrible", "recycled plot") nhưng khen cảnh quay ("excellent work there", "wasn't bad either") và chốt bằng điểm "Overall: B-". Review dài 382 token, bị cắt ở 128.
- **Hypothesis:** phần model nhìn thấy (đầu review) nghiêng hẳn về chê; điểm tổng kết kiểu chữ cái ở cuối bị cắt và (kể cả khi thấy) là biểu thức đặc thù domain mà model khó diễn giải. Nhãn positive cho review B- cũng mơ hồ. Cho nên confidence 0.986 là "tự tin trên phần nhìn thấy", không phải trên toàn review.

## Nhận xét chung về lỗi (observation → evidence → interpretation)
- *Observation:* 8 mẫu rơi vào các nhóm: mỉa mai/từ khen đặt trong ngữ cảnh chê (2766, 10459), phủ định + nhượng bộ (10459, 1219, 16711, 22033), truncation / long-range (16711, 23120), label ambiguity hoặc nhãn nhiễu (21540, 4118, 23120).
- *Evidence:* accuracy theo độ dài review (model T=128): ≤128 token 0.912 (n=614), 128–256 0.894 (n=2,354), 256–512 0.847 (n=1,383), >512 0.795 (n=649). Hai lỗi tự tin nhất đều mang dấu hiệu nhãn mơ hồ/nhiễu. 3/8 mẫu có nguyên nhân truncation đáng ngờ.
- *Interpretation:* không phải mọi lỗi là "model không hiểu": một phần là giới hạn do thiết kế (cắt 128 token mất phần kết luận), một phần do nhãn/mẫu mơ hồ; phần còn lại (mỉa mai, phủ định phức tạp) là giới hạn thật của model (rất nhiều khả năng tồn tại cả ở model lớn hơn, nhưng chưa thử). 8 mẫu này chọn theo quy tắc cố định, không đại diện thống kê cho 648 lỗi.

## Trả lời câu hỏi bắt buộc (đề mục 28)

> Mỗi câu tách *observation → evidence → interpretation*. Các số lấy từ `results.csv`, `domain_shift.csv`, `fine_tuning.ipynb`. Tất cả 1 seed, test subset 5,000 review (sai số chuẩn ≈ 0.5 điểm).

**Q1. Model nào hưởng lợi nhiều hơn khi training data ít?**
- *Observation:* khi giảm 100% → 10% dữ liệu, accuracy của model pretrained + fine-tuning giảm 87.0 → 79.7 (−7.4), frozen 79.2 → 73.3 (−5.9), small-scratch 74.8 → 59.5 (−15.4).
- *Evidence:* ở 10% dữ liệu pretrained cao hơn scratch 20.2 điểm (0.797 so với 0.595), trong khi ở 100% là 12.2 điểm. Control: cho model fine-tune 10% dữ liệu thêm bước huấn luyện (10 epoch) lên 81.8 → một phần mất mát ở 10% là do ít bước cập nhật.
- *Interpretation:* pretrained (cả fine-tune và frozen) hưởng lợi nhiều hơn scratch khi ít dữ liệu: pretrained đã học biểu diễn, scratch phải học cả biểu diễn lẫn task từ 1,000 review. Lưu ý frozen giảm ít nhất nhưng vì nó có ít khả năng học ngay từ đầu (1,538 tham số), không phải vì "hưởng lợi" nhiều hơn.

**Q2. Nếu pretrained model tốt hơn from-scratch, điều đó chứng minh điều gì?**
- *Observation/Evidence:* 0.870 (pretrained) vs 0.800 (cùng kiến trúc random init, cùng hyper-parameter chính) vs 0.748 (small scratch).
- *Interpretation:* nó chứng minh (trong điều kiện thí nghiệm này) rằng **trọng số pretrained mang thông tin hữu ích cho task** và giúp học nhanh hơn với dữ liệu nhỏ; cặp random-init/pretrained cùng kiến trúc loại trừ việc do kiến trúc. Nó **không** chứng minh model "hiểu ngôn ngữ", không chứng minh model from-scratch không thể bắt kịp nếu có nhiều dữ liệu/epoch/tuning hơn (lr chỉ thử 1 giá trị, 1 seed), và không chứng minh bền vững khi đổi domain (xem Q5).

**Q3. Nếu frozen model gần bằng fine-tuned model, điều đó có ý nghĩa gì?**
- *Trong lab này frozen không gần:* 0.792 so với 0.870 (kém 7.9 điểm); khoảng cách khá ổn định theo lượng dữ liệu (6.4 ở 10%, 7.9 ở 25%, 8.1 ở 50%, 7.9 ở 100%).
- *Interpretation (giả định):* nếu frozen gần bằng fine-tuned thì nghĩa là biểu diễn pretrained đã chứa gần đủ thông tin của task và tách tuyến tính được; head tuyến tính đủ dùng, ta tiết kiệm chi phí huấn luyện/bộ nhớ (frozen 0.44 GB so với 1.35 GB) và chỉ cần lưu một encoder dùng chung cho nhiều task; nhưng khi đó fine-tuning ít có lợi. Cũng cần kiểm tra xem task có "dễ" không (nếu ngay cả baseline đơn giản cũng gần bằng).

**Q4. Nếu fine-tuning tốt hơn frozen rất nhiều, điều đó gợi ý điều gì?**
- *Observation:* trong lab: +7.9 điểm accuracy, F1 +8.1.
- *Interpretation:* thông tin có ích cho task có trong biểu diễn nhưng **chưa được sắp xếp sao cho `[CLS]` + một siêu phẳng tuyến tính khai thác được**; mục tiêu pretraining (MLM/distillation) khác mục tiêu sentiment nên encoder cần được điều chỉnh (đặc biệt `[CLS]` chưa được huấn luyện cho classification, và phong cách review phim cần thích nghi). Cái giá: nhiều tham số/bộ nhớ hơn (66.4M so với 1,538 trainable) và phải lưu một bản model cho mỗi task. Mức "rất nhiều" ở đây là ~8 điểm; một vài biến thể rẻ hơn (pooling mean thay vì `[CLS]`, head MLP, fine-tune một số layer cuối) chưa được thử nên chưa kết luận rằng toàn bộ 8 điểm cần đến fine-tuning toàn mô hình.

**Q5. Nếu performance giảm mạnh trên domain khác, nguyên nhân có thể là gì?**
- *Observation:* IMDb → Rotten Tomatoes: 0.870 → 0.791 (fine-tuned), 0.748 → 0.610 (small scratch); trên Amazon không giảm (0.914). Trên RT model dự đoán positive 58.7% (nhãn 50%), precision 0.748, recall 0.878; ECE tăng 0.019 → 0.042.
- *Interpretation (nguyên nhân khả dĩ, chưa tách riêng từng yếu tố):* (i) lệch phân phối văn bản: câu rất ngắn (median 27 vs 220 token) và giọng văn của nhà phê bình; (ii) lệch phân phối nhãn/ranh giới nhãn: model đã học ranh giới "positive" từ review IMDb dài và kéo sang positive trên câu RT; (iii) từ vựng và tương quan đặc thù domain (tên phim, từ chuyên ngành) mà head/encoder đã fine-tune theo IMDb; (iv) cách gán nhãn khác nhau giữa các dataset; (v) khác biệt về độ khó/độ dài (làm so sánh accuracy tuyệt đối giữa các tập không công bằng — ví dụ Amazon cao hơn in-domain vì ngắn và tín hiệu rõ). Để tách các nguyên nhân cần baseline in-domain train trên chính từng tập — chưa làm.

**Q6. Nếu sequence length tăng nhưng performance gần như không đổi trong khi training time tăng mạnh, có nên tiếp tục tăng không?**
- *Câu trả lời:* không; nếu đường performance đã phẳng thì phần tăng thêm là chi phí vô ích (thời gian, bộ nhớ `O(T²)` ở attention, padding) — chọn T nhỏ nhất ở đầu vùng phẳng, hoặc chuyển sang chiến lược khác (cắt đầu + cuối, chia đoạn rồi tổng hợp).
- *Trong lab này giả định không thỏa:* performance **không** phẳng (0.793 / 0.858 / 0.897 cho T = 64 / 128 / 256) và chi phí đo được tăng khá chậm (benchmark: 1.0 / 1.09 / 1.40× thời gian/step cho T = 64/128/256 ở batch nhỏ) nên tăng đến 256 là hợp lý; chưa có số liệu cho T=512 về độ chính xác để biết điểm bão hòa. Quyết định phải dựa trên đường cong performance-so-với-chi-phí của bài toán cụ thể: Amazon (median 94 token) và RT (27 token) sẽ không hưởng lợi từ T lớn.
