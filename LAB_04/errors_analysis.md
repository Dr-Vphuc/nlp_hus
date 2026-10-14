# Error Analysis — LAB 04 (Đề §21)

**Model:** A — TF-IDF (unigram) + Logistic Regression · **Dataset:** IMDb test set
**Cách chọn mẫu:** ngẫu nhiên với seed = 42 (notebook mục 10)

## Bảng tổng hợp

| # | Text (rút gọn) | True | Predicted | P(pos) | Analysis |
|---|---|---|---|---|---|
| 16321 | great movie. post-apocalyptic films kick ass. this one is no exception... | positive | positive | 0.970 | Đúng nhờ great, appreciated, liked; nhưng feature mạnh nhất là "apocalyptic" (từ chỉ thể loại) |
| 2195 | it only took one viewing of this dog, for me to say "never again!"... | negative | negative | 0.098 | Đúng nhờ worst, awful; "bond", "best" (trong câu so sánh) kéo ngược |
| 19308 | this program is a favorite of our family, and we feel it must be released on dvd... | positive | positive | 0.995 | Đúng, tín hiệu tích cực áp đảo; "bad" trong "without worrying about bad language" bị hiểu sai |
| 2346 | on paper this movie has some chops: ... great stuff, everyone loves a happy ending... | negative | positive | 0.508 | Câu nhượng bộ "on paper ... though" + "great" trong tên "great expectations" |
| 17654 | the only reason i decided to view this film is because of the great acting talents of karen sillas... | positive | negative | 0.445 | Tên phim "bad money", "'99" + đoạn tóm tắt plot ("money", "lies") |
| 21483 | this film was original in an unoriginal way... | positive | negative | 0.150 | Negation "nothing less than enjoyable", "6.4" bị tách thành "4" |

## A. 3 mẫu phân loại đúng

### Mẫu 1 — idx = 16321
- **True / Predicted / P(pos):** positive / positive / 0.970
- **Từ đẩy về positive:** apocalyptic (+0.623), and (+0.470), great (+0.470), appreciated (+0.449), liked (+0.392), tight (+0.348)
- **Từ đẩy về negative:** interest (-0.300), development (-0.268), exception (-0.234), no (-0.212), dialogue (-0.208)

> ✍️ **Tự trả lời:** Model đoán đúng nhờ những từ nào? Các từ đó có thật sự mang sentiment không, hay model đúng nhờ từ không liên quan (tên riêng, số, từ chung chung)?
>
> Đúng nhờ các từ mang sentiment thật như "great", "appreciated", "liked", "tight". Tuy nhiên feature đóng góp lớn nhất lại là "apocalyptic" (từ chỉ thể loại phim), không mang sentiment, tức là model học cả bias chủ đề trong train. Ngược lại cụm "no exception" (ý khen) bị tách thành "no" và "exception" nên lại đẩy về negative.

### Mẫu 2 — idx = 2195
- **True / Predicted / P(pos):** negative / negative / 0.098
- **Từ đẩy về positive:** bond (+0.846), best (+0.363), entertaining (+0.327), bit (+0.323), personally (+0.266)
- **Từ đẩy về negative:** worst (-0.961), awful (-0.830), why (-0.348), even (-0.335), 3 (-0.302), 1 (-0.235), 2 (-0.216)

> ✍️ **Tự trả lời:** Model đoán đúng nhờ những từ nào? Các từ đó có thật sự mang sentiment không, hay model đúng nhờ từ không liên quan (tên riêng, số, từ chung chung)?
>
> Đúng nhờ "worst" và "awful", 2 từ tiêu cực rất mạnh. Phần còn lại khá nhiễu: "bond" (tên riêng) là feature đẩy về positive mạnh nhất, "best" và "entertaining" thật ra nằm trong câu so sánh để chê ("far more entertaining than this"), còn "1", "2", "3" chỉ là số thứ tự liệt kê. Model đúng vì "worst"/"awful" đủ mạnh để lấn át.

### Mẫu 3 — idx = 19308
- **True / Predicted / P(pos):** positive / positive / 0.995
- **Từ đẩy về positive:** favorite (+0.680), great (+0.532), dvd (+0.514), and (+0.466), themes (+0.343), beautiful (+0.293)
- **Từ đẩy về negative:** bad (-0.522), positive (-0.488), entire (-0.265), left (-0.237), promised (-0.212)

> ✍️ **Tự trả lời:** Model đoán đúng nhờ những từ nào? Các từ đó có thật sự mang sentiment không, hay model đúng nhờ từ không liên quan (tên riêng, số, từ chung chung)?
>
> Đúng nhờ "favorite", "great", "beautiful", đều là từ tích cực thật. Vẫn có nhiễu: "dvd" là từ domain-specific, "bad" nằm trong "without worrying about bad language" nhưng vẫn bị tính là tiêu cực, và ngay cả từ "positive" cũng có weight âm. Các lỗi này không đủ lật kết quả vì review tích cực quá rõ ràng.

## B. 3 mẫu phân loại sai

### Lỗi 1 — idx = 2346
- **True / Predicted / P(pos):** negative / positive / 0.508
- **Từ đẩy về positive:** great (+0.629), everyone (+0.489), the (+0.438), and (+0.433), learned (+0.284), most (+0.251)
- **Từ đẩy về negative:** annoying (-0.360), uninteresting (-0.336), nothing (-0.295), enough (-0.290), cheap (-0.233), offensive (-0.209)

> ✍️ **Tự trả lời:** Lỗi này thuộc loại nào? (lexical ambiguity / negation / sarcasm / short text / rare word / domain-specific term / data noise / class imbalance, có thể nhiều hơn một loại)
>
> Lexical ambiguity + domain-specific term (tên riêng), kèm câu nhượng bộ gần giống sarcasm.

> ✍️ **Tự trả lời:** Đoạn nào trong review là bằng chứng cho loại lỗi đó? Vì sao TF-IDF unigram không xử lý được?
>
> "on paper this movie has some chops ... great stuff, everyone loves a happy ending. in the theater though, ...", phần khen chỉ là giả định và bị "though" đảo lại ngay sau đó. Ngoài ra "great" trong "great expectations" (tên tiểu thuyết) vẫn là feature đẩy về positive mạnh nhất. Unigram xét từng từ độc lập nên không biết "on paper"/"though" làm đổi nghĩa phần trước, cũng không phân biệt được "great" trong tên riêng. Các từ chê như "hollow", "cartoonish", "dead inside" lại không nằm trong top đóng góp, bị pha loãng trong review dài → P(pos) sát 0.5.

### Lỗi 2 — idx = 17654
- **True / Predicted / P(pos):** positive / negative / 0.445
- **Từ đẩy về positive:** great (+1.321), cry (+0.347), and (+0.331), deeply (+0.242), you (+0.235)
- **Từ đẩy về negative:** talents (-0.511), money (-0.484), bad (-0.432), 99 (-0.310), only (-0.307), seems (-0.237), lies (-0.213)

> ✍️ **Tự trả lời:** Lỗi này thuộc loại nào? 
>
> Domain-specific term (tên phim, năm phát hành) + lexical ambiguity.

> ✍️ **Tự trả lời:** Đoạn nào trong review là bằng chứng cho loại lỗi đó? Vì sao TF-IDF unigram không xử lý được?
>
> `"bad money",'99` là tên phim kèm năm, nhưng "bad", "money", "99" đều bị tính là tiêu cực. Phần lớn review là tóm tắt plot ("never seems to stop telling lies", "troublesome") nên kéo về negative, trong khi đánh giá thật chỉ có "great talents", "big fan", "i miss her". "talents" còn mang weight âm (có lẽ vì hay đi với "waste of talents" trong review chê). Unigram không có ngữ cảnh nên không phân biệt được đâu là tên riêng / mô tả nội dung, đâu là đánh giá của người viết.

### Lỗi 3 — idx = 21483
- **True / Predicted / P(pos):** positive / negative / 0.150
- **Từ đẩy về positive:** enjoyable (+0.552), 9 (+0.495), fun (+0.484), love (+0.336), happy (+0.312)
- **Từ đẩy về negative:** nothing (-1.088), 4 (-0.571), just (-0.562), mental (-0.433), material (-0.424), unoriginal (-0.338)

> ✍️ **Tự trả lời:** Lỗi này thuộc loại nào? 

>
> Negation + lexical ambiguity (chơi chữ), kèm lỗi tokenize số.

> ✍️ **Tự trả lời:** Đoạn nào trong review là bằng chứng cho loại lỗi đó? Vì sao TF-IDF unigram không xử lý được?
>
> "in nothing less than enjoyable" nghĩa là rất thú vị, nhưng "nothing" lại là feature đẩy về negative mạnh nhất (-1.088). "original in an unoriginal way" là chơi chữ nhưng "unoriginal" vẫn bị tính là chê. "i give it a 6.4" bị tokenizer tách thành "6" và "4", mà "4" học từ các review chấm "4/10" nên mang weight âm. Unigram xử lý mỗi từ độc lập nên không hiểu được cụm "nothing less than", cũng không biết "6.4" là một điểm số.

## C. Tổng kết

> ✍️ **Tự trả lời:** Trong các lỗi đã xem, loại lỗi nào xuất hiện nhiều nhất? Điều đó có khớp với kết quả ở Experiment 4 (word order) và confusion matrix (§20) không?
>
> Nhiều nhất là lỗi do mất ngữ cảnh: negation / câu nhượng bộ ("nothing less than", "on paper ... though") và từ mang sentiment nằm trong tên riêng ("great expectations", "bad money"). Cả 3 lỗi đều quy về việc unigram coi mỗi từ là độc lập, khớp với Experiment 4: "not"/"nothing" luôn mang weight âm bất kể ngữ cảnh, và "good not bad" / "bad not good" cho cùng một kết quả. Bigram khắc phục được một phần (F1 0.8793 → 0.9021).
> Về confusion matrix: 2/3 lỗi là FN (positive bị đoán negative), khớp với FN > FP (1548 vs 1458) ở §20. Tuy nhiên chênh lệch nhỏ và dataset cân bằng nên đây không phải lỗi do class imbalance. Ngoài ra 2/3 lỗi có P(pos) gần 0.5, tức model sai chủ yếu ở các review khen chê lẫn lộn.