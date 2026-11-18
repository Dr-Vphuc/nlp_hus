# Error analysis — LAB 09 (DistilBERT-cased NER, CoNLL-2003 test)

> Dữ liệu lấy từ `ner_finetuning.ipynb` mục 7 (`error_samples.csv`): 6 mẫu ngẫu nhiên (seed 42) cho mỗi loại lỗi của **Transformer**.
> Phần "Explanation" là **giả thuyết** dựa trên văn bản + hành vi model, không phải kết luận đã chứng minh (không chạy ablation riêng cho từng giả thuyết).
> AI policy: bản nháp do AI soạn từ output thực nghiệm; sinh viên cần đọc lại và chỉnh theo hiểu biết của mình.

## Tổng quan lỗi (Transformer, 5,648 entity gold, 5,776 entity dự đoán)

| Loại (định nghĩa entity-level) | Số lượng | Ghi chú |
|---|---|---|
| False positive "spurious" (dự đoán entity không chồng lấn entity gold nào) | 199 | ORG 81, MISC 60, LOC 29, PER 29 |
| False negative "missing" (entity gold không chồng lấn dự đoán nào) | 102 | ORG 37, MISC 26, PER 21, LOC 18 |
| Boundary error (cùng type, chồng lấn, span khác) | 118 | ORG 53, MISC 41, LOC 15, PER 9 |
| Wrong type (chồng lấn nhưng khác type) | 361 | LOC→ORG 76, ORG→LOC 58, MISC→ORG 44, LOC→MISC 34, ORG→MISC 32, ORG→PER 31, PER→ORG 29 |
| Tổng FP (`fp + boundary + wrong_type`) = 678 | | FN thực tế = 550 (một entity gold có thể bị nhiều dự đoán sai chồng lấn) |

Nhận xét tổng quát (có số liệu):
- **Lỗi lớn nhất là nhầm type (361/678 ≈ 53% FP), chứ không phải bỏ sót**: cặp LOC↔ORG chiếm 134 ca. Trong CoNLL, tên thành phố/quốc gia dùng như tên đội thể thao là ORG (hoặc ngược lại tuỳ quy ước annotate), nên ranh giới LOC/ORG phụ thuộc context và quy ước.
- ORG và MISC có F1 thấp nhất (0.861 / 0.790; PER 0.949, LOC 0.914), khớp với lỗi tập trung ở ORG/MISC.
- Độ tin cậy trung bình (xác suất nhãn tại token đầu của entity dự đoán) là **0.979 khi đúng** và **0.805 khi sai**: model sai thường kém chắc chắn hơn, nhưng nhiều lỗi vẫn rất tự tin (xem bảng: 0.99+ cho `Zulu`, `UAE`, `Mantua`).
- Entity chứa word bị chia subword không khó hơn đáng kể: recall exact-match 0.900 (3,029 entity có ít nhất một word bị tách) so với 0.906 (2,619 entity không bị tách) ⇒ **alignment (Strategy A) không phải nguồn lỗi chính** ở lab này. Entity viết toàn chữ HOA (1,013 entity) khó hơn rõ: 0.857 so với 0.913.
- Một số "lỗi" là **nhiễu annotation của CoNLL** (ví dụ `Mantua` bị dự đoán LOC nhưng gold là `O`); điều này làm F1 thực tế của model bị đánh giá thấp hơn.

## Bảng lỗi (24 mẫu)

Ký hiệu `[span](TYPE)`; `conf` là xác suất của nhãn dự đoán tại token đầu của span.

### False positives (entity dự đoán không có trong gold) — 6 mẫu

| # | Text | Gold | Prediction | Error type | Explanation |
|---|---|---|---|---|---|
| 416 | The injury-plagued Indianapolis Colts lost ... last year 's **AFC** finalists rallied ... Philadelphia Eagles 37-10 ... | O | [AFC](ORG), conf 0.905 | spurious entity / ambiguous (nhiễu annotation) | `AFC` là tên giải đấu/hội nghị, viết hoa toàn bộ, trong câu thể thao; dự đoán ORG hợp lý theo ngôn ngữ nhưng gold gán `O` (CoNLL không nhất quán với các từ viết tắt giải đấu). Giả thuyết: lỗi do annotation hơn là do model. |
| 1607 | **Feedlots** reporting moderate inquiry . | O | [Feedlots](ORG), conf 0.787 | spurious entity, domain shift | Câu bản tin thị trường nông sản, từ đầu câu viết hoa nên trông như tên riêng. Giả thuyết: model quen "từ đầu dòng viết hoa + không có động từ trước" thường là tên tổ chức trong bảng/báo cáo thể thao-tài chính; câu này quá ngắn, không đủ context. |
| 1611 | Slaughter Heifers - Select and **Choice** 2-3 1050-1100 lbs 67.00 . | O | [Choice](ORG), conf 0.967 | spurious entity, domain shift | `Choice` là hạng thịt (grade), không phải tên riêng; viết hoa giữa dòng dạng bảng giá. Rất tự tin (0.967) nhưng sai: model chưa thấy ngữ cảnh "cattle report" đủ nhiều. |
| 1713 | **Oats** 286.5 284.6 31.9 937.3 581.2 | O | [Oats](ORG), conf 0.864 | spurious entity, context-dependent | Dòng bảng số liệu bắt đầu bằng một từ viết hoa; cấu trúc `Tên + dãy số` giống bảng thể thao (`Hull 22 6 11 5 20 22 29` trong test, gold ORG), nên model suy ra ORG. Context dạng bảng gây nhầm giữa tên đội (ORG) và tên hàng hoá. |
| 2125 | Addressing a convention on Italian unity in **Mantua** , where the party has ... | O | [Mantua](LOC), conf 0.991 | spurious entity (nhiễu annotation) | Mantua là thành phố Ý; dự đoán LOC là **đúng về ngôn ngữ**, gold bỏ sót (annotation error). Tính vào FP nhưng không phải lỗi thật của model. |
| 2447 | Previous victories ( two ) : slalom , Lillehammer Norway , 1993 ; super **G** , Flachau , Austria , 1995 . | O | [G](MISC), conf 0.795 | spurious entity, ambiguous | `G` là ký hiệu bộ môn (super-G), một chữ cái viết hoa trong danh sách địa danh/giải đấu; model coi là MISC. Giả thuyết: entity 1 chữ cái hiếm gặp, không rõ biên của "super G". |

### False negatives (entity gold bị bỏ sót hoàn toàn) — 6 mẫu

| # | Text | Gold | Prediction | Error type | Explanation |
|---|---|---|---|---|---|
| 248 | CRICKET - AUSTRALIA V WEST INDIES **WORLD SERIES** SCOREBOARD . | [WORLD SERIES](MISC) | không có (O) | missing entity, ALL-CAPS headline | Tiêu đề toàn chữ HOA: mất tín hiệu viết hoa phân biệt tên riêng với từ thường. Cùng câu model chỉ bắt `AUSTRALIA` (LOC), bỏ sót cả `WEST INDIES` lẫn `WORLD SERIES`; entity MISC (tên giải) khó hơn PER/LOC. |
| 557 | NCAA AMERICAN FOOTBALL-**OHIO STATE** 'S PACE FIRST REPEAT LOMBARDI AWARD WINNER . | [FOOTBALL-OHIO STATE](MISC) | không có | missing entity, tokenization / hyphen, ALL-CAPS | Gold gán `FOOTBALL-OHIO STATE` là MISC (annotation lạ: một word có gạch nối `FOOTBALL-OHIO` đi cùng `STATE`). Cả câu model đều lệch (đoán `NCAA AMERICAN` là MISC thay vì `NCAA` ORG, `LOMBARDI AWARD` là ORG thay vì MISC, bỏ sót `PACE`). Từ gạch nối + HOA bị tách thành nhiều subword. Gold khó đoán, không phải lỗi hoàn toàn của model. |
| 689 | SOCCER - WEAH HEAD-BUTT DEPRIVES **PORTUGAL** OF COSTA . | [PORTUGAL](LOC) | không có | missing entity, ALL-CAPS, context-dependent | Tên quốc gia nằm trong tiêu đề HOA, sau một chuỗi từ HOA lạ (`HEAD-BUTT DEPRIVES`). Cả câu model chỉ dự đoán `COSTA` (LOC, sai type: gold PER) và bỏ sót luôn `WEAH`. Giả thuyết: không có tín hiệu hình dạng chữ nên model phải dựa hoàn toàn vào ngữ nghĩa, và `PORTUGAL OF COSTA` rất lạ. |
| 1129 | MIDEAST / RED SEA - | [RED SEA](LOC) | không có | missing entity, thiếu context | Câu chỉ 5 token, toàn chữ HOA, không có ngữ cảnh. Địa danh 2 từ cần context để xác định là entity. |
| 1908 | COAL - Lantau Peak - 120,000 tones coal Hay Point or Newcastle / Kaohsiung 20-30/1 $ 5.25 ... fio respectively ... **China Steel** . | [China Steel](ORG) | không có | missing entity, domain shift (dòng giao dịch hàng hải) | Văn bản dạng ghi chú viết tắt/số liệu; tên công ty đứng cuối dòng mà không có từ chỉ thị (Inc., Ltd.). Domain khác xa văn bản tin thường. |
| 3388 | BOXING - SCHULZ DEFEATS **RIBALTA** IN IBF HEAVYWEIGHT FIGHT . | [RIBALTA](PER) | không có | missing entity, rare entity, ALL-CAPS | Họ tên võ sĩ hiếm, viết HOA, bị tách thành 4 subword (tên ngoại lai). Model nhận ra `SCHULZ` (PER, đúng) nhưng bỏ sót `RIBALTA` (vị trí tân ngữ sau `DEFEATS`). Giả thuyết: entity hiếm + HOA (`RIBALTA` tách thành `R ##IB ##AL ##TA`). Ngoài ra `IBF` bị đoán MISC thay vì ORG trong cùng câu. |

### Boundary errors (đúng type, sai biên) — 6 mẫu

| # | Text | Gold | Prediction | Error type | Explanation |
|---|---|---|---|---|---|
| 74 | SOCCER - **ASIAN CUP GROUP** C RESULTS . | [ASIAN CUP](MISC) | [ASIAN CUP GROUP](MISC), conf 0.992 | wrong boundary (over-extension) | Model kéo dài span sang `GROUP`. Annotation CoNLL không nhất quán trong việc có gộp "Group C" vào tên giải hay không; trong tiêu đề HOA, cả cụm trông như một tên. |
| 962 | ... but **van Boxmeer** says such a threat might be exaggerated ... | [Boxmeer](PER) | [van Boxmeer](PER), conf 0.882 | wrong boundary (over-extension), annotation inconsistency | Gold chỉ gán `Boxmeer` (không có `van`) dù ở các câu khác tên có `van` thuộc entity (ví dụ `Nikolaus van der Pas`). Dự đoán của model hợp lý theo quy ước chung; lỗi do annotation không nhất quán. |
| 1068 | ... flight data processing system at the **Oceanic Control Centre** in Prestwick ... | [Oceanic Control Centre](LOC) | [Control Centre](LOC), conf 0.564 | wrong boundary (truncated left) | Entity 3 từ; model tách thành hai entity: `Oceanic` (ORG) + `Control Centre` (LOC, conf 0.564). `Oceanic` bị tách subword (`Ocean ##ic`) và bản thân là từ dễ gợi tên công ty. conf thấp (0.564) cho thấy model không chắc. Phù hợp dự đoán 2: entity dài dễ sai biên hơn. |
| 1596 | ... a mixed **Christian-Shi'ite Moslem** force ... | [Moslem](MISC) | [Christian-Shi'ite Moslem](MISC), conf 0.997 | wrong boundary (over-extension), tokenization | `Christian-Shi'ite` là một word có gạch nối + dấu nháy, bị tách thành nhiều subword; model gộp cả cụm. Gold chỉ coi `Moslem` là MISC (không nhất quán với cách xử lý hyphen khác). |
| 2068 | ... expelled from **St Pius X High School** in Albuquerque ... | [St Pius X High School](ORG) | [High School](ORG), conf 0.69 | wrong boundary (truncated left), rare entity | Tên trường 5 token (nhóm entity 4+), phần đầu `St Pius X` hiếm. Model nhận phần "High School" (thường là ORG) nhưng không mở rộng sang trái. |
| 2270 | **Algoa Day** 21/11/96 6,000 Africa | [Algoa Day](MISC) | [Day](MISC), conf 0.791 | wrong boundary (truncated left), rare entity, domain shift | Dòng dữ liệu tàu biển; `Algoa` (địa danh hiếm) bị bỏ. Thiếu context. |

### Wrong type (chồng lấn nhưng sai loại) — thêm 6 mẫu, vì đây là loại lỗi lớn nhất

| # | Text | Gold | Prediction | Error type | Explanation |
|---|---|---|---|---|---|
| 888 | ... between the ANC and **Zulu** Chief Mangosuthu Buthelezi 's Inkatha Freedom Party ... | [Zulu](MISC) | [Zulu](ORG), conf 0.890 | wrong entity type, ambiguous | `Zulu` (dân tộc/ngôn ngữ) đứng cạnh `ANC`, `Inkatha Freedom Party` (ORG) nên bị kéo sang ORG; ranh giới giữa nhóm dân tộc (MISC) và tổ chức chính trị (ORG) mơ hồ. |
| 1571 | " It shows that Sakakibara is not for a stronger dollar either , " said **Sumitomo** 's Note . | [Sumitomo](ORG) | [Sumitomo](PER), conf 0.999 | wrong entity type, context-dependent | Cấu trúc `said X 's Note` giống `said <tên người>`; model rất tự tin (0.999) PER dù `Sumitomo` thường là công ty. Giả thuyết: mẫu "said + tên + 's" mạnh hơn kiến thức pretrain về từ `Sumitomo`. |
| 2039 | **LOS ANGELES** 1996-12-05 | [LOS ANGELES](ORG) | [LOS ANGELES](LOC), conf 0.968 | wrong entity type, context-dependent / annotation | Gold ORG (có thể là tên đội thể thao theo thành phố, vì ngữ cảnh bài không có trong câu này), model đoán LOC theo dạng dateline `CITY date`. Dateline giống hệt `NEW YORK 1996-12-06` (gold LOC), nên gần như không thể phân biệt nếu không có context bài. |
| 2382 | " In the coming five to ten years , **Ahold** plans to open many more stores ... " **Ahold** said in a statement . | [Ahold](ORG) | [Ahold](PER), conf 0.719 | wrong entity type, rare entity | `Ahold` (công ty Hà Lan) hiếm và bị tách subword (`Ah ##old`); `plans to open stores` lẽ ra đủ gợi ORG nhưng model dự đoán PER ở **cả hai** lần xuất hiện (conf 0.719 ở lần đầu). |
| 3238 | **South Korea** 4 Indonesia 2 ( 3-0 ) | [South Korea](LOC) | [South Korea](ORG), conf 0.996 | wrong entity type, context-dependent (quy ước annotation) | Kết quả thi đấu đội tuyển quốc gia. CoNLL gán quốc gia dùng làm đội tuyển là LOC, nhưng pattern `Team score Team score` thường là ORG ở các bảng CLB. Model rất tự tin nhưng ngược quy ước. |
| 3247 | **UAE** 2 1 1 0 4 3 4 | [UAE](LOC) | [UAE](ORG), conf 0.998 | wrong entity type, ambiguous (viết tắt + dạng bảng) | Tương tự #3238: viết tắt quốc gia trong bảng xếp hạng. |

## Giải thích tại sao các loại entity khác nhau (liên hệ Exp 4)

| Entity | F1 Transformer / Baseline | Giải thích |
|---|---|---|
| PER | 0.949 / 0.800 | Nhiều ví dụ train (6,600), pattern từ vựng rõ (`Mr`, `said`, họ tên 2 từ), ít nhập nhằng. Lỗi chủ yếu: họ hiếm viết HOA, hoặc tên công ty bị nhầm là người (`Sumitomo`, `Ahold`). |
| LOC | 0.914 / 0.837 | Nhiều ví dụ (7,140) và địa danh có tính từ vựng cao (kiến thức pretraining giúp mạnh). Lỗi chủ yếu do **ambiguity LOC↔ORG** (thành phố/quốc gia = đội thể thao). |
| ORG | 0.861 / 0.666 | Khó nhất trong 3 loại chính: tên tổ chức đa dạng, có nhiều dạng nhiều token (`St Pius X High School`), nhập nhằng với LOC/PER, hiếm (domain shift sang bảng thể thao/thương mại). Baseline kém rất nhiều vì ORG cần context. |
| MISC | 0.790 / 0.753 | Lớp "tạp" (tính từ quốc tịch, giải đấu, ngôn ngữ, sự kiện) nên không có pattern từ vựng thống nhất; ít ví dụ nhất (3,438); boundary khó xác định (`ASIAN CUP` vs `ASIAN CUP GROUP`). |

## Context analysis — cùng surface form, khác entity type

Tìm trong gold test: các surface form xuất hiện với ≥ 2 type khác nhau (mỗi type ≥ 2 lần). Có 24 surface form như vậy (218 occurrence; xem notebook mục 8). Ví dụ:

| Surface form | Context 1 (gold) | Context 2 (gold) | Baseline | Transformer |
|---|---|---|---|---|
| **Barcelona** | `meeting in Barcelona , decided ...` (LOC); `heroin in a taxi in Barcelona` (LOC) | `Goalkeepers - Vitor Baia ( Barcelona , Spain )` (ORG); `Fernando Couto ( Barcelona )` (ORG) | sai ở `Vitor Baia ( Barcelona , Spain )` (đoán LOC) | đúng ở các ví dụ hiển thị (`meeting in` → LOC; tên cầu thủ + `( Barcelona )` → ORG); tổng 17/24 occurrence đúng, baseline 18/24 |
| **Santa Fe** | `Wall St speculates about Santa Fe savior` (LOC) | `Santa Fe is so far mum ...`, `Santa Fe rejects unsolicited suitor Newmont` (ORG) | không dự đoán entity nào ở cả 18 lần (0/18 đúng) | đúng 9/18 (hay đảo type giữa các câu cùng một bài viết) |
| **Washington / Chicago / New York** | `WASHINGTON 1996-12-06`, `MONTREAL AT CHICAGO` (LOC, dateline / địa điểm trận đấu) | `WASHINGTON 7 9 .438 6`, `CHICAGO 17 1 .944 -` (ORG, đội trong bảng xếp hạng) | đúng hết | đúng hết (cùng từ nhưng khác định dạng dòng: dateline vs bảng điểm) |

Thêm các câu tự viết (ngoài test, dự đoán của Transformer, xem notebook mục 8):
- `Washington signed the treaty with Moscow .` → `Washington` = **LOC**; `Washington beat Dallas 24-17 on Sunday .` → `Washington` = **ORG** (cùng từ, khác động từ/context).
- `Jordan scored 30 points for Chicago .` → `Jordan` = **PER**; `The trip to Jordan was long .` → `Jordan` = **LOC**.
- `Leeds won 2-0 at Everton .` → `Leeds` = **ORG**; `She was born in Leeds .` → `Leeds` = **LOC**.

Tổng hợp trên 218 occurrence: baseline đúng (span + type) 155 (71.1%), Transformer đúng 176 (80.7%).

**Vì sao contextual representation có lợi thế so với representation tĩnh (LAB 03)?** Static embedding gán *một vector duy nhất cho mỗi từ* (word type), nên `Washington` (người, bang, đội bóng) có cùng một vector bất kể câu nào; phần phân biệt phải do tầng sau đảm nhận bằng cách nhìn các từ lân cận một cách rời rạc. Contextual representation (LAB 08/09) tính vector của `Washington` *sau khi* self-attention đã trộn thông tin từ cả câu (`signed the treaty with Moscow` vs `beat Dallas 24-17`), nên vector ở hai câu khác nhau và một lớp tuyến tính đủ để tách LOC/ORG. Baseline của lab (đặc trưng từ ±2) có một phần thông tin ngữ cảnh nhưng chỉ là cửa sổ cố định, không học được quan hệ xa hay tổ hợp; kết quả ở `Santa Fe` (0/18) và `Barcelona` thể hiện sự khác biệt. Lưu ý cẩn thận: nhiều trường hợp trong test là bảng thể thao (định dạng dòng đã đủ để đoán type), nên bảng trên không chỉ đo "hiểu ngữ nghĩa" mà còn đo khả năng dùng cấu trúc dòng; hơn nữa `Santa Fe` cho thấy Transformer vẫn chưa phân biệt tốt (9/18).

## Đối chiếu với Prediction ban đầu
- **Prediction 1** (Transformer vượt baseline rõ rệt; baseline 0.60–0.75, Transformer 0.88–0.92): Transformer 0.8925 nằm trong khoảng dự đoán; baseline 0.7652 hơi **cao hơn** khoảng dự đoán (0.75 trên cùng) — tức tôi đã đánh giá baseline hơi thấp.
- **Prediction 2** (entity dài khó hơn): F1 Transformer theo độ dài: 1 token 0.888, 2 token 0.910, 3 token 0.848, 4+ token 0.812; baseline 0.745 / 0.832 / 0.655 / 0.495. Xu hướng giảm từ 2 token trở đi đúng như dự đoán, nhưng entity 1 token **thấp hơn** entity 2 token (do nhập nhằng ORG/LOC/MISC của các tên một từ), giống điều tôi đã nêu là ngoại lệ có thể xảy ra. Nhóm 4+ chỉ 62 entity nên F1 không ổn định.
- **Prediction 3** (entity hiếm/chưa thấy gây lỗi nhiều hơn): đúng. Transformer F1: frequent (≥5 lần trong train) 0.945, rare (1–4) 0.899, unseen 0.846; Baseline: 0.922 / 0.789 / 0.623. Khoảng cách giữa hai model lớn nhất ở entity chưa thấy (+0.22 F1), phù hợp dự đoán.
