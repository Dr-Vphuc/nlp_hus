# Error Analysis — LAB 05 (Đề §24)

**Model:** GRU (val F1 = 0.8463), `max_length = 200`, truncation head · **Dataset:** IMDb test set  
**Cách chọn mẫu:** ngẫu nhiên với seed = 42 (notebook mục 11)  
**Token quan trọng:** occlusion: xóa từng token và đo P(pos) thay đổi bao nhiêu (`+` = token đẩy về positive, `−` = đẩy về negative; `@k` = vị trí token)

> Số liệu bên dưới copy từ output notebook mục 11. Các ô ✍️ và cột **Analysis** tự điền.

## Bảng tổng hợp

| # | Text (rút gọn) | True | Predicted | P(pos) | P(pos) TF-IDF + LR | n_tokens | Analysis |
|---|---|---|---|---|---|---|---|
| 15920 | overall, the dad's army movie is very funny, although the humour isn't quite as catchy and sparkly as in the t... | positive | positive | 0.793 | 0.961 | 163 | |
| 2129 | not to be mistaken as the highly touted samuel l. jackson vehicle snakes on a plane; snakes on a train is low ... | negative | negative | 0.198 | 0.229 | 164 | |
| 19021 | key west, for too short a time was "appointment tv" for my family. i'd stop by red lobster and pick up a party... | positive | positive | 0.872 | 0.950 | 133 | |
| 2840 | while this isn't an all time classic comedy it is a pretty good little movie to watch if you catch it on a rai... | negative | positive | 0.914 | 0.748 | 286 | |
| 19078 | dogtown and z-boys summary: dogtown and z-boys is a documentary about a group of revolutionary teenagers that ... | positive | negative | 0.136 | 0.805 | 844 | |
| 22095 | it's fun to watch a young james cagney doing his thing. he plays the cheapskate weights and measures guy who t... | positive | negative | 0.323 | 0.327 | 187 | |

## A. 3 mẫu phân loại đúng

### Mẫu 1 — idx = 15920
- **True / Predicted / P(pos):** positive / positive / 0.793
- **TF-IDF + LR:** P(pos) = 0.961 (đúng)
- **Độ dài:** 163 token — không bị cắt
- **Token đẩy về positive:** `series` (+0.060, @108), `unique` (+0.056, @112), `negative` (+0.043, @161), `it's` (+0.041, @110), `must` (+0.038, @137), `very` (+0.035, @6)
- **Token đẩy về negative:** `nothing` (-0.059, @94), `make` (-0.054, @151), `lamest` (-0.044, @140), `positive` (-0.043, @158), `mind` (-0.038, @154), `bbc` (-0.032, @129)
- **Context của các token ảnh hưởng mạnh nhất:**
  - +0.060 — `of the tv [series] but it's a`
  - -0.059 — `the movie holds [nothing] new as the`
  - +0.056 — `but it's a [unique] chance of seeing`
  - -0.054 — `universe i can't [make] up my mind`

<details><summary>Review gốc (tối đa 1500 ký tự)</summary>

overall, the dad's army movie is very funny, although the humour isn't quite as catchy and sparkly as in the tv and radio series. so where does this leave us, the viewers? if you've never seen dad's army then the movie is a good way of bringing yourself up to speed and getting hooked on the mad world of walmington o/s. the downside is that you might not "get it" because, as i said, the humour in the movie is a bit on the stolid side. for dad's army buffs the movie holds nothing new as the story is more or less a cutup of the tv series, but it's a unique chance of seeing your favourites in "high def" as compared to the shitty quality of the bbc video recordings. the movie also features what must be the lamest holdup sequence in the history of the universe. i can't make up my mind if that's a positive or a negative, though.

</details>

> ✍️ **Tự trả lời:** Model đúng nhờ những từ / đoạn context nào? Có đúng vì lý do hợp lý không (từ mang sentiment thật hay từ không liên quan)?
>
> 

### Mẫu 2 — idx = 2129
- **True / Predicted / P(pos):** negative / negative / 0.198
- **TF-IDF + LR:** P(pos) = 0.229 (đúng)
- **Độ dài:** 164 token — không bị cắt
- **Token đẩy về positive:** `aboard` (+0.110, @125), `passengers` (+0.106, @124), `size` (+0.060, @122), `best` (+0.059, @146), `castro` (+0.053, @158), `lives` (+0.052, @83)
- **Token đẩy về negative:** `receiving` (-0.228, @153), `credit` (-0.182, @155), `decent` (-0.158, @32), `this` (-0.145, @148), `flick` (-0.120, @151), `no` (-0.115, @143)
- **Context của các token ảnh hưởng mạnh nhất:**
  - -0.228 — `minute flick also [receiving] acting credit are`
  - -0.182 — `also receiving acting [credit] are alby castro`
  - -0.158 — `but some pretty [decent] visual effects an`
  - -0.145 — `the best of [this] 91 minute flick`

<details><summary>Review gốc (tối đa 1500 ký tự)</summary>

not to be mistaken as the highly touted samuel l. jackson vehicle snakes on a plane; snakes on a train is low budget, features no actors(to speak of), but some pretty decent visual effects. an attractive young woman(julia "rayanne" ruiz)does not want to marry someonelse's choice for her husband; so she is put under a powerful mayan curse that has snakes hatching inside her body, slowly devouring from the inside out. her only hope for survival is a shaman who lives across the border in texas. time is running out for her; and she is put on a train from el paso to los angeles. before long the snakes are leaving her pain wrecked body and rapidly growing in size. the passengers aboard the train are now trapped and soon to be snacks for the snakes. the finale sequence is no doubt the best of this 91 minute flick. also receiving acting credit are: alby castro, al galvez and giovanni bejarno.

</details>

> ✍️ **Tự trả lời:** Model đúng nhờ những từ / đoạn context nào? Có đúng vì lý do hợp lý không (từ mang sentiment thật hay từ không liên quan)?
>
> 

### Mẫu 3 — idx = 19021
- **True / Predicted / P(pos):** positive / positive / 0.872
- **TF-IDF + LR:** P(pos) = 0.950 (đúng)
- **Độ dài:** 133 token — không bị cắt
- **Token đẩy về positive:** `perfect` (+0.063, @69), `always` (+0.050, @90), `key` (+0.047, @0), `well` (+0.044, @126), `hurricane` (+0.034, @115), `dreams` (+0.033, @79)
- **Token đẩy về negative:** `rest` (-0.042, @129), `alone` (-0.035, @119), `couldn't` (-0.033, @71), `youtube` (-0.028, @110), `the` (-0.028, @131), `have` (-0.023, @121)
- **Context của các token ảnh hưởng mạnh nhất:**
  - +0.063 — `excellent fisher stevens [perfect] who couldn't fail`
  - +0.050 — `jennifer tilly was [always] remarkable and she`
  - +0.047 — `[key] west for too`
  - +0.044 — `an emmy as [well] as the rest`

<details><summary>Review gốc (tối đa 1500 ký tự)</summary>

key west, for too short a time was "appointment tv" for my family. i'd stop by red lobster and pick up a party tray for the night it was on. the irony of the situation was that i was working for a fox affiliate at the time, and every one at the station was incensed at them not renewing the show. everyone in that cast was excellent. fisher stevens... perfect. who couldn't fail to identify with an "everyman" who dreams of being a writer in key west? jennifer tilly was always remarkable (and she is one heck of a texas hold'em player). you can still find the pilot episode on youtube. wish they'd post the hurricane one. that episode alone, should have won an emmy, as well as the rest of the cast.

</details>

> ✍️ **Tự trả lời:** Model đúng nhờ những từ / đoạn context nào? Có đúng vì lý do hợp lý không (từ mang sentiment thật hay từ không liên quan)?
>
> 

## B. 3 mẫu phân loại sai

### Lỗi 1 — idx = 2840
- **True / Predicted / P(pos):** negative / positive / 0.914
- **TF-IDF + LR:** P(pos) = 0.748 (sai)
- **Độ dài:** 286 token — bị cắt: model chỉ đọc 200 token đầu, mất 86 token cuối
- **Token đẩy về positive:** `quite` (+0.034, @192), `also` (+0.020, @191), `perfect` (+0.020, @105), `good` (+0.020, @193), `portrayal` (+0.018, @170), `being` (+0.013, @194)
- **Token đẩy về negative:** `script` (-0.024, @189), `in` (-0.013, @196), `make` (-0.010, @103), `dignity` (-0.009, @177), `cobb` (-0.009, @172), `1934` (-0.009, @197)
- **Context của các token ảnh hưởng mạnh nhất:**
  - +0.034 — `script were also [quite] good being made`
  - -0.024 — `the story and [script] were also quite`
  - +0.020 — `and script were [also] quite good being`
  - +0.020 — `would make a [perfect] patsy to run`

<details><summary>Review gốc (tối đa 1500 ký tự)</summary>

while this isn't an all time classic comedy it is a pretty good little movie to watch if you catch it on a rainy saturday morning with not a lot else going on right then. harold lloyd plays ezekiel cobb, an american coming home after growing up in china where his dad was a missionary.he has come home from china to find an american wife and plans to return to china to continue his father's work.cobb unwittingly is recruited to run for mayor of his corrupt home town when the existing political machine that controls the town realizes that he would make a perfect patsy to run against the current mayor who also is the head of the town's underworld.they figure the bumbling ,stumbling cobb has no chance to win and therefore the current mayor continues to run the town and run his schemes which makes them all rich. lloyd isn't doing the physical comedy here like he did in his silent films.he does a convincing portrayal of cobb with a reserved understated dignity.the acting was good from all involved and the story and script were also quite good. being made in 1934 the film does have some rather racist language when talking about the chinese and it also has a typical black character from that era but these stereotypes aren't nearly as mean as i've seen from other films from that time. although not close to being lloyd's best film ,this movie does entertain and lloyd is very good as cobb.you won't be blown away by this film but it may be enough to peak your interest in lloyd and make yo ...

**Phần bị cắt (model không đọc):** does have some rather racist language when talking about the chinese and it also has a typical black character from that era but these stereotypes aren't nearly as mean as i've seen from other films from that time although not close to being lloyd's best film this movie does entertain and lloyd is very good as cobb you won't be blown away by this film but it may be enough to peak your interest in lloyd and make you want to see more of his work

</details>

> ✍️ **Tự trả lời:** Lỗi này thuộc loại nào? (negation / long-range dependency / sarcasm / ambiguous sentiment / rare word `<unk>` / domain-specific term / very long sequence / truncation / noise — có thể nhiều hơn một loại)
>
> 

> ✍️ **Tự trả lời:** Đoạn nào trong review là bằng chứng? Lỗi này có phải do sequence modeling hay không? (so với P(pos) của TF-IDF + LR, phần bị cắt, vị trí token quan trọng)
>
> 

### Lỗi 2 — idx = 19078
- **True / Predicted / P(pos):** positive / negative / 0.136
- **TF-IDF + LR:** P(pos) = 0.805 (đúng)
- **Độ dài:** 844 token — bị cắt: model chỉ đọc 200 token đầu, mất 644 token cuối
- **Token đẩy về positive:** `still` (+0.052, @174), `surfing` (+0.047, @177), `well` (+0.040, @175), `teenagers` (+0.031, @17), `and` (+0.026, @176), `documentary` (+0.025, @11)
- **Token đẩy về negative:** `serving` (-0.061, @188), `summary` (-0.060, @4), `drug` (-0.058, @191), `photographer` (-0.048, @128), `jay` (-0.041, @179), `except` (-0.036, @167)
- **Context của các token ảnh hưởng mạnh nhất:**
  - -0.061 — `the documentary was [serving] time on drug`
  - -0.060 — `and z boys [summary] dogtown and z`
  - -0.058 — `serving time on [drug] related charges chris`
  - +0.052 — `chris cahill are [still] well and surfing`

<details><summary>Review gốc (tối đa 1500 ký tự)</summary>

dogtown and z-boys summary: dogtown and z-boys is a documentary about a group of revolutionary teenagers that changed the world of surfing and skateboarding in venice (dogtown), california as we know it today. with their low pivotal style, they embarked on a larry bertlemen influenced journey that would lead to countless successes and a couple failures. after the dogtown articles were featured in a reinstated skateboarder magazine, the sport was revamped and the members of the zephyr skateboard team forgot about jeff ho, and looked to be on summer vacation for the rest of their lives by joining other skateboard teams that could afford to pay them like movie stars. the original zephyr skateboard team put together by jeff ho (zephyr surfshop owner) and craig stecyk (photographer) included jay adams, tony alva, stacy peralta, bob biniak, chris cahill, shogo kubo, paul constantineau, jim muir, peggy oki (the only female), nathan pratt, wentzle ruml iv, allen sarlo, and david ray perry. all of the original members except jay adams and chris cahill are still well and surfing/skating. jay adams, at the time of the documentary, was serving time on drug related charges. chris cahill, at the time of the documentary, was last seen in mexico. themes: the themes of this documentary are kind of read-between-the-lines, but if there were a clear-cut theme it would be that even kids can spark revolutions. other themes would include extensive partying in one's past may lead to an unfulfilling  ...

**Phần bị cắt (model không đọc):** the documentary was last seen in mexico themes the themes of this documentary are kind of read between the lines but if there were a clear cut theme it would be that even kids can spark revolutions other themes would include extensive partying in one's past may lead to an unfulfilling future and by planning and being careful with one's assets a very rewarding future could be at hand other works stacy peralta writer and director has a fairly wide range of documentaries that he has either written produced or directed most with a common theme of surfing or skateboarding such as riding giants sk8

</details>

> ✍️ **Tự trả lời:** Lỗi này thuộc loại nào? (negation / long-range dependency / sarcasm / ambiguous sentiment / rare word `<unk>` / domain-specific term / very long sequence / truncation / noise — có thể nhiều hơn một loại)
>
> 

> ✍️ **Tự trả lời:** Đoạn nào trong review là bằng chứng? Lỗi này có phải do sequence modeling hay không? (so với P(pos) của TF-IDF + LR, phần bị cắt, vị trí token quan trọng)
>
> 

### Lỗi 3 — idx = 22095
- **True / Predicted / P(pos):** positive / negative / 0.323
- **TF-IDF + LR:** P(pos) = 0.327 (sai)
- **Độ dài:** 187 token — không bị cắt
- **Token đẩy về positive:** `tough` (+0.075, @184), `fun` (+0.069, @168), `suspense` (+0.068, @137), `fun` (+0.066, @1), `almost` (+0.055, @115), `the` (+0.055, @170)
- **Token đẩy về negative:** `bad` (-0.228, @171), `guy` (-0.140, @185), `posturing` (-0.108, @186), `lack` (-0.085, @134), `do` (-0.069, @182), `doesn't` (-0.066, @92)
- **Context của các token ảnh hưởng mạnh nhất:**
  - -0.228 — `fun with the [bad] guys kind of`
  - -0.140 — `do his tough [guy] posturing`
  - -0.108 — `his tough guy [posturing]`
  - -0.085 — `me is a [lack] of sustained suspense`

<details><summary>Review gốc (tối đa 1500 ký tự)</summary>

it's fun to watch a young james cagney doing his thing. he plays the cheapskate weights and measures guy who takes his job very seriously, stepping on the toes of a group of crooked politicians. he is offered the world, but keeps his integrity. he is beaten and set up, but that's the problem. we never know if he is really in danger. they say he's in a spot, but still seems to have carte blanche to move around and do what he needs to do. at times he's so cocky he doesn't do much to protect himself. his allies are in the police department but just about everything else is pretty corrupt. he perseveres (almost too good to be true), of course, and we pull for him. the problem for me is a lack of sustained suspense. it would have been much better if he had had to clear his name. he never drops into the depths, even when rejected by his wife to be. it's still fun with the bad guys kind of imploding. see it just to watch cagney do his tough guy posturing.

</details>

> ✍️ **Tự trả lời:** Lỗi này thuộc loại nào? (negation / long-range dependency / sarcasm / ambiguous sentiment / rare word `<unk>` / domain-specific term / very long sequence / truncation / noise — có thể nhiều hơn một loại)
>
> 

> ✍️ **Tự trả lời:** Đoạn nào trong review là bằng chứng? Lỗi này có phải do sequence modeling hay không? (so với P(pos) của TF-IDF + LR, phần bị cắt, vị trí token quan trọng)
>
> 

## C. Ví dụ trong đề: "I expected the movie to be bad, but it was actually wonderful."

| Model | P(pos) | Predicted |
|---|---|---|
| TF-IDF + LR | 0.6880 | positive |
| RNN | 0.2589 | negative |
| LSTM | 0.4766 | negative |
| GRU | 0.6743 | positive |

Occlusion của GRU (token ảnh hưởng mạnh nhất):

| token | delta | context |
|---|---|---|
| wonderful | +0.4256 | `it was actually [wonderful]` |
| bad | -0.1858 | `movie to be [bad] but it was` |
| to | +0.0789 | `expected the movie [to] be bad but` |
| i | +0.0532 | `[i] expected the movie` |
| but | +0.0362 | `to be bad [but] it was actually` |
| the | +0.0254 | `i expected [the] movie to be` |
| actually | -0.0223 | `but it was [actually] wonderful` |
| movie | -0.0148 | `i expected the [movie] to be bad` |

> ✍️ **Tự trả lời:** Nếu model sai: negation, contrast, sequence length, vocabulary, training data, hay nguyên nhân khác?
>
> 

## D. Số liệu hỗ trợ: error rate theo độ dài review, so với TF-IDF + LR

| n_tokens | số review | error rate GRU (%) | error rate TF-IDF + LR (%) |
|---|---|---|---|
| ≤100 | 3163 | 12.74 | 11.16 |
| 101–200 | 11761 | 12.44 | 11.73 |
| 201–400 | 6971 | 20.01 | 12.32 |
| >400 | 3105 | 27.38 | 13.37 |

|  | số review |
|---|---|
| cả hai đúng | 19702 |
| GRU đúng, TF-IDF sai | 1187 |
| GRU sai, TF-IDF đúng | 2292 |
| cả hai sai | 1819 |
