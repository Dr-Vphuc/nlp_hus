# Calculations — LAB 05

> Phần này tự làm tay (AI policy §27: không dùng AI cho calculation, giải thích RNN forward pass, LSTM gates).
> Có thể chụp ảnh bài làm tay và để trong thư mục `calculations/` như LAB 04.
> Sau khi tính xong, có thể đối chiếu Bài 1 bằng mục 4.2 của notebook.

## Bài 1 — Một bước RNN (§6)

Cho

$$
x_t = \begin{bmatrix}1\\0\end{bmatrix},\quad
h_{t-1} = \begin{bmatrix}0.5\\0.2\end{bmatrix},\quad
W_{xh} = \begin{bmatrix}0.5 & 0.2\\0.1 & 0.4\end{bmatrix},\quad
W_{hh} = \begin{bmatrix}0.3 & 0.1\\0.2 & 0.5\end{bmatrix},\quad
b_h = \begin{bmatrix}0\\0\end{bmatrix}
$$

**Bước 1:** $W_{xh}x_t = $

**Bước 2:** $W_{hh}h_{t-1} = $

**Bước 3:** $a_t = W_{xh}x_t + W_{hh}h_{t-1} + b_h = $

**Bước 4:** $h_t = \tanh(a_t) = $

---

## Bài 2 — Information flow (§7)

Sequence: `"I did not like this movie"`, model đọc từ trái sang phải.

| Bước | Token | Hidden state $h_t$ mang thông tin gì? |
|---|---|---|
| 1 | I | |
| 2 | did | |
| 3 | not | |
| 4 | like | |
| 5 | this | |
| 6 | movie | |

**Tại thời điểm model đọc `"like"`, thông tin `"not"` có thể ảnh hưởng tới hidden state như thế nào?**

**So sánh với Average Word2Vec:**

---

## Bài 3 — Sequence order (§8)

S1 = `"dog bites man"`, S2 = `"man bites dog"`

**Average Word2Vec có thể phân biệt hai câu này dựa trên thứ tự hay không?** (giải thích bằng công thức)

**RNN có khả năng phân biệt chúng dựa trên thứ tự như thế nào?** (dựa trên cơ chế $h_t = f(x_t, h_{t-1})$,
không chỉ dựa trên kết quả thực nghiệm)

---

## Vanishing / exploding gradient (§9)

**Vì sao sequence càng dài có thể khiến RNN khó học dependency ở xa?**
(gợi ý: tích các Jacobian $\prod_k \partial h_k / \partial h_{k-1}$; minh họa số ở mục 4.4 của notebook)

---

## Bài tập LSTM gates (§17)

Cho: forget gate $f_t = 0.9$, input gate $i_t = 0.2$, candidate $\tilde{c}_t = 0.5$, $c_{t-1} = 0.8$.

$$c_t = f_t c_{t-1} + i_t \tilde{c}_t = $$

**Nếu forget gate gần 1, model có xu hướng giữ hay loại bỏ thông tin cũ?**

**Nếu forget gate gần 0 thì sao?**
