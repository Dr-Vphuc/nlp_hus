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

**Bước 1:** $W_{xh}x_t = \begin{bmatrix}0.5\cdot 1 + 0.2\cdot 0\\0.1\cdot 1 + 0.4\cdot 0\end{bmatrix} = \begin{bmatrix}0.5\\0.1\end{bmatrix}$

**Bước 2:** $W_{hh}h_{t-1} = \begin{bmatrix}0.3\cdot 0.5 + 0.1\cdot 0.2\\0.2\cdot 0.5 + 0.5\cdot 0.2\end{bmatrix} = \begin{bmatrix}0.15 + 0.02\\0.10 + 0.10\end{bmatrix} = \begin{bmatrix}0.17\\0.20\end{bmatrix}$

**Bước 3:** $a_t = W_{xh}x_t + W_{hh}h_{t-1} + b_h = \begin{bmatrix}0.5\\0.1\end{bmatrix} + \begin{bmatrix}0.17\\0.20\end{bmatrix} + \begin{bmatrix}0\\0\end{bmatrix} = \begin{bmatrix}0.67\\0.30\end{bmatrix}$

**Bước 4:** $h_t = \tanh(a_t) = \begin{bmatrix}\tanh(0.67)\\\tanh(0.30)\end{bmatrix} \approx \begin{bmatrix}0.5850\\0.2913\end{bmatrix}$

> Nhận xét: vì $x_t = [1, 0]^\top$ nên $W_{xh}x_t$ chính là cột thứ nhất của $W_{xh}$. Hàm tanh nén $a_t$ về khoảng $(-1, 1)$; $h_t$ vừa chứa thông tin của input hiện tại, vừa chứa thông tin từ $h_{t-1}$.

---

## Bài 2 — Information flow (§7)

Sequence: `"I did not like this movie"`, model đọc từ trái sang phải.

| Bước | Token | Hidden state $h_t$ mang thông tin gì? |
|---|---|---|
| 1 | I | Chỉ có thông tin của "I" (vì $h_0 = 0$): chủ ngữ ngôi thứ nhất, người nói đang nêu cảm nhận của chính mình. |
| 2 | did | "I did": chủ ngữ + trợ động từ quá khứ, báo hiệu sắp có động từ chính. |
| 3 | not | "I did not": thêm tín hiệu **phủ định**, được lưu trong $h_3$ và "chờ" động từ phía sau để áp lên. |
| 4 | like | "I did not like": phủ định kết hợp với động từ cảm xúc tích cực, nên cảm xúc chung chuyển sang **tiêu cực**. |
| 5 | this | "I did not like this": từ hạn định, báo hiệu sắp có tân ngữ; thông tin "không thích" tiếp tục được mang theo (có thể suy giảm dần). |
| 6 | movie | Tóm tắt cả câu: người nói không thích bộ phim này, tức sentiment **negative**. $h_6$ thường được đưa vào classifier. |

*Lưu ý: $h_t$ là một vector, không diễn giải trực tiếp được từng chiều. Bảng trên mô tả thông tin mà $h_t$ có thể mã hóa nếu model được huấn luyện tốt.*

**Tại thời điểm model đọc `"like"`, thông tin `"not"` có thể ảnh hưởng tới hidden state như thế nào?**

$$
h_4 = \tanh\left(W_{xh}x_{\text{like}} + W_{hh}h_3 + b_h\right),\qquad h_3 = \tanh\left(W_{xh}x_{\text{not}} + W_{hh}h_2 + b_h\right)
$$

- Thông tin của "not" đã được ghi vào $h_3$, và đi vào $h_4$ qua số hạng $W_{hh}h_3$.
- Cùng một input $x_{\text{like}}$, nhưng $h_3$ khác nhau giữa "I did like" và "I did not like", nên pre-activation $a_4$ bị dịch chuyển và $h_4$ khác nhau. Nghĩa của "like" được hiểu **theo ngữ cảnh**.
- Qua huấn luyện, model có thể học $W_{xh}, W_{hh}$ sao cho thành phần phủ định trong $h_3$ đẩy $h_4$ về vùng "tiêu cực" thay vì "tích cực". Tính phi tuyến của tanh cùng các bước tiếp theo cho phép biểu diễn tương tác "not × like".
- Thông tin "not" còn tiếp tục truyền sang $h_5, h_6$ qua chuỗi $h_t = f(x_t, h_{t-1})$, dù có thể yếu dần theo khoảng cách.

**So sánh với Average Word2Vec:**

$$
v = \frac{1}{6}\left(e_{\text{I}} + e_{\text{did}} + e_{\text{not}} + e_{\text{like}} + e_{\text{this}} + e_{\text{movie}}\right)
$$

- "not" chỉ cộng thêm một vector cố định $\tfrac{1}{6}e_{\text{not}}$, **không phụ thuộc** vào việc nó đứng ở đâu hay phủ định từ nào. Không có tương tác giữa "not" và "like".
- $e_{\text{like}}$ (gần các từ tích cực như "love", "enjoy") vẫn kéo $v$ về phía tích cực, nên dễ phân loại sai.
- Mọi hoán vị của 6 từ cho cùng một vector $v$, nên thông tin thứ tự bị mất hoàn toàn.
- RNN ngược lại: xử lý tuần tự, trạng thái tại "like" đã "biết" có "not" phía trước.

---

## Bài 3 — Sequence order (§8)

S1 = `"dog bites man"`, S2 = `"man bites dog"`

**Average Word2Vec có thể phân biệt hai câu này dựa trên thứ tự hay không?** (giải thích bằng công thức)

$$
v(S_1) = \frac{1}{3}\left(e_{\text{dog}} + e_{\text{bites}} + e_{\text{man}}\right) = \frac{1}{3}\left(e_{\text{man}} + e_{\text{bites}} + e_{\text{dog}}\right) = v(S_2)
$$

**Không.** Phép cộng có tính giao hoán nên hai vector **giống hệt nhau** ($\cos(v(S_1), v(S_2)) = 1$). Mọi classifier đặt trên $v$ đều cho cùng một output cho hai câu. Thông tin thứ tự bị mất ngay từ bước biểu diễn, trước cả khi huấn luyện.

**RNN có khả năng phân biệt chúng dựa trên thứ tự như thế nào?** (dựa trên cơ chế $h_t = f(x_t, h_{t-1})$,
không chỉ dựa trên kết quả thực nghiệm)

Hidden state cuối là một **hàm hợp lồng nhau**, phụ thuộc thứ tự input:

$$
S_1:\ h_3 = f\big(x_{\text{man}},\ f(x_{\text{bites}},\ f(x_{\text{dog}}, h_0))\big)
$$

$$
S_2:\ h_3 = f\big(x_{\text{dog}},\ f(x_{\text{bites}},\ f(x_{\text{man}}, h_0))\big)
$$

Để thấy rõ, xét xấp xỉ tuyến tính (bỏ tanh, $b_h = 0$, $h_0 = 0$):

$$
h_1 = W_{xh}x_1,\quad h_2 = W_{hh}W_{xh}x_1 + W_{xh}x_2,\quad h_3 = W_{hh}^2W_{xh}x_1 + W_{hh}W_{xh}x_2 + W_{xh}x_3
$$

Mỗi vị trí bị nhân với một ma trận khác nhau ($W_{hh}^2$, $W_{hh}$, $I$), tức **vị trí được mã hóa** qua số lần đi qua $W_{hh}$. Hiệu giữa hai câu:

$$
h_3^{(S_1)} - h_3^{(S_2)} = \left(W_{hh}^2 - I\right)W_{xh}\left(x_{\text{dog}} - x_{\text{man}}\right) \neq 0 \text{ (nói chung)}
$$

Ví dụ với $W_{hh}$ của Bài 1: $W_{hh}^2 - I = \begin{bmatrix}-0.89 & 0.08\\0.16 & -0.73\end{bmatrix}$, $\det \approx 0.637 \neq 0$; và $\det W_{xh} = 0.18 \neq 0$. Vậy chỉ cần $x_{\text{dog}} \neq x_{\text{man}}$ là $h_3$ của hai câu khác nhau.

Khi có tanh (phi tuyến), $f(a, f(b, \cdot)) \neq f(b, f(a, \cdot))$ nói chung, nên thứ tự càng ảnh hưởng mạnh. Như vậy RNN **có khả năng về mặt cấu trúc** để phân biệt "ai cắn ai". Việc nó có thực sự dùng thông tin này đúng hay không phụ thuộc vào dữ liệu và quá trình huấn luyện.

---

## Vanishing / exploding gradient (§9)

**Vì sao sequence càng dài có thể khiến RNN khó học dependency ở xa?**
(gợi ý: tích các Jacobian $\prod_k \partial h_k / \partial h_{k-1}$; minh họa số ở mục 4.4 của notebook)

Với $h_k = \tanh(W_{xh}x_k + W_{hh}h_{k-1} + b_h)$:

$$
\frac{\partial h_k}{\partial h_{k-1}} = \mathrm{diag}\left(1 - h_k^2\right)W_{hh}
$$

Gradient của loss (tính ở bước cuối $T$) truyền ngược về bước $t$:

$$
\frac{\partial \mathcal{L}}{\partial h_t} = \frac{\partial \mathcal{L}}{\partial h_T}\cdot\frac{\partial h_T}{\partial h_t},\qquad
\frac{\partial h_T}{\partial h_t} = \prod_{k=t+1}^{T}\frac{\partial h_k}{\partial h_{k-1}}
$$

$$
\left\|\frac{\partial h_T}{\partial h_t}\right\| \le \prod_{k=t+1}^{T}\left\|\mathrm{diag}(1-h_k^2)\right\|\,\left\|W_{hh}\right\| \le \left(\gamma\,\sigma_{\max}(W_{hh})\right)^{T-t},\quad \gamma = \max \tanh' \le 1
$$

- Khoảng cách $T - t$ càng lớn thì tích càng có nhiều thừa số, nên gradient thay đổi theo hàm mũ.
- Vanishing: nếu $\gamma\,\sigma_{\max} < 1$, gradient từ token ở xa tiến về 0. Ví dụ "not" ở đầu một review dài gần như không đóng góp vào cập nhật trọng số, nên model chỉ học được dependency ngắn. Khi tanh bão hòa ($|h_k| \to 1$) thì $1 - h_k^2 \to 0$, càng làm vanishing nặng hơn.
- Exploding: nếu $W_{hh}$ có trị riêng/giá trị suy biến lớn hơn 1, tích có thể tăng theo hàm mũ, khiến training không ổn định (loss nhảy, NaN). Cách xử lý thường dùng: gradient clipping.

Minh họa số (giả sử mỗi Jacobian có độ lớn xấp xỉ một hằng số $r$):

| Số bước $n$ | $0.9^n$ | $0.5^n$ | $1.1^n$ |
|---|---|---|---|
| 10 | 0.349 | $9.8\times10^{-4}$ | 2.59 |
| 50 | $5.2\times10^{-3}$ | $8.9\times10^{-16}$ | 117.4 |
| 100 | $2.7\times10^{-5}$ | $7.9\times10^{-31}$ | 13 781 |

Với chính $W_{hh}$ của Bài 1: $\sigma_{\max}(W_{hh}) \approx 0.583$, nên cận trên của gradient sau 10 bước là $0.583^{10} \approx 4.6\times10^{-3}$, sau 20 bước là $\approx 2.1\times10^{-5}$. Jacobian thực tế tại bước của Bài 1: $\mathrm{diag}(0.6578,\ 0.9151)\,W_{hh} = \begin{bmatrix}0.1973 & 0.0658\\0.1830 & 0.4576\end{bmatrix}$, còn nhỏ hơn $W_{hh}$, tức gradient bị co lại ở mỗi bước.

---

## Bài tập LSTM gates (§17)

Cho: forget gate $f_t = 0.9$, input gate $i_t = 0.2$, candidate $\tilde{c}_t = 0.5$, $c_{t-1} = 0.8$.

$$c_t = f_t c_{t-1} + i_t \tilde{c}_t = 0.9 \times 0.8 + 0.2 \times 0.5 = 0.72 + 0.10 = 0.82$$

Giữ lại 90% bộ nhớ cũ (0.72) và chỉ thêm một chút thông tin mới (0.10), nên $c_t$ gần với $c_{t-1}$.

**Nếu forget gate gần 1, model có xu hướng giữ hay loại bỏ thông tin cũ?**

**Giữ lại.** Khi $f_t \approx 1$ thì $c_t \approx c_{t-1} + i_t\tilde{c}_t$: bộ nhớ cũ được truyền gần như nguyên vẹn sang bước sau. Đồng thời, theo đường cell state:

$$
\frac{\partial c_t}{\partial c_{t-1}} = f_t \approx 1 \quad\Rightarrow\quad \frac{\partial c_T}{\partial c_t} = \prod_{k=t+1}^{T} f_k \approx 1
$$

nên gradient không bị tiêu biến qua nhiều bước. Đây là lý do LSTM học dependency xa tốt hơn simple RNN (ví dụ giữ thông tin "not" qua một câu dài).

**Nếu forget gate gần 0 thì sao?**

**Loại bỏ (quên) thông tin cũ.** Khi $f_t \approx 0$ thì $c_t \approx i_t\tilde{c}_t$: cell state gần như được reset và chỉ còn phụ thuộc vào input hiện tại. Ví dụ với $f_t = 0.05$ (các giá trị khác giữ nguyên): $c_t = 0.05 \times 0.8 + 0.1 = 0.14$, thay vì 0.82. Điều này hữu ích khi ngữ cảnh thay đổi (ví dụ như sang câu mới, chủ ngữ mới) và thông tin cũ không còn liên quan. Đổi lại, gradient qua đường cell state tại bước đó cũng bị cắt.