# Reflection — LAB 05

> Tự viết (AI policy §27: không dùng AI cho reflection).

## 1. Sequence model khác static representation ở điểm nào?
So với TF-IDF / Average Word2Vec ở LAB 04, thêm hidden state $h_t = f(x_t, h_{t-1})$ thay đổi điều gì?
Bằng chứng nào trong notebook (mục 6, 7) cho thấy điều đó?

## 2. Prediction nào của mình sai? Vì sao?
(đối chiếu `prediction.md` với kết quả)

## 3. Mô hình tốt hơn có phải luôn là mô hình phức tạp hơn?
(accuracy / F1 so với số tham số, training time, memory — mục 7, 8, 10)

## 4. Giới hạn của recurrent models
`x1 → x2 → x3 → ... → x100`: muốn dùng thông tin ở $x_1$ khi xử lý $x_{100}$, thông tin phải đi qua bao nhiêu bước?
Điều đó gây ra vấn đề gì về gradient, về tốc độ (có song song hóa được không)?

## 5. RNN → Attention
Có cách nào để một token trực tiếp "nhìn" những token quan trọng khác trong sequence mà không phải truyền
toàn bộ thông tin qua một chuỗi recurrent dài? (§32)

## 6. Nếu làm lại lab này, mình sẽ thay đổi gì?
