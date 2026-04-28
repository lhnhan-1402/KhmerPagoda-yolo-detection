# Ứng Dụng Mô Hình YOLO Trong Nhận Dạng Hoa Văn Trang Trí Chùa Khmer Nam Bộ

> Đề tài ứng dụng Computer Vision và Deep Learning nhằm phát hiện các hoa văn trang trí trong kiến trúc Chùa Khmer Nam Bộ từ hình ảnh.

---

## Giới thiệu đề tài

Chùa Khmer Nam Bộ là một trong những công trình kiến trúc mang giá trị văn hóa đặc sắc của cộng đồng Khmer tại khu vực Đồng bằng sông Cửu Long.  
Những hoa văn trang trí trên mái, cột, tường, cổng và phù điêu không chỉ mang tính thẩm mỹ mà còn thể hiện bản sắc dân tộc và yếu tố tín ngưỡng.

Tuy nhiên, việc nhận dạng và thống kê các hoa văn hiện nay chủ yếu được thực hiện thủ công, tốn nhiều thời gian và công sức.

Vì vậy, đề tài này xây dựng hệ thống nhận dạng hoa văn Chùa Khmer bằng mô hình YOLO kết hợp thuật toán hậu xử lý cải tiến nhằm nâng cao hiệu quả phát hiện đối tượng.

---

## Mục tiêu nghiên cứu

- Xây dựng bộ dữ liệu hình ảnh hoa văn Chùa Khmer Nam Bộ
- Gán nhãn dữ liệu theo chuẩn YOLO
- Huấn luyện mô hình phát hiện đối tượng
- So sánh hiệu năng nhiều phiên bản YOLO
- Đề xuất thuật toán hậu xử lý cải tiến
- Ứng dụng AI trong bảo tồn văn hóa truyền thống

---

## Công nghệ sử dụng

- Python
- YOLOv5m
- YOLOv8m
- YOLO26m
- OpenCV
- PyTorch
- Torchvision
- Ultralytics
- NumPy
- Roboflow
- Matplotlib

---

## Bộ dữ liệu

### Nguồn dữ liệu

Dữ liệu được thu thập trực tiếp tại các chùa Khmer tiêu biểu:

- Chùa Âng (Trà Vinh)
- Chùa Som Rong (Sóc Trăng)
- Chùa Xiêm Cán (Bạc Liêu)
- Chùa Pitu Khosa Rangsay (Cần Thơ)
- Và nhiều công trình khác
Ngoài ra còn có một số dữ liệu trên Internet
---

### Thống kê dữ liệu

| Thành phần | Số lượng |
|-----------|---------|
| Ảnh gốc | 432 |
| Sau tăng cường dữ liệu | 1789 |
| Bounding Box | 34,228 |

---

### Các lớp đối tượng

| Nhãn | Ý nghĩa |
|------|--------|
| la | Lá |
| hoa | Hoa |
| lua | Lửa |
| ran | Rắn |
| than | Thần |

---

## Cấu hình huấn luyện

| Tham số | Giá trị |
|--------|--------|
| Kích thước ảnh | 768 x 768 |
| Batch size | 8 |
| Epoch | 250 |
| Early stopping | 50 |
| GPU | RTX 4050 6GB |
| RAM | 16GB |

---

## Các mô hình thử nghiệm

- YOLOv5m
- YOLOv8m
- YOLO26m

---

## Kết quả thực nghiệm

| Mô hình | Precision | Recall | mAP@50 | mAP@50-95 |
|--------|----------|--------|-------|----------|
| YOLOv5m | 0.62 | 0.47 | 0.47 | 0.23 |
| YOLOv8m | 0.64 | 0.46 | 0.45 | 0.22 |
| YOLO26m | 0.60 | 0.47 | **0.49** | **0.24** |

### Mô hình tốt nhất: YOLO26m

YOLO26m cho hiệu năng tổng thể cao nhất trên tập dữ liệu thử nghiệm.

---

# Thuật toán cải tiến đề xuất

Ngoài mô hình YOLO gốc, đề tài đề xuất bước hậu xử lý giúp tăng độ chính xác phát hiện:

## YOLO + Spatial Confidence Boosting + Class-wise NMS

---

### 1️. Tăng độ tin cậy theo không gian

Các bounding box có confidence cao được chọn làm box tham chiếu (anchor).

Những box gần vị trí anchor và có kích thước tương đồng sẽ được tăng confidence.

```text
C' = min(Ci + αCa, 1)
Trong đó:

- `Ci`: độ tin cậy ban đầu của bounding box  
- `Ca`: độ tin cậy của anchor box  
- `α`: hệ số tăng cường  

Phương pháp này giúp các đối tượng nhỏ hoặc khó nhận dạng có thêm cơ hội được giữ lại.

---

### 2. Ngưỡng confidence riêng cho từng lớp

Mỗi lớp đối tượng có kích thước và đặc điểm khác nhau nên sử dụng ngưỡng confidence riêng:

- `la` → ngưỡng thấp hơn để giữ chi tiết nhỏ  
- Các lớp khác → ngưỡng chuẩn  

---

### 3. Non-Maximum Suppression theo từng lớp

Ngưỡng IoU:

- `la` → 0.8  
- Lớp khác → 0.6  

Giúp giảm box chồng lấp nhưng vẫn giữ được nhiều hoa văn nhỏ.

---

## ✅ Hiệu quả đạt được

- Giảm số lượng bounding box dư thừa  
- Giảm box trùng lặp  
- Tăng khả năng phát hiện đối tượng nhỏ  
- Giảm bỏ sót hoa văn  
- Kết quả trực quan rõ ràng hơn  
- Tăng độ tin cậy dự đoán  

---

## 📁 Cấu trúc project

```text
CT201e/
│── dataset/
│── models/
│── runs/
│── results/
│── train.py
│── detect.py
│── requirements.txt
│── README.md


