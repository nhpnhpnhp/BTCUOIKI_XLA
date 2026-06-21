# Bài Tập Cuối Kỳ - Xử Lý Ảnh Số
## Nhóm 10

### Thành viên

| STT | MSSV     | Họ và Tên           |
|-----|----------|---------------------|
| 1   | 23133054 | Nguyễn Hồ Phát      |
| 2   | 23133061 | Phan Trọng Quí      |
| 3   | 23133056 | Phan Trọng Phú      |
| 4   | 23133035 | Trần Minh Khánh     |

---

## Cách Cài Đặt

### 1. Cài đặt Python
Yêu cầu Python 3.7 trở lên.

### 2. Cài đặt thư viện
```bash
pip install opencv-python numpy matplotlib
```

### 3. Chuẩn bị ảnh
Đặt 10 ảnh màu vào thư mục `input_images/`. Hỗ trợ định dạng: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`.

### 4. Chạy chương trình
```bash
cd image_processing_final
python main.py
```

### 5. Kết quả
- Ảnh kết quả lưu trong thư mục `output/`.
- Báo cáo PDF tại `output/report.pdf`.

---

## Cấu Trúc Project

```
image_processing_final/
│
├── input_images/           # Thư mục chứa 10 ảnh đầu vào
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
├── output/                 # Thư mục kết quả (tự tạo)
│   ├── histograms/         # Kết quả Nhiệm vụ 1
│   ├── convolution/        # Kết quả Nhiệm vụ 2
│   ├── lbp/                # Kết quả Nhiệm vụ 3
│   └── report.pdf          # Báo cáo PDF
│
├── main.py                 # Chương trình chính
├── histogram_processing.py # Module xử lý histogram
├── convolution_processing.py # Module tích chập & lọc trung vị
├── lbp_processing.py       # Module LBP
├── report_generator.py     # Module tạo báo cáo PDF
└── README.md               # File hướng dẫn (file này)
```

---

## Ý Nghĩa Từng File Python

### `main.py` — Chương trình chính
- Điều phối toàn bộ quy trình xử lý.
- Đọc ảnh từ thư mục `input_images/`.
- Gọi 3 module xử lý (histogram, tích chập, LBP).
- Gọi module tạo báo cáo PDF.
- Có kiểm tra lỗi: thư mục không tồn tại, ảnh không đọc được.

### `histogram_processing.py` — Xử lý histogram
- `chuyen_anh_xam()`: Chuyển ảnh màu BGR sang ảnh xám.
- `tinh_histogram()`: Đếm số pixel cho mỗi mức xám 0-255.
- `ve_histogram()`: Vẽ biểu đồ histogram và lưu file.
- `can_bang_histogram()`: Cân bằng histogram (histogram equalization).
- `thu_hep_muc_xam()`: Co giãn mức xám về khoảng [30, 120].

### `convolution_processing.py` — Tích chập và lọc trung vị
- `tich_chap()`: Phép tích chập thủ công với padding và stride.
- `tao_kernel_trung_binh()`: Tạo kernel trung bình (1/9, 1/25, 1/49).
- `loc_trung_vi()`: Lọc trung vị (median filter).
- `padding_cho_cung_kich_thuoc()`: Padding ảnh I4 cho cùng kích thước I5.
- `tao_anh_I6()`: Tạo ảnh ngưỡng I6 từ I4 và I5.

### `lbp_processing.py` — Local Binary Patterns
- `noi_suy_bilinear()`: Nội suy bilinear cho điểm không nguyên.
- `tinh_lbp_P8()`: Tính LBP với P=8 điểm (chuỗi 8 bit → thập phân).
- `tinh_lbp_P16()`: Tính LBP với P=16 (tách 2×8 bit, lấy max).
- `tinh_lbp_P24()`: Tính LBP với P=24 (tách 3×8 bit, lấy max).
- `tinh_lbp_toan_anh()`: Áp dụng LBP cho toàn bộ ảnh.

### `report_generator.py` — Tạo báo cáo PDF
- Dùng `matplotlib.backends.backend_pdf.PdfPages`.
- Tạo trang bìa, mô tả thuật toán, kết quả từng ảnh, nhận xét.

---

## Giải Thích Chi Tiết Các Hàm Chính

### 1. Chuyển ảnh màu sang ảnh xám

```python
def chuyen_anh_xam(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray
```

**Công thức**: `Gray = 0.299 × R + 0.587 × G + 0.114 × B`

- Mắt người nhạy cảm nhất với màu xanh lá (G = 0.587), ít nhạy với xanh dương (B = 0.114).
- OpenCV đọc ảnh theo thứ tự BGR (không phải RGB).
- Ảnh xám chỉ có 1 kênh (0-255), ảnh màu có 3 kênh (B, G, R).

### 2. Vẽ histogram

```python
def tinh_histogram(gray_img):
    hist = np.zeros(256, dtype=np.int64)
    for pixel in gray_img.flatten():
        hist[pixel] += 1
    return hist
```

- Tạo mảng 256 phần tử (tương ứng 256 mức xám).
- Duyệt từng pixel, tăng bộ đếm tại vị trí tương ứng.
- Kết quả: `hist[i]` = số pixel có mức xám `i`.

### 3. Cân bằng histogram (Histogram Equalization)

```python
def can_bang_histogram(gray_img):
    hist = tinh_histogram(gray_img)          # Bước 1: Tính histogram
    cdf = np.cumsum(hist)                     # Bước 2: Tính CDF
    cdf_norm = (cdf * 255) / total_pixels     # Bước 3: Chuẩn hóa
    result = cdf_norm[gray_img]               # Bước 4: Ánh xạ
    return result
```

**Các bước**:
1. Tính histogram: đếm số pixel cho mỗi mức xám.
2. Tính CDF (Cumulative Distribution Function): `CDF[i] = Σ hist[0..i]`.
3. Chuẩn hóa CDF: `CDF_norm[i] = round(CDF[i] × 255 / tổng_pixel)`.
4. Ánh xạ: `pixel_mới = CDF_norm[pixel_cũ]`.

**Mục đích**: Làm cho phân bố mức xám trải đều trên [0, 255], tăng độ tương phản.

### 4. Thu hẹp mức xám về [30, 120]

```python
def thu_hep_muc_xam(gray_img, new_min=30, new_max=120):
    result = (pixel - old_min) * (new_max - new_min) / (old_max - old_min) + new_min
    return result
```

**Công thức ánh xạ tuyến tính**:
```
pixel_mới = (pixel_cũ - min_cũ) × (120 - 30) / (max_cũ - min_cũ) + 30
```

- Pixel nhỏ nhất → 30, pixel lớn nhất → 120.
- Tất cả pixel được co giãn tuyến tính vào khoảng [30, 120].

### 5. Tích chập với padding và stride

```python
def tich_chap(gray_img, kernel, padding=0, stride=1):
    # Thêm viền 0 xung quanh ảnh
    anh_padding = np.pad(gray_img, padding, constant_values=0)
    
    # Trượt kernel qua ảnh
    for i in range(output_h):
        for j in range(output_w):
            vung = anh_padding[i*stride : i*stride+k_h, j*stride : j*stride+k_w]
            output[i, j] = sum(vung * kernel)
```

**Ví dụ**: Kernel trung bình 3×3:
```
[1/9  1/9  1/9]
[1/9  1/9  1/9]
[1/9  1/9  1/9]
```
- Tại mỗi pixel, lấy 9 pixel lân cận, nhân với kernel, cộng tổng.
- Kết quả = trung bình 9 pixel → làm mờ ảnh.

### 6. Lọc trung vị

```python
def loc_trung_vi(gray_img, kich_thuoc):
    for mỗi pixel:
        vung = lấy vùng lân cận kich_thuoc × kich_thuoc
        output[pixel] = median(vung)  # Giá trị ở giữa khi sắp xếp
```

**Ví dụ**: Cửa sổ 3×3, các giá trị: `[10, 20, 200, 15, 18, 12, 14, 16, 13]`
- Sắp xếp: `[10, 12, 13, 14, 15, 16, 18, 20, 200]`
- Trung vị = `15` (vị trí thứ 5 trong 9 giá trị)
- Giá trị 200 (nhiễu) bị loại bỏ!

### 7. Tạo ảnh I6

```python
def tao_anh_I6(I4, I5):
    I6 = I5.copy()
    I6[I4 > I5] = 0  # Đặt = 0 nơi I4 > I5
```

- I4 = lọc trung vị(I3, 3×3) — từ tích chập stride=2.
- I5 = lọc trung vị(I1, 5×5) — từ tích chập stride=1.
- I6: giữ giá trị I5 ở nơi I5 ≥ I4, đặt = 0 ở nơi I4 > I5.

### 8. Tính LBP (P=8, 16, 24)

**P = 8**:
```python
# 8 điểm trên đường tròn bán kính R
for p in range(8):
    x_p = hang - R * cos(2π * p / 8)
    y_p = cot + R * sin(2π * p / 8)
    bit = 1 nếu img[x_p, y_p] >= img[hang, cot]
    chuoi += str(bit)
lbp = int(chuoi, 2)  # "10110010" → 178
```

**P = 16**: Tách 16 bit → 2 × 8 bit → max(phần_1, phần_2)

**P = 24**: Tách 24 bit → 3 × 8 bit → max(phần_1, phần_2, phần_3)

---

## Câu Hỏi Vấn Đáp Thường Gặp

### 1. Histogram là gì?
**Trả lời**: Histogram là biểu đồ thống kê phân bố mức xám (hoặc màu) của ảnh. Trục X là giá trị mức xám (0-255), trục Y là số lượng pixel có mức xám đó. Histogram cho biết ảnh sáng hay tối, tương phản cao hay thấp.

- Ảnh tối: histogram lệch trái (nhiều pixel mức xám thấp).
- Ảnh sáng: histogram lệch phải (nhiều pixel mức xám cao).
- Ảnh tương phản thấp: histogram co cụm ở một vùng hẹp.

### 2. Cân bằng histogram để làm gì?
**Trả lời**: Cân bằng histogram (Histogram Equalization) làm cho phân bố mức xám trải đều trên toàn bộ dải [0, 255]. Mục đích:

- **Tăng độ tương phản**: Ảnh ban đầu có thể bị mờ, thiếu tương phản. Sau cân bằng, ảnh rõ nét hơn.
- **Nguyên lý**: Dùng hàm phân phối tích lũy (CDF) để ánh xạ mức xám cũ sang mức xám mới. CDF đảm bảo các mức xám được phân bố đều.
- **Ứng dụng**: Cải thiện ảnh y tế, ảnh vệ tinh, ảnh thiếu sáng.

### 3. Padding là gì?
**Trả lời**: Padding là thêm viền (thường là giá trị 0) xung quanh ảnh trước khi thực hiện phép tích chập.

- **Lý do**: Khi trượt kernel qua ảnh, pixel ở biên không có đủ lân cận. Padding giải quyết vấn đề này.
- **Padding = 1**: Thêm 1 hàng/cột giá trị 0 xung quanh → ảnh từ (h, w) thành (h+2, w+2).
- **Same padding**: Chọn padding sao cho ảnh đầu ra cùng kích thước đầu vào. Ví dụ: kernel 3×3 → padding = 1, kernel 5×5 → padding = 2.

### 4. Stride là gì?
**Trả lời**: Stride là bước nhảy khi trượt kernel qua ảnh trong phép tích chập.

- **Stride = 1**: Trượt từng pixel một → ảnh đầu ra cùng kích thước (nếu có padding phù hợp).
- **Stride = 2**: Nhảy 2 pixel mỗi lần → ảnh đầu ra nhỏ hơn (khoảng một nửa).
- **Công thức**: `output_size = (input + 2×padding - kernel) / stride + 1`
- **Ứng dụng**: Giảm kích thước ảnh, downsampling trong mạng CNN.

### 5. Kernel là gì?
**Trả lời**: Kernel (hay filter/nhân tích chập) là ma trận nhỏ (thường 3×3, 5×5, 7×7) dùng để trượt qua ảnh trong phép tích chập.

- **Kernel trung bình**: Tất cả phần tử bằng nhau (1/n²) → làm mờ ảnh, giảm nhiễu.
- **Kernel Gaussian**: Phần tử ở giữa lớn, ở biên nhỏ → làm mờ tự nhiên hơn.
- **Kernel Sobel**: Phát hiện biên cạnh (edge detection).
- **Kernel Laplacian**: Phát hiện biên theo mọi hướng.

### 6. Tích chập hoạt động như thế nào?
**Trả lời**: Phép tích chập (convolution) hoạt động theo 3 bước:

1. **Đặt kernel**: Đặt kernel tại một vị trí trên ảnh.
2. **Nhân và cộng**: Nhân từng phần tử kernel với pixel ảnh tương ứng, rồi cộng tổng tất cả.
3. **Di chuyển**: Trượt kernel sang vị trí tiếp theo (theo stride) và lặp lại.

**Ví dụ** (kernel 3×3 trung bình):
```
Ảnh:     Kernel:        Tích:
10 20 30   1/9 1/9 1/9    10/9 + 20/9 + 30/9
40 50 60 × 1/9 1/9 1/9  = 40/9 + 50/9 + 60/9
70 80 90   1/9 1/9 1/9    70/9 + 80/9 + 90/9
                         = 450/9 = 50 (trung bình)
```

### 7. Lọc trung vị dùng để làm gì?
**Trả lời**: Lọc trung vị (Median Filter) dùng để **loại bỏ nhiễu** trong ảnh, đặc biệt là nhiễu muối tiêu (salt-and-pepper noise).

- **Cách hoạt động**: Sắp xếp các pixel lân cận, lấy giá trị ở giữa (trung vị).
- **Ưu điểm so với lọc trung bình**: Giữ được biên (edge) tốt hơn, vì trung vị không bị ảnh hưởng bởi giá trị cực đoan (outlier).
- **Nhược điểm**: Chậm hơn lọc trung bình (do phải sắp xếp), có thể làm mất chi tiết nhỏ.

### 8. LBP là gì?
**Trả lời**: LBP (Local Binary Pattern) là một toán tử mô tả **texture cục bộ** của ảnh.

- **Ý tưởng**: Tại mỗi pixel, so sánh với P điểm lân cận trên đường tròn bán kính R.
- **Kết quả**: Chuỗi nhị phân P bit → chuyển sang số thập phân → giá trị LBP.
- **Đặc điểm**: Bất biến với thay đổi độ sáng (chỉ so sánh tương đối), tính toán nhanh.
- **Ứng dụng**: Nhận dạng khuôn mặt, phân loại texture, phát hiện vật thể.

### 9. Vì sao P=16 phải tách thành 2 chuỗi 8 bit?
**Trả lời**: Vì ảnh xám chỉ có 256 mức xám (0-255), tương đương 8 bit.

- P=16 tạo chuỗi nhị phân 16 bit → 2^16 = **65,536** giá trị khác nhau.
- Giá trị 65,536 **vượt quá 255** → không thể hiển thị trên ảnh xám 8-bit.
- **Giải pháp**: Tách 16 bit thành 2 phần, mỗi phần 8 bit (0-255).
- Chuyển mỗi phần sang thập phân, lấy **max** → giữ pattern nổi bật nhất.
- Kết quả nằm trong [0, 255] → hiển thị được bình thường.

### 10. Vì sao P=24 phải tách thành 3 chuỗi 8 bit?
**Trả lời**: Tương tự lý do P=16, nhưng quy mô lớn hơn:

- P=24 tạo chuỗi 24 bit → 2^24 = **16,777,216** giá trị.
- 24 / 8 = 3 → tách thành **3 phần**, mỗi phần 8 bit (0-255).
- Lấy **max** trong 3 giá trị → giữ pattern đặc trưng nhất.
- Cách tách này đảm bảo kết quả vừa với ảnh 8-bit mà vẫn giữ thông tin.

### 11. Cách tạo ảnh I6 hoạt động như thế nào?
**Trả lời**: Ảnh I6 được tạo bằng phép so sánh pixel-by-pixel giữa I4 và I5:

```
Nếu I4(x,y) > I5(x,y) → I6(x,y) = 0 (đen)
Ngược lại             → I6(x,y) = I5(x,y) (giữ nguyên)
```

- **I4** = lọc trung vị 3×3 của I3 (tích chập 7×7, stride=2) — phản ánh đặc trưng vùng rộng, ảnh nhỏ hơn.
- **I5** = lọc trung vị 5×5 của I1 (tích chập 3×3) — phản ánh chi tiết cục bộ.
- **I6** giữ pixel I5 ở nơi I5 ≥ I4, tức là nơi chi tiết cục bộ (I5) có cường độ tương đương hoặc lớn hơn vùng rộng (I4).
- Vùng đen (I4 > I5) cho thấy pixel đó thuộc vùng tối cục bộ hoặc vùng biên yếu.
- Ứng dụng: phân tách vùng ảnh dựa trên so sánh multi-scale.

---

## Lưu Ý

1. **LBP chạy chậm** vì tính toán thủ công từng pixel. Với ảnh lớn, mỗi cấu hình có thể mất vài phút.
2. **Ảnh đầu vào** nên có kích thước vừa phải (VD: 500×500) để chạy nhanh hơn.
3. **Nếu muốn chạy nhanh hơn**, có thể resize ảnh trước khi xử lý LBP.

---

## Ôn Vấn Đáp Ngắn Gọn

### Histogram là gì?
Histogram là biểu đồ đếm số pixel theo từng mức xám từ 0 đến 255.

### Cân bằng histogram dùng để làm gì?
Cân bằng histogram dùng để tăng độ tương phản bằng cách phân bố lại mức xám rộng hơn trên khoảng 0 đến 255.

### Vì sao cần thu hẹp mức xám về `[30, 120]`?
Thu hẹp mức xám về `[30, 120]` giúp kiểm soát dải sáng đầu ra và quan sát ảnh khi độ tương phản bị nén lại.

### Padding là gì?
Padding là thêm viền quanh ảnh, thường là giá trị 0, để kernel xử lý được vùng biên.

### Stride là gì?
Stride là bước nhảy của kernel khi trượt trên ảnh; stride lớn hơn làm ảnh đầu ra nhỏ hơn.

### Kernel là gì?
Kernel là ma trận nhỏ dùng để nhân với vùng ảnh lân cận trong phép convolution.

### Convolution hoạt động như thế nào?
Convolution đặt kernel lên từng vùng ảnh, nhân từng phần tử tương ứng rồi cộng lại để tạo pixel mới.

### I1, I2, I3 khác nhau ở đâu?
I1 dùng kernel trung bình 3x3, padding 1, stride 1; I2 dùng 5x5, padding 2, stride 1; I3 dùng 7x7, padding 3, stride 2 nên nhỏ hơn.

### Median filter là gì?
Median filter lấy trung vị của các pixel trong cửa sổ lân cận để giảm nhiễu.

### I4 và I5 được tạo như thế nào?
I4 là ảnh I3 sau lọc trung vị 3x3; I5 là ảnh I1 sau lọc trung vị 5x5.

### Vì sao I4 và I5 có thể khác kích thước?
I4 lấy từ I3, mà I3 dùng stride 2 nên kích thước giảm; I5 lấy từ I1 dùng stride 1 nên giữ kích thước gần ảnh gốc.

### I6 được tạo theo quy tắc nào?
Nếu `I4(x, y) > I5(x, y)` thì `I6(x, y) = 0`, ngược lại `I6(x, y) = I5(x, y)`.

### LBP là gì?
LBP là phương pháp mô tả texture bằng cách so sánh pixel trung tâm với các pixel lân cận và tạo chuỗi nhị phân.

### P và R trong LBP là gì?
P là số điểm lân cận được lấy trên đường tròn, R là bán kính đường tròn quanh pixel trung tâm.

### Vì sao P=16 phải tách thành 2 nhóm 8 bit?
Vì 16 bit có thể vượt quá 255, nên tách thành 2 nhóm 8 bit rồi lấy giá trị lớn nhất để lưu được trên ảnh xám 8-bit.

### Vì sao P=24 phải tách thành 3 nhóm 8 bit?
Vì 24 bit tạo giá trị rất lớn, nên tách thành 3 nhóm 8 bit rồi lấy giá trị lớn nhất để kết quả vẫn nằm trong 0 đến 255.

### Bilinear interpolation trong LBP dùng để làm gì?
Bilinear interpolation dùng để ước lượng giá trị pixel khi điểm lân cận nằm ở tọa độ không nguyên.

### Pipeline xử lý một ảnh diễn ra như thế nào?
Chương trình đọc ảnh, chuyển sang xám, xử lý histogram, tạo I1 đến I6, tính 5 ảnh LBP, lưu kết quả và đưa vào PDF.

### Cách chạy với Conda trên máy này
Nếu lệnh `python` trỏ tới Windows Store alias, có thể chạy bằng Conda:

```bash
C:\Users\Admin\miniconda3\Scripts\conda.exe run -n base python main.py
```
