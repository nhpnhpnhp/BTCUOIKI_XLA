Bạn hãy sửa file `convolution_processing.py` trong project xử lý ảnh.

Mục tiêu:

* Chỉ dùng **một loại kernel duy nhất: Gaussian kernel**.
* Khai báo sẵn 3 kernel Gaussian hằng: `3x3`, `5x5`, `7x7`.
* Không dùng hàm tạo kernel trung bình trong pipeline chính.
* Không giới hạn kết quả tích chập về `[0,255]` trong bước tính toán.
* Không ép kết quả tính toán sang `uint8`.
* Chỉ khi lưu ảnh bằng `cv2.imwrite()` mới chuẩn hóa min-max về `[0,255]`.

## 1. Thêm 3 kernel Gaussian hằng ở đầu file

Sau phần import trong `convolution_processing.py`, thêm:

```python
# =============================================================================
# CÁC KERNEL GAUSSIAN HẰNG
# =============================================================================
# Đề bài chỉ quy định kích thước kernel 3x3, 5x5, 7x7,
# không quy định giá trị cụ thể bên trong kernel.
# Nhóm chọn Gaussian kernel vì đây là kernel lọc làm mịn ảnh phổ biến.
# Gaussian cho trọng số ở tâm lớn hơn, các điểm xa tâm có trọng số nhỏ hơn.

KERNEL_3X3_GAUSSIAN = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], dtype=np.float64) / 16.0

KERNEL_5X5_GAUSSIAN = np.array([
    [1,  4,  6,  4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1,  4,  6,  4, 1]
], dtype=np.float64) / 256.0

KERNEL_7X7_GAUSSIAN = np.array([
    [0, 0, 1, 2, 1, 0, 0],
    [0, 3, 13, 22, 13, 3, 0],
    [1, 13, 59, 97, 59, 13, 1],
    [2, 22, 97, 159, 97, 22, 2],
    [1, 13, 59, 97, 59, 13, 1],
    [0, 3, 13, 22, 13, 3, 0],
    [0, 0, 1, 2, 1, 0, 0]
], dtype=np.float64)

KERNEL_7X7_GAUSSIAN = KERNEL_7X7_GAUSSIAN / KERNEL_7X7_GAUSSIAN.sum()
```

## 2. Sửa hàm `xu_ly_tich_chap()`

Trong hàm `xu_ly_tich_chap()`, thay các dòng tạo kernel trung bình:

```python
kernel_3x3 = tao_kernel_trung_binh(3)
kernel_5x5 = tao_kernel_trung_binh(5)
kernel_7x7 = tao_kernel_trung_binh(7)
```

bằng:

```python
kernel_3x3 = KERNEL_3X3_GAUSSIAN
kernel_5x5 = KERNEL_5X5_GAUSSIAN
kernel_7x7 = KERNEL_7X7_GAUSSIAN
```

Có thể giữ lại hàm `tao_kernel_trung_binh()` nếu muốn, nhưng pipeline chính không được dùng hàm đó nữa.

## 3. Thêm hàm chuẩn hóa min-max để lưu ảnh

Thêm hàm sau vào file:

```python
def chuan_hoa_minmax_de_luu_anh(img_float):
    """
    Chuẩn hóa ảnh float về ảnh uint8 trong khoảng [0, 255] chỉ để lưu/hiển thị.

    Công thức:
        pixel_luu = (pixel - min) * 255 / (max - min)

    Lưu ý:
    - Không dùng hàm này trong bước tính toán chính.
    - Chỉ dùng trước khi gọi cv2.imwrite().
    """
    img = img_float.astype(np.float64)

    min_val = img.min()
    max_val = img.max()

    if max_val == min_val:
        return np.zeros_like(img, dtype=np.uint8)

    img_norm = (img - min_val) * 255.0 / (max_val - min_val)
    img_norm = np.clip(img_norm, 0, 255)

    return img_norm.astype(np.uint8)
```

## 4. Sửa hàm `tich_chap()`

Trong hàm `tich_chap(gray_img, kernel, padding=0, stride=1)`, nếu cuối hàm đang có:

```python
output = np.clip(output, 0, 255).astype(np.uint8)
return output
```

hãy thay bằng:

```python
return output
```

Yêu cầu:

* `output` giữ kiểu `np.float64`.
* Không giới hạn về `[0,255]`.
* Không ép `uint8`.

## 5. Sửa hàm `loc_trung_vi()`

Nếu cuối hàm `loc_trung_vi()` đang có:

```python
return np.clip(output, 0, 255).astype(np.uint8)
```

hãy thay bằng:

```python
return output
```

Yêu cầu:

* Lọc trung vị trả về float.
* Không clip.
* Không ép `uint8`.

## 6. Sửa hàm `padding_cho_cung_kich_thuoc()`

Nếu cuối hàm đang có:

```python
return img_padded[:h_mong_muon, :w_mong_muon].astype(np.uint8)
```

hãy đổi thành:

```python
return img_padded[:h_mong_muon, :w_mong_muon]
```

Yêu cầu:

* Không ép kiểu về `uint8`.
* Giữ nguyên kiểu dữ liệu của ảnh đầu vào.

## 7. Sửa hàm `tao_anh_I6(I4, I5)`

Nếu hàm đang có:

```python
I6 = I5.copy().astype(np.uint8)
I6[I4 > I5] = 0
return I6
```

hãy đổi thành:

```python
I6 = I5.copy()
I6[I4 > I5] = 0
return I6
```

Yêu cầu:

* I6 giữ dạng float.
* Không ép `uint8`.
* Quy tắc tạo I6 không đổi.

## 8. Sửa các lệnh `cv2.imwrite()` trong `xu_ly_tich_chap()`

Không lưu trực tiếp `I1`, `I2`, `I3`, `I4`, `I5`, `I6`.

Thay bằng dạng:

```python
cv2.imwrite(path, chuan_hoa_minmax_de_luu_anh(I1))
```

Áp dụng cụ thể:

```python
cv2.imwrite(os.path.join(thu_muc_output, f"{ten_anh}_I1_conv3x3.png"), chuan_hoa_minmax_de_luu_anh(I1))
cv2.imwrite(os.path.join(thu_muc_output, f"{ten_anh}_I2_conv5x5.png"), chuan_hoa_minmax_de_luu_anh(I2))
cv2.imwrite(os.path.join(thu_muc_output, f"{ten_anh}_I3_conv7x7_stride2.png"), chuan_hoa_minmax_de_luu_anh(I3))
cv2.imwrite(os.path.join(thu_muc_output, f"{ten_anh}_I4_median3x3.png"), chuan_hoa_minmax_de_luu_anh(I4))
cv2.imwrite(os.path.join(thu_muc_output, f"{ten_anh}_I5_median5x5.png"), chuan_hoa_minmax_de_luu_anh(I5))
cv2.imwrite(os.path.join(thu_muc_output, f"{ten_anh}_I6_threshold.png"), chuan_hoa_minmax_de_luu_anh(I6))
```

## 9. Dictionary trả về vẫn giữ dữ liệu float gốc

Ở cuối `xu_ly_tich_chap()`, các key sau vẫn phải giữ ảnh float gốc:

```python
'I1': I1,
'I2': I2,
'I3': I3,
'I4': I4,
'I4_padded': I4_padded,
'I5': I5,
'I6': I6,
```

Không được trả về ảnh đã chuẩn hóa để lưu.

## 10. Cập nhật comment/docstring

Cập nhật giải thích:

* Đề bài không quy định giá trị cụ thể của kernel.
* Nhóm chọn Gaussian kernel cho cả 3 kích thước.
* Gaussian kernel dùng để làm mịn ảnh, với trọng số trung tâm lớn hơn.
* Kết quả tính toán giữ dạng float để không mất thông tin.
* Chỉ chuẩn hóa min-max về `[0,255]` khi lưu ảnh.

## 11. Kiểm tra

Sau khi sửa:

* Chạy lại `main.py`.
* Đảm bảo ảnh kết quả vẫn được lưu.
* Đảm bảo không lỗi khi gọi `cv2.imwrite()`.
* In lại toàn bộ nội dung file `convolution_processing.py` đã cập nhật.
