import numpy as np
import cv2
import os
from histogram_processing import chuyen_anh_xam


# =============================================================================
# CÁC KERNEL GAUSSIAN HẰNG
# =============================================================================
# Đề bài chỉ quy định kích thước kernel 3x3, 5x5, 7x7,
# không quy định giá trị cụ thể bên trong kernel.
# Nhóm chọn Gaussian kernel vì đây là kernel lọc làm mịn ảnh phổ biến.
# Gaussian cho trọng số ở tâm lớn hơn, các điểm xa tâm có trọng số nhỏ hơn.
#??? tại sao chọn kernel như này
KERNEL_3X3_GAUSSIAN = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], dtype=np.float64)

KERNEL_5X5_GAUSSIAN = np.array([
    [1,  4,  6,  4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1,  4,  6,  4, 1]
], dtype=np.float64)

KERNEL_7X7_GAUSSIAN = np.array([
    [0, 0, 1, 2, 1, 0, 0],
    [0, 3, 13, 22, 13, 3, 0],
    [1, 13, 59, 97, 59, 13, 1],
    [2, 22, 97, 159, 97, 22, 2],
    [1, 13, 59, 97, 59, 13, 1],
    [0, 3, 13, 22, 13, 3, 0],
    [0, 0, 1, 2, 1, 0, 0]
], dtype=np.float64)

# =============================================================================
# HÀM 1: TÍCH CHẬP (CONVOLUTION) VỚI PADDING VÀ STRIDE
# =============================================================================
def tich_chap(gray_img, kernel, padding=0, stride=1):

    # Lấy kích thước ảnh và kernel
    chieu_cao, chieu_rong = gray_img.shape
    k_h, k_w = kernel.shape  # kích thước kernel
    
    # ---- Bước 1: Thêm padding (viền 0) xung quanh ảnh ----
    if padding > 0:
        # np.pad thêm viền giá trị 0 xung quanh ảnh
        anh_padding = np.pad(
            gray_img.astype(np.float64),
            pad_width=padding,
            mode='constant',
            constant_values=0
        )
    else:
        anh_padding = gray_img.astype(np.float64)
    
    # Kích thước ảnh sau padding
    h_pad, w_pad = anh_padding.shape
    
    # ---- Bước 2: Tính kích thước ảnh đầu ra ----
    output_h = (h_pad - k_h) // stride + 1
    output_w = (w_pad - k_w) // stride + 1
    
    # Tạo ảnh đầu ra
    output = np.zeros((output_h, output_w), dtype=np.float64)
    
    # ---- Bước 3: Trượt kernel qua ảnh ----
    for i in range(output_h):
        for j in range(output_w):
            # Vị trí bắt đầu trên ảnh padding
            row_start = i * stride
            col_start = j * stride
            
            # Cắt vùng ảnh tương ứng với kernel
            vung_anh = anh_padding[row_start:row_start + k_h, col_start:col_start + k_w]
            
            # Nhân từng phần tử và lấy tổng
            output[i, j] = np.sum(vung_anh * kernel)
    
    return output


# =============================================================================
# HÀM 3: CHUẨN HÓA MIN-MAX ĐỂ LƯU ẢNH
# =============================================================================
def chuan_hoa_minmax_de_luu_anh(img_float):

    img = img_float.astype(np.float64)

    min_val = img.min()
    max_val = img.max()

    if max_val == min_val:
        return np.zeros_like(img, dtype=np.uint8)

    img_norm = (img - min_val) * 255.0 / (max_val - min_val)
    img_norm = np.clip(img_norm, 0, 255)

    return img_norm.astype(np.uint8)


# =============================================================================
# HÀM 4: LỌC TRUNG VỊ (MEDIAN FILTER)
# =============================================================================
def loc_trung_vi(gray_img, kich_thuoc_cua_so):

    chieu_cao, chieu_rong = gray_img.shape
    
    # Tính bán kính padding (nửa kích thước cửa sổ)
    h_out = chieu_cao - kich_thuoc_cua_so + 1
    w_out = chieu_rong - kich_thuoc_cua_so + 1
    
    # Tạo ảnh đầu ra
    output = np.zeros((h_out, w_out), dtype=np.float64)
    
    # Duyệt qua từng pixel
    for i in range(h_out):
        for j in range(w_out):
            # Lấy vùng lân cận xung quanh pixel (i, j)
            vung_lan_can = gray_img[
                i:i + kich_thuoc_cua_so,
                j:j + kich_thuoc_cua_so
            ]
            
            # Sắp xếp các giá trị và lấy giá trị trung vị (ở giữa)
            output[i, j] = np.median(vung_lan_can)
    
    return output


# =============================================================================
# HÀM 5: PADDING ẢNH ĐỂ CÁC ẢNH CÓ CÙNG KÍCH THƯỚC
# =============================================================================
def padding_cho_cung_kich_thuoc(img_nho, kich_thuoc_mong_muon):

    h_mong_muon, w_mong_muon = kich_thuoc_mong_muon
    h_hien_tai, w_hien_tai = img_nho.shape
    
    # Tính số hàng/cột cần thêm
    pad_h = max(0, h_mong_muon - h_hien_tai)
    pad_w = max(0, w_mong_muon - w_hien_tai)
    
    # Thêm padding ở phía dưới và bên phải
    # (pad_top, pad_bottom), (pad_left, pad_right)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    img_padded = np.pad(
        img_nho,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode='constant',
        constant_values=0
    )
    
    return img_padded


# =============================================================================
# HÀM 6: TẠO ẢNH I6 (ẢNH NGƯỠNG)
# =============================================================================
def tao_anh_I6(I4, I5):

    # Tạo ảnh I6 ban đầu = bản sao của I5
    I6 = I5.copy()
    
    # Tại vị trí nào I4 > I5, gán I6 = 0
    I6[I4 > I5] = 0
    
    return I6


# =============================================================================
# HÀM 7: XỬ LÝ TÍCH CHẬP CHO MỘT ẢNH (GỌI TẤT CẢ CÁC BƯỚC)
# =============================================================================
def xu_ly_tich_chap(img, ten_anh, thu_muc_output):

    os.makedirs(thu_muc_output, exist_ok=True)
    
    # Bước 1: Chuyển sang ảnh xám
    anh_xam = chuyen_anh_xam(img)
    
    # Đề bài không quy định giá trị cụ thể của kernel; nhóm chọn Gaussian
    # cho cả 3 kích thước để làm mịn ảnh với trọng số trung tâm lớn hơn.
    kernel_3x3 = KERNEL_3X3_GAUSSIAN
    kernel_5x5 = KERNEL_5X5_GAUSSIAN
    kernel_7x7 = KERNEL_7X7_GAUSSIAN
    
    # ---- Bước 2: Tích chập Gaussian 3x3, padding=1 → I1 ----
    I1 = tich_chap(anh_xam, kernel_3x3, padding=1, stride=1)
    cv2.imwrite(os.path.join(thu_muc_output, f"{ten_anh}_I1_conv3x3.png"), chuan_hoa_minmax_de_luu_anh(I1))
    print(f"  [Convolution] I1 (3x3): kich thuoc = {I1.shape}")
    
    # ---- Bước 3: Tích chập Gaussian 5x5, padding=2 → I2 ----
    I2 = tich_chap(anh_xam, kernel_5x5, padding=2, stride=1)
    cv2.imwrite(os.path.join(thu_muc_output, f"{ten_anh}_I2_conv5x5.png"), chuan_hoa_minmax_de_luu_anh(I2))
    print(f"  [Convolution] I2 (5x5): kich thuoc = {I2.shape}")
    
    # ---- Bước 4: Tích chập Gaussian 7x7, padding=3, stride=2 → I3 ----
    I3 = tich_chap(anh_xam, kernel_7x7, padding=3, stride=2)
    cv2.imwrite(os.path.join(thu_muc_output, f"{ten_anh}_I3_conv7x7_stride2.png"), chuan_hoa_minmax_de_luu_anh(I3))
    print(f"  [Convolution] I3 (7x7, stride=2): kich thuoc = {I3.shape}")
    
    # ---- Bước 5: Lọc trung vị I3 với 3x3 → I4 ----
    I4 = loc_trung_vi(I3, 3)
    cv2.imwrite(os.path.join(thu_muc_output, f"{ten_anh}_I4_median3x3.png"), chuan_hoa_minmax_de_luu_anh(I4))
    print(f"  [Convolution] I4 (median 3x3 cua I3): kich thuoc = {I4.shape}")
    
    # ---- Bước 6: Lọc trung vị I1 với 5x5 → I5 ----
    I5 = loc_trung_vi(I1, 5)
    cv2.imwrite(os.path.join(thu_muc_output, f"{ten_anh}_I5_median5x5.png"), chuan_hoa_minmax_de_luu_anh(I5))
    print(f"  [Convolution] I5 (median 5x5 cua I1): kich thuoc = {I5.shape}")
    
    # ---- Bước 7: Padding I4 nếu kích thước khác I5 ----
    I4_padded = padding_cho_cung_kich_thuoc(I4, I5.shape)
    print(f"  [Convolution] I4 sau padding: {I4_padded.shape}")

    
    # ---- Bước 8: Tạo ảnh I6 ----
    I6 = tao_anh_I6(I4_padded, I5)
    cv2.imwrite(os.path.join(thu_muc_output, f"{ten_anh}_I6_threshold.png"), chuan_hoa_minmax_de_luu_anh(I6))
    print(f"  [Convolution] I6 (anh nguong): kich thuoc = {I6.shape}")
