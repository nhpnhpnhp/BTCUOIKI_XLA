"""
=============================================================================
Module: convolution_processing.py
Mô tả: Xử lý tích chập và lọc trung vị
    - Tích chập với kernel trung bình (3x3, 5x5, 7x7)
    - Hỗ trợ padding và stride
    - Lọc trung vị (Median Filter)
    - Tạo ảnh ngưỡng I6

Nhóm 10 - Bài tập cuối kỳ Xử lý ảnh
=============================================================================
"""

import numpy as np
import cv2
import os
from histogram_processing import chuyen_anh_xam


# =============================================================================
# HÀM 1: TÍCH CHẬP (CONVOLUTION) VỚI PADDING VÀ STRIDE
# =============================================================================
def tich_chap(gray_img, kernel, padding=0, stride=1):
    """
    Thực hiện phép tích chập (convolution) trên ảnh xám.
    
    Giải thích phép tích chập:
        - Trượt một cửa sổ (kernel) qua từng vị trí trên ảnh.
        - Tại mỗi vị trí, nhân từng phần tử của kernel với pixel tương ứng.
        - Tổng các tích đó = giá trị pixel mới tại vị trí trung tâm.
    
    Giải thích padding:
        - Padding là thêm viền (thường là 0) xung quanh ảnh.
        - Mục đích: giữ nguyên kích thước ảnh sau tích chập.
        - Ví dụ: padding = 1 → thêm 1 hàng/cột giá trị 0 xung quanh ảnh.
    
    Giải thích stride:
        - Stride là bước nhảy khi trượt kernel.
        - Stride = 1: trượt từng pixel một (mặc định).
        - Stride = 2: nhảy 2 pixel mỗi lần → ảnh đầu ra nhỏ hơn.
    
    Công thức kích thước đầu ra:
        output_h = (input_h + 2*padding - kernel_h) // stride + 1
        output_w = (input_w + 2*padding - kernel_w) // stride + 1
    
    Tham số:
        gray_img: ảnh xám đầu vào (numpy array 2D).
        kernel: ma trận kernel (numpy array 2D), ví dụ: 3x3, 5x5, 7x7.
        padding: số hàng/cột thêm vào viền (int, mặc định 0).
        stride: bước nhảy khi trượt kernel (int, mặc định 1).
    
    Trả về:
        output: ảnh sau tích chập (numpy array 2D), kiểu uint8.
    """
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
    
    # Giới hạn giá trị về [0, 255] và chuyển sang uint8
    output = np.clip(output, 0, 255).astype(np.uint8)
    
    return output


# =============================================================================
# HÀM 2: TẠO KERNEL TRUNG BÌNH
# =============================================================================
def tao_kernel_trung_binh(kich_thuoc):
    """
    Tạo kernel trung bình (mean/average kernel).
    
    Giải thích:
        - Kernel trung bình có tất cả phần tử bằng nhau.
        - Giá trị mỗi phần tử = 1 / (tổng số phần tử).
        - Ví dụ: kernel 3x3 → mỗi phần tử = 1/9.
        - Tác dụng: làm mờ (blur) ảnh, giảm nhiễu.
    
    Tham số:
        kich_thuoc: kích thước kernel (int), ví dụ: 3, 5, 7.
    
    Trả về:
        kernel: ma trận kernel trung bình (numpy array 2D).
    """
    # Tạo ma trận toàn 1, rồi chia cho tổng số phần tử
    kernel = np.ones((kich_thuoc, kich_thuoc), dtype=np.float64)
    kernel = kernel / (kich_thuoc * kich_thuoc)
    return kernel


# =============================================================================
# HÀM 3: LỌC TRUNG VỊ (MEDIAN FILTER)
# =============================================================================
def loc_trung_vi(gray_img, kich_thuoc_cua_so):
    """
    Lọc trung vị (Median Filter) cho ảnh xám.
    
    Giải thích:
        - Tại mỗi pixel, lấy tất cả các pixel lân cận trong cửa sổ.
        - Sắp xếp các giá trị pixel theo thứ tự tăng dần.
        - Lấy giá trị ở giữa (trung vị) làm giá trị pixel mới.
        - Tác dụng: loại bỏ nhiễu muối tiêu (salt-and-pepper noise)
          mà vẫn giữ được biên (edge) tốt hơn lọc trung bình.
    
    Ví dụ với cửa sổ 3x3:
        Các giá trị pixel lân cận: [10, 20, 15, 200, 18, 12, 14, 16, 13]
        Sắp xếp: [10, 12, 13, 14, 15, 16, 18, 20, 200]
        Trung vị: 15 (giá trị ở vị trí giữa)
        → Giá trị 200 (nhiễu) bị loại bỏ!
    
    Tham số:
        gray_img: ảnh xám đầu vào (numpy array 2D).
        kich_thuoc_cua_so: kích thước cửa sổ lọc (int), phải là số lẻ.
    
    Trả về:
        output: ảnh sau lọc trung vị (numpy array 2D), kiểu uint8.
    """
    chieu_cao, chieu_rong = gray_img.shape
    
    # Tính bán kính padding (nửa kích thước cửa sổ)
    ban_kinh = kich_thuoc_cua_so // 2
    
    # Thêm padding để xử lý pixel ở biên ảnh
    anh_padding = np.pad(
        gray_img.astype(np.float64),
        pad_width=ban_kinh,
        mode='constant',
        constant_values=0
    )
    
    # Tạo ảnh đầu ra
    output = np.zeros((chieu_cao, chieu_rong), dtype=np.float64)
    
    # Duyệt qua từng pixel
    for i in range(chieu_cao):
        for j in range(chieu_rong):
            # Lấy vùng lân cận xung quanh pixel (i, j)
            vung_lan_can = anh_padding[
                i:i + kich_thuoc_cua_so,
                j:j + kich_thuoc_cua_so
            ]
            
            # Sắp xếp các giá trị và lấy giá trị trung vị (ở giữa)
            output[i, j] = np.median(vung_lan_can)
    
    return np.clip(output, 0, 255).astype(np.uint8)


# =============================================================================
# HÀM 4: PADDING ẢNH ĐỂ CÁC ẢNH CÓ CÙNG KÍCH THƯỚC
# =============================================================================
def padding_cho_cung_kich_thuoc(img_nho, kich_thuoc_mong_muon):
    """
    Thêm padding (viền 0) vào ảnh nhỏ để đạt kích thước mong muốn.
    
    Giải thích:
        - Khi I3 (tích chập với stride=2) có kích thước nhỏ hơn I1,
          ta cần thêm viền 0 xung quanh I4 (lọc trung vị từ I3) 
          để I4 cùng kích thước với I5 (lọc trung vị từ I1).
        - Ưu tiên padding (thêm viền 0) thay vì resize, 
          để giữ nguyên bản chất kết quả xử lý.
    
    Tham số:
        img_nho: ảnh có kích thước nhỏ hơn (numpy array 2D).
        kich_thuoc_mong_muon: tuple (h, w) - kích thước cần đạt.
    
    Trả về:
        img_padded: ảnh sau padding (numpy array 2D), kiểu uint8.
    """
    h_mong_muon, w_mong_muon = kich_thuoc_mong_muon
    h_hien_tai, w_hien_tai = img_nho.shape
    
    # Nếu ảnh đã đủ kích thước, trả về luôn
    if h_hien_tai >= h_mong_muon and w_hien_tai >= w_mong_muon:
        return img_nho[:h_mong_muon, :w_mong_muon].copy()
    
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
    
    # Cắt nếu vượt quá kích thước mong muốn
    return img_padded[:h_mong_muon, :w_mong_muon].astype(np.uint8)


# =============================================================================
# HÀM 5: TẠO ẢNH I6 (ẢNH NGƯỠNG)
# =============================================================================
def tao_anh_I6(I4, I5):
    """
    Tạo ảnh I6 theo công thức ngưỡng:
        Nếu I4(x, y) > I5(x, y) thì I6(x, y) = 0 (đen)
        Ngược lại: I6(x, y) = I5(x, y)
    
    Giải thích:
        - Đây là phép so sánh pixel-by-pixel giữa hai ảnh.
        - Tại vị trí (x, y), nếu giá trị ảnh I4 lớn hơn I5,
          pixel tương ứng trên I6 sẽ bằng 0 (đen).
        - Ngược lại, giữ nguyên giá trị từ I5.
        - Kết quả I6 là ảnh kết hợp, làm nổi bật những vùng
          mà I5 có giá trị lớn hơn hoặc bằng I4.
    
    Lưu ý:
        I4 và I5 phải có cùng kích thước. 
        Nếu khác kích thước, cần padding trước khi gọi hàm này.
    
    Tham số:
        I4: ảnh I4 (numpy array 2D), kiểu uint8.
        I5: ảnh I5 (numpy array 2D), kiểu uint8.
    
    Trả về:
        I6: ảnh kết quả (numpy array 2D), kiểu uint8.
    """
    # Tạo ảnh I6 ban đầu = bản sao của I5
    I6 = I5.copy().astype(np.uint8)
    
    # Tại vị trí nào I4 > I5, gán I6 = 0
    I6[I4 > I5] = 0
    
    return I6


# =============================================================================
# HÀM 6: XỬ LÝ TÍCH CHẬP CHO MỘT ẢNH (GỌI TẤT CẢ CÁC BƯỚC)
# =============================================================================
def xu_ly_tich_chap(img, ten_anh, thu_muc_output):
    """
    Thực hiện toàn bộ quy trình tích chập và lọc trung vị cho một ảnh.
    
    Các bước:
        1. Chuyển ảnh sang xám.
        2. Tích chập 3x3, padding=1 → I1.
        3. Tích chập 5x5, padding=2 → I2.
        4. Tích chập 7x7, padding=3, stride=2 → I3.
        5. Lọc trung vị I3 với 3x3 → I4.
        6. Lọc trung vị I1 với 5x5 → I5.
        7. Padding I4 cho cùng kích thước I5 (nếu cần).
        8. Tạo ảnh I6 = so sánh I4 và I5.
    
    Tham số:
        img: ảnh màu đầu vào (numpy array).
        ten_anh: tên file ảnh gốc (string).
        thu_muc_output: đường dẫn thư mục lưu kết quả.
    
    Trả về:
        dict chứa các ảnh kết quả.
    """
    os.makedirs(thu_muc_output, exist_ok=True)
    
    # Bước 1: Chuyển sang ảnh xám
    anh_xam = chuyen_anh_xam(img)
    
    # ---- Bước 2: Tích chập 3x3, padding=1 → I1 ----
    print(f"  [Convolution] Dang tich chap 3x3...")
    kernel_3x3 = tao_kernel_trung_binh(3)  # Kernel 3x3, mỗi phần tử = 1/9
    I1 = tich_chap(anh_xam, kernel_3x3, padding=1, stride=1)
    cv2.imwrite(os.path.join(thu_muc_output, f"{ten_anh}_I1_conv3x3.png"), I1)
    print(f"  [Convolution] I1 (3x3): kich thuoc = {I1.shape}")
    
    # ---- Bước 3: Tích chập 5x5, padding=2 → I2 ----
    print(f"  [Convolution] Dang tich chap 5x5...")
    kernel_5x5 = tao_kernel_trung_binh(5)  # Kernel 5x5, mỗi phần tử = 1/25
    I2 = tich_chap(anh_xam, kernel_5x5, padding=2, stride=1)
    cv2.imwrite(os.path.join(thu_muc_output, f"{ten_anh}_I2_conv5x5.png"), I2)
    print(f"  [Convolution] I2 (5x5): kich thuoc = {I2.shape}")
    
    # ---- Bước 4: Tích chập 7x7, padding=3, stride=2 → I3 ----
    print(f"  [Convolution] Dang tich chap 7x7, stride=2...")
    kernel_7x7 = tao_kernel_trung_binh(7)  # Kernel 7x7, mỗi phần tử = 1/49
    I3 = tich_chap(anh_xam, kernel_7x7, padding=3, stride=2)
    cv2.imwrite(os.path.join(thu_muc_output, f"{ten_anh}_I3_conv7x7_stride2.png"), I3)
    print(f"  [Convolution] I3 (7x7, stride=2): kich thuoc = {I3.shape}")
    
    # ---- Bước 5: Lọc trung vị I3 với 3x3 → I4 ----
    print(f"  [Convolution] Dang loc trung vi I3 voi cua so 3x3...")
    I4 = loc_trung_vi(I3, 3)
    cv2.imwrite(os.path.join(thu_muc_output, f"{ten_anh}_I4_median3x3.png"), I4)
    print(f"  [Convolution] I4 (median 3x3 cua I3): kich thuoc = {I4.shape}")
    
    # ---- Bước 6: Lọc trung vị I1 với 5x5 → I5 ----
    print(f"  [Convolution] Dang loc trung vi I1 voi cua so 5x5...")
    I5 = loc_trung_vi(I1, 5)
    cv2.imwrite(os.path.join(thu_muc_output, f"{ten_anh}_I5_median5x5.png"), I5)
    print(f"  [Convolution] I5 (median 5x5 cua I1): kich thuoc = {I5.shape}")
    
    # ---- Bước 7: Padding I4 nếu kích thước khác I5 ----
    if I4.shape != I5.shape:
        print(f"  [Convolution] I4 {I4.shape} khac kich thuoc I5 {I5.shape}, dang padding...")
        I4_padded = padding_cho_cung_kich_thuoc(I4, I5.shape)
        print(f"  [Convolution] I4 sau padding: {I4_padded.shape}")
    else:
        I4_padded = I4
    
    # ---- Bước 8: Tạo ảnh I6 ----
    I6 = tao_anh_I6(I4_padded, I5)
    cv2.imwrite(os.path.join(thu_muc_output, f"{ten_anh}_I6_threshold.png"), I6)
    print(f"  [Convolution] I6 (anh nguong): kich thuoc = {I6.shape}")
    
    return {
        'I1': I1, 'I2': I2, 'I3': I3,
        'I4': I4, 'I4_padded': I4_padded,
        'I5': I5, 'I6': I6,
        'duong_dan': {
            'I1': os.path.join(thu_muc_output, f"{ten_anh}_I1_conv3x3.png"),
            'I2': os.path.join(thu_muc_output, f"{ten_anh}_I2_conv5x5.png"),
            'I3': os.path.join(thu_muc_output, f"{ten_anh}_I3_conv7x7_stride2.png"),
            'I4': os.path.join(thu_muc_output, f"{ten_anh}_I4_median3x3.png"),
            'I5': os.path.join(thu_muc_output, f"{ten_anh}_I5_median5x5.png"),
            'I6': os.path.join(thu_muc_output, f"{ten_anh}_I6_threshold.png"),
        }
    }
