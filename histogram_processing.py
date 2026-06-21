"""
=============================================================================
Module: histogram_processing.py
Mô tả: Xử lý histogram cho ảnh xám
    - Chuyển ảnh màu sang ảnh xám
    - Tính và vẽ histogram
    - Cân bằng histogram (Histogram Equalization)
    - Thu hẹp mức xám về khoảng [30, 120]

Nhóm 10 - Bài tập cuối kỳ Xử lý ảnh
=============================================================================
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


# =============================================================================
# HÀM 1: CHUYỂN ẢNH MÀU SANG ẢNH XÁM
# =============================================================================
def chuyen_anh_xam(img):
    """
    Chuyển ảnh màu (BGR) sang ảnh xám.
    
    Công thức chuyển đổi (chuẩn ITU-R BT.601):
        Gray = 0.299 * R + 0.587 * G + 0.114 * B
    
    Giải thích:
        - Mắt người nhạy cảm nhất với màu xanh lá (Green), 
          nên hệ số của G lớn nhất (0.587).
        - Màu đỏ (Red) có hệ số 0.299.
        - Màu xanh dương (Blue) có hệ số nhỏ nhất (0.114).
        - OpenCV đọc ảnh theo thứ tự BGR (không phải RGB).
    
    Tham số:
        img: ảnh đầu vào (numpy array), có thể là ảnh màu hoặc ảnh xám.
    
    Trả về:
        gray: ảnh xám (numpy array 2D), kiểu uint8.
    """
    # Nếu ảnh đã là ảnh xám (chỉ có 2 chiều), trả về luôn
    if len(img.shape) == 2:
        return img.copy()
    
    # Nếu ảnh có 3 kênh màu, dùng OpenCV để chuyển BGR -> Gray
    # Bên trong, OpenCV dùng công thức: Gray = 0.299*R + 0.587*G + 0.114*B
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


# =============================================================================
# HÀM 2: TÍNH HISTOGRAM CỦA ẢNH XÁM
# =============================================================================
def tinh_histogram(gray_img):
    """
    Tính histogram của ảnh xám.
    
    Giải thích:
        - Histogram là biểu đồ đếm số lượng pixel cho mỗi mức xám (0-255).
        - Mức xám 0 = đen, mức xám 255 = trắng.
        - Histogram cho biết phân bố độ sáng của ảnh.
    
    Cách tính:
        - Tạo mảng 256 phần tử (tương ứng 256 mức xám).
        - Duyệt qua từng pixel, tăng giá trị đếm tại vị trí tương ứng.
    
    Tham số:
        gray_img: ảnh xám (numpy array 2D), kiểu uint8.
    
    Trả về:
        hist: mảng 256 phần tử, hist[i] = số pixel có mức xám i.
    """
    # Tạo mảng histogram với 256 phần tử, ban đầu bằng 0
    hist = np.zeros(256, dtype=np.int64)
    
    # Duyệt qua tất cả pixel trong ảnh
    # flatten() chuyển ảnh 2D thành mảng 1D để dễ duyệt
    for gia_tri_pixel in gray_img.flatten():
        hist[gia_tri_pixel] += 1
    
    return hist


# =============================================================================
# HÀM 3: VẼ HISTOGRAM VÀ LƯU THÀNH FILE ẢNH
# =============================================================================
def ve_histogram(gray_img, tieu_de, duong_dan_luu):
    """
    Vẽ histogram của ảnh xám và lưu thành file ảnh.
    
    Giải thích:
        - Trục X: mức xám từ 0 đến 255.
        - Trục Y: số lượng pixel tương ứng.
        - Biểu đồ dạng cột (bar chart).
    
    Tham số:
        gray_img: ảnh xám (numpy array 2D).
        tieu_de: tiêu đề của biểu đồ (string).
        duong_dan_luu: đường dẫn file để lưu biểu đồ (string).
    """
    # Tính histogram
    hist = tinh_histogram(gray_img)
    
    # Tạo figure với kích thước 10x4 inch
    plt.figure(figsize=(10, 4))
    
    # Vẽ biểu đồ cột
    plt.bar(range(256), hist, color='steelblue', width=1.0)
    
    # Đặt tiêu đề và nhãn trục
    plt.title(tieu_de, fontsize=14)
    plt.xlabel('Muc xam (0-255)', fontsize=11)
    plt.ylabel('So luong pixel', fontsize=11)
    plt.xlim([0, 255])
    
    # Lưu biểu đồ
    plt.tight_layout()
    plt.savefig(duong_dan_luu, dpi=150, bbox_inches='tight')
    plt.close()  # Đóng figure để giải phóng bộ nhớ


# =============================================================================
# HÀM 4: CÂN BẰNG HISTOGRAM (HISTOGRAM EQUALIZATION)
# =============================================================================
def can_bang_histogram(gray_img):
    """
    Cân bằng histogram của ảnh xám.
    
    Giải thích:
        Cân bằng histogram là kỹ thuật làm cho phân bố mức xám trải đều
        trên toàn bộ dải [0, 255], giúp tăng độ tương phản của ảnh.
    
    Các bước thực hiện:
        Bước 1: Tính histogram h(i) - đếm số pixel cho mỗi mức xám i.
        Bước 2: Tính CDF (Cumulative Distribution Function) - hàm phân phối tích lũy.
                 CDF(i) = tổng h(0) + h(1) + ... + h(i)
        Bước 3: Chuẩn hóa CDF về khoảng [0, 255]:
                 muc_xam_moi(i) = round(CDF(i) * 255 / tổng_số_pixel)
        Bước 4: Ánh xạ từng pixel sang mức xám mới.
    
    Tham số:
        gray_img: ảnh xám đầu vào (numpy array 2D), kiểu uint8.
    
    Trả về:
        equalized: ảnh sau cân bằng histogram (numpy array 2D), kiểu uint8.
    """
    # Lấy kích thước ảnh
    chieu_cao, chieu_rong = gray_img.shape
    tong_pixel = chieu_cao * chieu_rong
    
    # Bước 1: Tính histogram
    hist = tinh_histogram(gray_img)
    
    # Bước 2: Tính CDF (hàm phân phối tích lũy)
    # CDF[i] = tổng số pixel có mức xám từ 0 đến i
    cdf = np.cumsum(hist)
    
    # Bước 3: Chuẩn hóa CDF về khoảng [0, 255]
    # Công thức: muc_xam_moi = round(CDF[i] * 255 / tổng_pixel)
    cdf_chuan_hoa = np.round((cdf * 255.0) / tong_pixel).astype(np.uint8)
    
    # Bước 4: Ánh xạ từng pixel sang mức xám mới
    # Dùng kỹ thuật look-up table: pixel mới = bảng_ánh_xạ[pixel cũ]
    anh_can_bang = cdf_chuan_hoa[gray_img]
    
    return anh_can_bang


# =============================================================================
# HÀM 5: THU HẸP MỨC XÁM VỀ KHOẢNG [30, 120]
# =============================================================================
def thu_hep_muc_xam(gray_img, muc_xam_min_moi=30, muc_xam_max_moi=120):
    """
    Thu hẹp (co giãn) mức xám của ảnh về khoảng [muc_xam_min_moi, muc_xam_max_moi].
    
    Giải thích:
        Ánh xạ tuyến tính mức xám từ khoảng [old_min, old_max] sang [new_min, new_max].
    
    Công thức:
        pixel_moi = (pixel_cu - old_min) * (new_max - new_min) / (old_max - old_min) + new_min
    
    Ví dụ:
        Nếu ảnh gốc có mức xám từ 0 đến 255, sau khi thu hẹp về [30, 120]:
        - Pixel có giá trị 0 → 30
        - Pixel có giá trị 255 → 120
        - Pixel có giá trị 128 → khoảng 75
    
    Tham số:
        gray_img: ảnh xám đầu vào (numpy array 2D).
        muc_xam_min_moi: giá trị mức xám tối thiểu mới (mặc định 30).
        muc_xam_max_moi: giá trị mức xám tối đa mới (mặc định 120).
    
    Trả về:
        result: ảnh sau khi thu hẹp mức xám (numpy array 2D), kiểu uint8.
    """
    # Tìm giá trị min và max hiện tại của ảnh
    gia_tri_min_cu = float(gray_img.min())
    gia_tri_max_cu = float(gray_img.max())
    
    # Trường hợp đặc biệt: nếu ảnh chỉ có 1 mức xám duy nhất
    if gia_tri_max_cu == gia_tri_min_cu:
        return np.full_like(gray_img, (muc_xam_min_moi + muc_xam_max_moi) // 2)
    
    # Áp dụng công thức ánh xạ tuyến tính
    # pixel_moi = (pixel_cu - min_cu) * (max_moi - min_moi) / (max_cu - min_cu) + min_moi
    ket_qua = (muc_xam_max_moi - muc_xam_min_moi) / (gia_tri_max_cu - gia_tri_min_cu) * (gray_img.astype(np.float64) - gia_tri_min_cu) + muc_xam_min_moi
    return ket_qua.astype(np.uint8)


# =============================================================================
# HÀM 6: XỬ LÝ HISTOGRAM CHO MỘT ẢNH (GỌI TẤT CẢ CÁC BƯỚC)
# =============================================================================
def xu_ly_histogram(img, ten_anh, thu_muc_output):
    """
    Thực hiện toàn bộ quy trình xử lý histogram cho một ảnh.
    
    Các bước:
        1. Chuyển ảnh sang xám.
        2. Vẽ histogram H1 (ảnh xám gốc).
        3. Cân bằng histogram → vẽ histogram H2.
        4. Thu hẹp mức xám về [30, 120] → vẽ histogram.
        5. Lưu tất cả kết quả.
    
    Tham số:
        img: ảnh màu đầu vào (numpy array).
        ten_anh: tên file ảnh gốc (string), ví dụ: "image1".
        thu_muc_output: đường dẫn thư mục lưu kết quả.
    
    Trả về:
        dict chứa các ảnh kết quả để dùng cho báo cáo.
    """
    # Tạo thư mục output nếu chưa có
    os.makedirs(thu_muc_output, exist_ok=True)
    
    # ---- Bước 1: Chuyển ảnh sang xám ----
    anh_xam = chuyen_anh_xam(img)
    duong_dan_anh_xam = os.path.join(thu_muc_output, f"{ten_anh}_gray.png")
    cv2.imwrite(duong_dan_anh_xam, anh_xam)
    print(f"  [Histogram] Da luu anh xam: {duong_dan_anh_xam}")
    
    # ---- Bước 2: Vẽ histogram H1 (ảnh xám gốc) ----
    duong_dan_h1 = os.path.join(thu_muc_output, f"{ten_anh}_H1_histogram.png")
    ve_histogram(anh_xam, f"H1 - Histogram anh xam goc ({ten_anh})", duong_dan_h1)
    print(f"  [Histogram] Da luu H1: {duong_dan_h1}")
    
    # ---- Bước 3: Cân bằng histogram ----
    anh_can_bang = can_bang_histogram(anh_xam)
    duong_dan_can_bang = os.path.join(thu_muc_output, f"{ten_anh}_equalized.png")
    cv2.imwrite(duong_dan_can_bang, anh_can_bang)
    print(f"  [Histogram] Da luu anh can bang: {duong_dan_can_bang}")
    
    # ---- Bước 4: Vẽ histogram H2 (sau cân bằng) ----
    duong_dan_h2 = os.path.join(thu_muc_output, f"{ten_anh}_H2_histogram.png")
    ve_histogram(anh_can_bang, f"H2 - Histogram sau can bang ({ten_anh})", duong_dan_h2)
    print(f"  [Histogram] Da luu H2: {duong_dan_h2}")
    
    # ---- Bước 5: Thu hẹp mức xám về [30, 120] ----
    anh_thu_hep = thu_hep_muc_xam(anh_can_bang, 30, 120)
    duong_dan_thu_hep = os.path.join(thu_muc_output, f"{ten_anh}_narrowed_30_120.png")
    cv2.imwrite(duong_dan_thu_hep, anh_thu_hep)
    print(f"  [Histogram] Da luu anh thu hep: {duong_dan_thu_hep}")
    
    # ---- Bước 6: Vẽ histogram sau thu hẹp ----
    duong_dan_h3 = os.path.join(thu_muc_output, f"{ten_anh}_H3_narrowed_histogram.png")
    ve_histogram(anh_thu_hep, f"Histogram sau thu hep [30,120] ({ten_anh})", duong_dan_h3)
    print(f"  [Histogram] Da luu histogram thu hep: {duong_dan_h3}")
    