"""
=============================================================================
Module: lbp_processing.py
Mô tả: Xử lý Local Binary Patterns (LBP)
    - Cài đặt thuật toán LBP thủ công
    - Hỗ trợ các cấu hình: P=8, P=16, P=24 với các bán kính R khác nhau
    - Sử dụng nội suy bilinear cho điểm lân cận không nguyên

Nhóm 10 - Bài tập cuối kỳ Xử lý ảnh
=============================================================================
"""

import numpy as np
import cv2
import os
import math
from histogram_processing import chuyen_anh_xam


# =============================================================================
# HÀM 1: NỘI SUY BILINEAR
# =============================================================================
def noi_suy_bilinear(img, x, y):
    """
    Nội suy bilinear để lấy giá trị pixel tại tọa độ thực (x, y).
    
    Giải thích:
        - Khi tính LBP, các điểm lân cận trên đường tròn bán kính R
          thường không nằm đúng vị trí pixel nguyên.
        - Ví dụ: với P=8, R=1, điểm ở góc 45° có tọa độ (0.707, 0.707).
        - Nội suy bilinear ước tính giá trị pixel tại tọa độ thực 
          dựa trên 4 pixel nguyên xung quanh.
    
    Công thức nội suy bilinear:
        f(x, y) = (1-a)(1-b)*f(x1,y1) + a*(1-b)*f(x2,y1) 
                  + (1-a)*b*f(x1,y2) + a*b*f(x2,y2)
        Trong đó:
            (x1, y1) = pixel trên-trái (làm tròn xuống)
            (x2, y2) = pixel dưới-phải (làm tròn lên)
            a = phần thập phân theo trục x
            b = phần thập phân theo trục y
    
    Tham số:
        img: ảnh xám (numpy array 2D).
        x: tọa độ thực theo hàng (float).
        y: tọa độ thực theo cột (float).
    
    Trả về:
        Giá trị pixel nội suy (float).
    """
    h, w = img.shape
    
    # Tọa độ 4 pixel nguyên xung quanh
    x1 = int(math.floor(x))  # Hàng trên
    y1 = int(math.floor(y))  # Cột trái
    x2 = min(x1 + 1, h - 1)  # Hàng dưới (không vượt biên)
    y2 = min(y1 + 1, w - 1)  # Cột phải (không vượt biên)
    
    # Phần thập phân
    a = x - x1  # Khoảng cách theo hàng
    b = y - y1  # Khoảng cách theo cột
    
    # Nội suy bilinear: trung bình có trọng số của 4 pixel
    gia_tri = (1 - a) * (1 - b) * img[x1, y1] + \
              a * (1 - b) * img[x2, y1] + \
              (1 - a) * b * img[x1, y2] + \
              a * b * img[x2, y2]
    
    return gia_tri


# =============================================================================
# HÀM 2: TÍNH LBP CHO MỘT PIXEL (P=8)
# =============================================================================
def tinh_lbp_P8(img, hang, cot, R):
    """
    Tính giá trị LBP cho một pixel với P=8 điểm lân cận.
    
    Giải thích thuật toán LBP (Local Binary Pattern):
        1. Chọn P điểm trên đường tròn bán kính R quanh pixel trung tâm.
        2. So sánh giá trị mỗi điểm lân cận với pixel trung tâm:
           - Nếu lân cận >= trung tâm → bit = 1
           - Nếu lân cận < trung tâm → bit = 0
        3. Ghép P bit thành chuỗi nhị phân.
        4. Chuyển chuỗi nhị phân sang số thập phân → giá trị LBP.
    
    Ví dụ với P=8, R=1:
        8 điểm lân cận nằm trên đường tròn bán kính 1 quanh pixel trung tâm.
        Chuỗi nhị phân 8 bit, ví dụ: 10110010 → giá trị thập phân = 178.
    
    Vị trí các điểm lân cận:
        Điểm thứ p (p = 0, 1, ..., P-1) có tọa độ:
        x_p = hang - R * cos(2 * pi * p / P)
        y_p = cot + R * sin(2 * pi * p / P)
    
    Tham số:
        img: ảnh xám (numpy array 2D).
        hang: tọa độ hàng của pixel trung tâm (int).
        cot: tọa độ cột của pixel trung tâm (int).
        R: bán kính đường tròn (int).
    
    Trả về:
        gia_tri_lbp: giá trị LBP thập phân (int, 0-255).
    """
    P = 8  # Số điểm lân cận
    gia_tri_trung_tam = float(img[hang, cot])
    
    # Tạo chuỗi nhị phân 8 bit
    chuoi_nhi_phan = ""
    
    for p in range(P):
        # Tính tọa độ điểm lân cận thứ p trên đường tròn
        goc = 2.0 * math.pi * p / P
        x_p = hang - R * math.cos(goc)  # Tọa độ hàng
        y_p = cot + R * math.sin(goc)   # Tọa độ cột
        
        # Kiểm tra biên ảnh
        h, w = img.shape
        if x_p < 0 or x_p >= h - 1 or y_p < 0 or y_p >= w - 1:
            # Nếu ra ngoài biên, coi giá trị = 0
            gia_tri_lan_can = 0
        else:
            # Nội suy bilinear nếu tọa độ không nguyên
            gia_tri_lan_can = noi_suy_bilinear(img, x_p, y_p)
        
        # So sánh với pixel trung tâm
        if gia_tri_lan_can >= gia_tri_trung_tam:
            chuoi_nhi_phan += "1"
        else:
            chuoi_nhi_phan += "0"
    
    # Chuyển chuỗi nhị phân 8 bit sang số thập phân
    gia_tri_lbp = int(chuoi_nhi_phan, 2)
    
    return gia_tri_lbp


# =============================================================================
# HÀM 3: TÍNH LBP CHO MỘT PIXEL (P=16)
# =============================================================================
def tinh_lbp_P16(img, hang, cot, R):
    """
    Tính giá trị LBP cho một pixel với P=16 điểm lân cận.
    
    Giải thích:
        - 16 điểm lân cận tạo ra chuỗi nhị phân 16 bit.
        - Chuỗi 16 bit → 2^16 = 65536 giá trị khác nhau (quá lớn cho ảnh 8-bit).
        - Giải pháp: Tách 16 bit thành 2 phần, mỗi phần 8 bit.
        - Chuyển từng phần thành số thập phân (0-255).
        - Lấy giá trị LỚN NHẤT trong 2 phần làm kết quả.
    
    Lý do tách thành 2 chuỗi 8 bit:
        - Ảnh xám chỉ có 256 mức xám (0-255), tức 8 bit.
        - Nếu dùng 16 bit, giá trị LBP vượt quá 255 → không hiển thị được.
        - Tách thành 2 phần 8 bit giúp giữ giá trị trong khoảng [0, 255].
        - Lấy max để giữ thông tin nổi bật nhất.
    
    Tham số:
        img: ảnh xám (numpy array 2D).
        hang: tọa độ hàng pixel trung tâm (int).
        cot: tọa độ cột pixel trung tâm (int).
        R: bán kính (int).
    
    Trả về:
        gia_tri_lbp: giá trị LBP (int, 0-255).
    """
    P = 16
    gia_tri_trung_tam = float(img[hang, cot])
    
    # Tạo chuỗi nhị phân 16 bit
    chuoi_nhi_phan = ""
    
    for p in range(P):
        goc = 2.0 * math.pi * p / P
        x_p = hang - R * math.cos(goc)
        y_p = cot + R * math.sin(goc)
        
        h, w = img.shape
        if x_p < 0 or x_p >= h - 1 or y_p < 0 or y_p >= w - 1:
            gia_tri_lan_can = 0
        else:
            gia_tri_lan_can = noi_suy_bilinear(img, x_p, y_p)
        
        if gia_tri_lan_can >= gia_tri_trung_tam:
            chuoi_nhi_phan += "1"
        else:
            chuoi_nhi_phan += "0"
    
    # Tách 16 bit thành 2 phần, mỗi phần 8 bit
    phan_1 = chuoi_nhi_phan[0:8]   # 8 bit đầu (bit 0-7)
    phan_2 = chuoi_nhi_phan[8:16]  # 8 bit sau (bit 8-15)
    
    # Chuyển mỗi phần sang số thập phân
    gia_tri_1 = int(phan_1, 2)  # 0-255
    gia_tri_2 = int(phan_2, 2)  # 0-255
    
    # Lấy giá trị lớn nhất
    gia_tri_lbp = max(gia_tri_1, gia_tri_2)
    
    return gia_tri_lbp


# =============================================================================
# HÀM 4: TÍNH LBP CHO MỘT PIXEL (P=24)
# =============================================================================
def tinh_lbp_P24(img, hang, cot, R):
    """
    Tính giá trị LBP cho một pixel với P=24 điểm lân cận.
    
    Giải thích:
        - 24 điểm lân cận tạo chuỗi nhị phân 24 bit.
        - 2^24 = 16,777,216 giá trị → quá lớn cho ảnh 8-bit.
        - Giải pháp: Tách 24 bit thành 3 phần, mỗi phần 8 bit.
        - Chuyển từng phần sang số thập phân (0-255).
        - Lấy giá trị LỚN NHẤT trong 3 phần làm kết quả.
    
    Lý do tách thành 3 chuỗi 8 bit:
        - 24 / 8 = 3 phần → mỗi phần vừa đúng 1 byte (8 bit).
        - Giá trị mỗi phần nằm trong [0, 255] → hiển thị được trên ảnh xám.
        - Lấy max giữ lại pattern nổi bật nhất trong 3 phần.
    
    Tham số:
        img: ảnh xám (numpy array 2D).
        hang: tọa độ hàng pixel trung tâm (int).
        cot: tọa độ cột pixel trung tâm (int).
        R: bán kính (int).
    
    Trả về:
        gia_tri_lbp: giá trị LBP (int, 0-255).
    """
    P = 24
    gia_tri_trung_tam = float(img[hang, cot])
    
    # Tạo chuỗi nhị phân 24 bit
    chuoi_nhi_phan = ""
    
    for p in range(P):
        goc = 2.0 * math.pi * p / P
        x_p = hang - R * math.cos(goc)
        y_p = cot + R * math.sin(goc)
        
        h, w = img.shape
        if x_p < 0 or x_p >= h - 1 or y_p < 0 or y_p >= w - 1:
            gia_tri_lan_can = 0
        else:
            gia_tri_lan_can = noi_suy_bilinear(img, x_p, y_p)
        
        if gia_tri_lan_can >= gia_tri_trung_tam:
            chuoi_nhi_phan += "1"
        else:
            chuoi_nhi_phan += "0"
    
    # Tách 24 bit thành 3 phần, mỗi phần 8 bit
    phan_1 = chuoi_nhi_phan[0:8]    # 8 bit đầu (bit 0-7)
    phan_2 = chuoi_nhi_phan[8:16]   # 8 bit giữa (bit 8-15)
    phan_3 = chuoi_nhi_phan[16:24]  # 8 bit cuối (bit 16-23)
    
    # Chuyển mỗi phần sang số thập phân
    gia_tri_1 = int(phan_1, 2)
    gia_tri_2 = int(phan_2, 2)
    gia_tri_3 = int(phan_3, 2)
    
    # Lấy giá trị lớn nhất trong 3 phần
    gia_tri_lbp = max(gia_tri_1, gia_tri_2, gia_tri_3)
    
    return gia_tri_lbp


# =============================================================================
# HÀM 5: TÍNH LBP CHO TOÀN BỘ ẢNH
# =============================================================================
def tinh_lbp_toan_anh(gray_img, P, R):
    """
    Tính LBP cho toàn bộ ảnh xám.
    
    Giải thích:
        - Duyệt qua từng pixel (trừ viền R pixel ở biên).
        - Tại mỗi pixel, tính giá trị LBP dựa trên P và R.
        - Pixel ở viền không đủ lân cận → giữ giá trị 0.
    
    Tham số:
        gray_img: ảnh xám đầu vào (numpy array 2D).
        P: số điểm lân cận (8, 16, hoặc 24).
        R: bán kính đường tròn (int).
    
    Trả về:
        lbp_img: ảnh LBP (numpy array 2D), kiểu uint8.
    """
    h, w = gray_img.shape
    lbp_img = np.zeros((h, w), dtype=np.uint8)
    
    # Chọn hàm tính LBP phù hợp với P
    if P == 8:
        ham_tinh_lbp = tinh_lbp_P8
    elif P == 16:
        ham_tinh_lbp = tinh_lbp_P16
    elif P == 24:
        ham_tinh_lbp = tinh_lbp_P24
    else:
        raise ValueError(f"P = {P} khong duoc ho tro. Chi ho tro P = 8, 16, 24.")
    
    # Duyệt qua từng pixel (bỏ viền R pixel ở biên)
    # Pixel ở viền không đủ lân cận nên bỏ qua
    tong_pixel = (h - 2 * R) * (w - 2 * R)
    dem = 0
    
    for i in range(R, h - R):
        for j in range(R, w - R):
            lbp_img[i, j] = ham_tinh_lbp(gray_img, i, j, R)
            dem += 1
        
        # In tiến trình mỗi 50 hàng
        if (i - R) % 50 == 0:
            phan_tram = (dem / tong_pixel) * 100
            print(f"    LBP P={P} R={R}: {phan_tram:.1f}% ({dem}/{tong_pixel})")
    
    return lbp_img


# =============================================================================
# HÀM 6: XỬ LÝ LBP CHO MỘT ẢNH (GỌI TẤT CẢ CÁC CẤU HÌNH)
# =============================================================================
def xu_ly_lbp(img, ten_anh, thu_muc_output):
    """
    Thực hiện toàn bộ quy trình LBP cho một ảnh.
    
    Các cấu hình:
        1. P=8, R=1: 8 điểm, bán kính 1 (chi tiết nhỏ, pattern cục bộ).
        2. P=8, R=2: 8 điểm, bán kính 2 (pattern rộng hơn).
        3. P=16, R=2: 16 điểm, bán kính 2 (chi tiết hơn, vùng rộng).
        4. P=16, R=3: 16 điểm, bán kính 3 (pattern lớn).
        5. P=24, R=3: 24 điểm, bán kính 3 (chi tiết nhất, vùng lớn).
    
    Tham số:
        img: ảnh màu đầu vào (numpy array).
        ten_anh: tên file ảnh gốc (string).
        thu_muc_output: đường dẫn thư mục lưu kết quả.
    
    Trả về:
        dict chứa các ảnh LBP kết quả.
    """
    os.makedirs(thu_muc_output, exist_ok=True)
    
    # Chuyển sang ảnh xám
    anh_xam = chuyen_anh_xam(img)
    
    # Danh sách các cấu hình (P, R)
    cau_hinh = [
        (8, 1),   # P=8, R=1
        (8, 2),   # P=8, R=2
        (16, 2),  # P=16, R=2
        (16, 3),  # P=16, R=3
        (24, 3),  # P=24, R=3
    ]
    
    ket_qua = {}
    
    for P, R in cau_hinh:
        print(f"  [LBP] Dang tinh LBP P={P}, R={R}...")
        
        # Tính LBP cho toàn bộ ảnh
        lbp_img = tinh_lbp_toan_anh(anh_xam, P, R)
        
        # Lưu ảnh LBP
        ten_file = f"{ten_anh}_LBP_P{P}_R{R}.png"
        duong_dan = os.path.join(thu_muc_output, ten_file)
        cv2.imwrite(duong_dan, lbp_img)
        print(f"  [LBP] Da luu: {duong_dan}")
        
        ket_qua[f"P{P}_R{R}"] = {
            'anh': lbp_img,
            'duong_dan': duong_dan,
            'P': P,
            'R': R,
        }
    
    return ket_qua
