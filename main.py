"""
=============================================================================
File: main.py
Mô tả: Chương trình chính - Xử lý ảnh cuối kỳ
    - Đọc tất cả ảnh từ thư mục input_images/
    - Thực hiện 3 nhiệm vụ: Histogram, Tích chập, LBP
    - Lưu kết quả vào thư mục output/

Cách chạy:
    python main.py

Yêu cầu:
    - Đặt 10 ảnh vào thư mục input_images/
    - Cài đặt: pip install opencv-python numpy matplotlib

Nhóm 10 - Bài tập cuối kỳ Xử lý ảnh
=============================================================================
"""

import os
import sys
import time
import cv2
import numpy as np

# Import các module xử lý
from histogram_processing import xu_ly_histogram
from convolution_processing import xu_ly_tich_chap
from lbp_processing import xu_ly_lbp

# =============================================================================
# CẤU HÌNH ĐƯỜNG DẪN
# =============================================================================
# Thư mục gốc của project (nơi chứa file main.py)
THU_MUC_GOC = os.path.dirname(os.path.abspath(__file__))

# Thư mục chứa ảnh đầu vào
THU_MUC_INPUT = os.path.join(THU_MUC_GOC, "input_images")

# Thư mục chứa kết quả
THU_MUC_OUTPUT = os.path.join(THU_MUC_GOC, "output")
THU_MUC_HISTOGRAM = os.path.join(THU_MUC_OUTPUT, "histograms")
THU_MUC_CONVOLUTION = os.path.join(THU_MUC_OUTPUT, "convolution")
THU_MUC_LBP = os.path.join(THU_MUC_OUTPUT, "lbp")

# Các định dạng ảnh được hỗ trợ
DINH_DANG_ANH = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp')


# =============================================================================
# HÀM ĐỌC DANH SÁCH ẢNH
# =============================================================================
def doc_danh_sach_anh():
    """
    Đọc danh sách các file ảnh từ thư mục input.
    
    Trả về:
        list các tên file ảnh (sorted), hoặc list rỗng nếu không có ảnh.
    """
    danh_sach = []
    
    for ten_file in sorted(os.listdir(THU_MUC_INPUT)):
        # Kiểm tra định dạng file
        if ten_file.lower().endswith(DINH_DANG_ANH):
            danh_sach.append(ten_file)
    return danh_sach


# =============================================================================
# HÀM ĐỌC MỘT ẢNH
# =============================================================================
def doc_anh(duong_dan):
    """
    Đọc một ảnh từ đường dẫn.
    
    Tham số:
        duong_dan: đường dẫn file ảnh (string).
    
    Trả về:
        img: ảnh (numpy array), hoặc None nếu lỗi.
    """
    img = cv2.imread(duong_dan)
    return img


# =============================================================================
# HÀM XỬ LÝ MỘT ẢNH (TẤT CẢ 3 NHIỆM VỤ)
# =============================================================================
def xu_ly_mot_anh(ten_file):
    """
    Xử lý một ảnh: thực hiện cả 3 nhiệm vụ (Histogram, Tích chập, LBP).
    
    Tham số:
        ten_file: tên file ảnh (string).
        tong_so: tổng số ảnh cần xử lý (int).
    
    Trả về:
        dict chứa kết quả xử lý, hoặc None nếu lỗi.
    """
    # Lấy tên ảnh (không phần mở rộng) để đặt tên file kết quả
    ten_anh = os.path.splitext(ten_file)[0]
    duong_dan_anh = os.path.join(THU_MUC_INPUT, ten_file)
    
    print(f"\n{'='*60}")
    print(f"ANH {ten_file}")
    print(f"{'='*60}")
    
    # Đọc ảnh
    img = doc_anh(duong_dan_anh)
    print(f"  Kich thuoc anh: {img.shape}")
    
    ket_qua = {
        'ten_anh': ten_anh,
        'ten_file': ten_file,
        'histogram': None,
        'convolution': None,
        'lbp': None,
    }
    
    # ---- NHIỆM VỤ 1: HISTOGRAM ----
    print(f"\n--- NHIEM VU 1: HISTOGRAM ---")
    thoi_gian_bat_dau = time.time()
    ket_qua['histogram'] = xu_ly_histogram(img, ten_anh, THU_MUC_HISTOGRAM)
    thoi_gian = time.time() - thoi_gian_bat_dau
    print(f"  >> Hoan thanh histogram trong {thoi_gian:.2f} giay.")
    
    # ---- NHIỆM VỤ 2: TÍCH CHẬP ----
    print(f"\n--- NHIEM VU 2: TICH CHAP & LOC TRUNG VI ---")
    thoi_gian_bat_dau = time.time()
    ket_qua['convolution'] = xu_ly_tich_chap(img, ten_anh, THU_MUC_CONVOLUTION)
    thoi_gian = time.time() - thoi_gian_bat_dau
    print(f"  >> Hoan thanh tich chap trong {thoi_gian:.2f} giay.")

    
    # ---- NHIỆM VỤ 3: LBP ----
    print(f"\n--- NHIEM VU 3: LOCAL BINARY PATTERNS ---")
    thoi_gian_bat_dau = time.time()
    ket_qua['lbp'] = xu_ly_lbp(img, ten_anh, THU_MUC_LBP)
    thoi_gian = time.time() - thoi_gian_bat_dau
    print(f"  >> Hoan thanh LBP trong {thoi_gian:.2f} giay.")
    
    return ket_qua


# =============================================================================
# HÀM CHÍNH
# =============================================================================
def main():
    """
    Hàm chính: điều phối toàn bộ quy trình xử lý ảnh.
    
    Luồng xử lý:
         Đọc danh sách ảnh.
         Xử lý từng ảnh (3 nhiệm vụ).
    """
    danh_sach_anh = doc_danh_sach_anh()

    
    #Xử lý từng ảnh
    thoi_gian_tong = time.time()
    ket_qua_tat_ca = []
    
    for ten_file in danh_sach_anh:
        ket_qua = xu_ly_mot_anh(ten_file)
        if ket_qua is not None:
            ket_qua_tat_ca.append(ket_qua)
    
    
    # Tổng kết
    thoi_gian_tong = time.time() - thoi_gian_tong
    print(f"\n{'='*60}")
    print(f"HOAN THANH!")
    print(f"  So anh da xu ly: {len(ket_qua_tat_ca)}/{len(danh_sach_anh)}")
    print(f"  Tong thoi gian: {thoi_gian_tong:.2f} giay")
    print(f"  Ket qua luu tai: {THU_MUC_OUTPUT}")
    print(f"{'='*60}")


# =============================================================================
# CHẠY CHƯƠNG TRÌNH
# =============================================================================
if __name__ == "__main__":
    main()
