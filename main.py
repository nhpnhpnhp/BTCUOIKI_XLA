"""
=============================================================================
File: main.py
Mô tả: Chương trình chính - Xử lý ảnh cuối kỳ
    - Đọc tất cả ảnh từ thư mục input_images/
    - Thực hiện 3 nhiệm vụ: Histogram, Tích chập, LBP
    - Lưu kết quả vào thư mục output/
    - Tạo báo cáo PDF tự động

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
from report_generator import tao_bao_cao_pdf


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

# Đường dẫn file báo cáo PDF
DUONG_DAN_PDF = os.path.join(THU_MUC_OUTPUT, "report.pdf")

# Các định dạng ảnh được hỗ trợ
DINH_DANG_ANH = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp')


# =============================================================================
# HÀM KIỂM TRA VÀ TẠO THƯ MỤC
# =============================================================================
def kiem_tra_va_tao_thu_muc():
    """
    Kiểm tra thư mục input tồn tại và tạo thư mục output nếu chưa có.
    
    Trả về:
        True nếu mọi thứ OK, False nếu có lỗi.
    """
    # Kiểm tra thư mục input
    if not os.path.exists(THU_MUC_INPUT):
        print(f"LOI: Khong tim thay thu muc input: {THU_MUC_INPUT}")
        print(f"Hay tao thu muc 'input_images/' va dat 10 anh vao do.")
        return False
    
    # Tạo các thư mục output
    for thu_muc in [THU_MUC_OUTPUT, THU_MUC_HISTOGRAM, THU_MUC_CONVOLUTION, THU_MUC_LBP]:
        os.makedirs(thu_muc, exist_ok=True)
    
    print(f"Thu muc input: {THU_MUC_INPUT}")
    print(f"Thu muc output: {THU_MUC_OUTPUT}")
    return True


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
    
    if len(danh_sach) == 0:
        print(f"LOI: Khong tim thay anh nao trong {THU_MUC_INPUT}")
        print(f"Dinh dang ho tro: {', '.join(DINH_DANG_ANH)}")
    else:
        print(f"\nTim thay {len(danh_sach)} anh:")
        for i, ten in enumerate(danh_sach, 1):
            print(f"  {i}. {ten}")
    
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
    try:
        img = cv2.imread(duong_dan)
        if img is None:
            print(f"  LOI: Khong doc duoc anh: {duong_dan}")
            print(f"  Kiem tra lai dinh dang hoac duong dan file.")
            return None
        return img
    except Exception as e:
        print(f"  LOI khi doc anh {duong_dan}: {str(e)}")
        return None


# =============================================================================
# HÀM XỬ LÝ MỘT ẢNH (TẤT CẢ 3 NHIỆM VỤ)
# =============================================================================
def xu_ly_mot_anh(ten_file, so_thu_tu, tong_so):
    """
    Xử lý một ảnh: thực hiện cả 3 nhiệm vụ (Histogram, Tích chập, LBP).
    
    Tham số:
        ten_file: tên file ảnh (string).
        so_thu_tu: số thứ tự ảnh hiện tại (int).
        tong_so: tổng số ảnh cần xử lý (int).
    
    Trả về:
        dict chứa kết quả xử lý, hoặc None nếu lỗi.
    """
    # Lấy tên ảnh (không phần mở rộng) để đặt tên file kết quả
    ten_anh = os.path.splitext(ten_file)[0]
    duong_dan_anh = os.path.join(THU_MUC_INPUT, ten_file)
    
    print(f"\n{'='*60}")
    print(f"ANH {so_thu_tu}/{tong_so}: {ten_file}")
    print(f"{'='*60}")
    
    # Đọc ảnh
    img = doc_anh(duong_dan_anh)
    if img is None:
        return None
    
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
    try:
        ket_qua['histogram'] = xu_ly_histogram(img, ten_anh, THU_MUC_HISTOGRAM)
        thoi_gian = time.time() - thoi_gian_bat_dau
        print(f"  >> Hoan thanh histogram trong {thoi_gian:.2f} giay.")
    except Exception as e:
        print(f"  LOI xu ly histogram: {str(e)}")
    
    # ---- NHIỆM VỤ 2: TÍCH CHẬP ----
    print(f"\n--- NHIEM VU 2: TICH CHAP & LOC TRUNG VI ---")
    thoi_gian_bat_dau = time.time()
    try:
        ket_qua['convolution'] = xu_ly_tich_chap(img, ten_anh, THU_MUC_CONVOLUTION)
        thoi_gian = time.time() - thoi_gian_bat_dau
        print(f"  >> Hoan thanh tich chap trong {thoi_gian:.2f} giay.")
    except Exception as e:
        print(f"  LOI xu ly tich chap: {str(e)}")
    
    # ---- NHIỆM VỤ 3: LBP ----
    print(f"\n--- NHIEM VU 3: LOCAL BINARY PATTERNS ---")
    thoi_gian_bat_dau = time.time()
    try:
        ket_qua['lbp'] = xu_ly_lbp(img, ten_anh, THU_MUC_LBP)
        thoi_gian = time.time() - thoi_gian_bat_dau
        print(f"  >> Hoan thanh LBP trong {thoi_gian:.2f} giay.")
    except Exception as e:
        print(f"  LOI xu ly LBP: {str(e)}")
    
    return ket_qua


# =============================================================================
# HÀM CHÍNH
# =============================================================================
def main():
    """
    Hàm chính: điều phối toàn bộ quy trình xử lý ảnh.
    
    Luồng xử lý:
        1. Kiểm tra thư mục input/output.
        2. Đọc danh sách ảnh.
        3. Xử lý từng ảnh (3 nhiệm vụ).
        4. Tạo báo cáo PDF.
    """
    print("=" * 60)
    print("  BAI TAP CUOI KY - XU LY ANH SO")
    print("  NHOM 10")
    print("=" * 60)
    
    # Bước 1: Kiểm tra thư mục
    if not kiem_tra_va_tao_thu_muc():
        sys.exit(1)
    
    # Bước 2: Đọc danh sách ảnh
    danh_sach_anh = doc_danh_sach_anh()
    if len(danh_sach_anh) == 0:
        sys.exit(1)
    
    # Bước 3: Xử lý từng ảnh
    thoi_gian_tong = time.time()
    ket_qua_tat_ca = []
    
    for i, ten_file in enumerate(danh_sach_anh, 1):
        ket_qua = xu_ly_mot_anh(ten_file, i, len(danh_sach_anh))
        if ket_qua is not None:
            ket_qua_tat_ca.append(ket_qua)
    
    # Bước 4: Tạo báo cáo PDF
    if len(ket_qua_tat_ca) > 0:
        tao_bao_cao_pdf(DUONG_DAN_PDF, ket_qua_tat_ca)
    else:
        print("\nKhong co ket qua de tao bao cao.")
    
    # Tổng kết
    thoi_gian_tong = time.time() - thoi_gian_tong
    print(f"\n{'='*60}")
    print(f"HOAN THANH!")
    print(f"  So anh da xu ly: {len(ket_qua_tat_ca)}/{len(danh_sach_anh)}")
    print(f"  Tong thoi gian: {thoi_gian_tong:.2f} giay")
    print(f"  Ket qua luu tai: {THU_MUC_OUTPUT}")
    print(f"  Bao cao PDF: {DUONG_DAN_PDF}")
    print(f"{'='*60}")


# =============================================================================
# CHẠY CHƯƠNG TRÌNH
# =============================================================================
if __name__ == "__main__":
    main()
