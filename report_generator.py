"""
=============================================================================
Module: report_generator.py
Mô tả: Tạo báo cáo PDF tự động
    - Thông tin nhóm sinh viên
    - Mô tả thuật toán
    - Kết quả xử lý cho từng ảnh
    - Ảnh minh họa trước và sau xử lý

Nhóm 10 - Bài tập cuối kỳ Xử lý ảnh
=============================================================================
"""

import os
import matplotlib
matplotlib.use('Agg')  # Dùng backend không hiển thị GUI (cho server)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
import numpy as np


# =============================================================================
# HÀM 1: THÊM TRANG TIÊU ĐỀ
# =============================================================================
def them_trang_tieu_de(pdf):
    """
    Thêm trang bìa với thông tin nhóm vào báo cáo PDF.
    
    Tham số:
        pdf: đối tượng PdfPages.
    """
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 ngang
    fig.patch.set_facecolor('#1a1a2e')
    
    # Tiêu đề lớn
    fig.text(0.5, 0.82, 'BAI TAP CUOI KY', fontsize=30, fontweight='bold',
             ha='center', va='center', color='#e94560',
             fontfamily='monospace')
    fig.text(0.5, 0.72, 'XU LY ANH SO', fontsize=36, fontweight='bold',
             ha='center', va='center', color='#ffffff',
             fontfamily='monospace')
    
    # Đường kẻ trang trí
    ax = fig.add_axes([0.15, 0.65, 0.7, 0.005])
    ax.set_facecolor('#e94560')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Thông tin nhóm
    fig.text(0.5, 0.58, 'NHOM 10', fontsize=24, fontweight='bold',
             ha='center', va='center', color='#0f3460',
             fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#e94560', alpha=0.9))
    
    # Danh sách thành viên
    thanh_vien = [
        '1. 23133054 - Nguyễn Hồ Phát',
        '2. 23133061 - Phan Trọng Quí',
        '3. 23133056 - Phan Trọng Phú',
        '4. 23133035 - Trần Minh Khánh',
    ]
    
    y_start = 0.48
    for i, tv in enumerate(thanh_vien):
        fig.text(0.5, y_start - i * 0.06, tv, fontsize=16,
                 ha='center', va='center', color='#ffffff',
                 fontfamily='DejaVu Sans')
    
    # Footer
    fig.text(0.5, 0.08, 'Python | OpenCV | NumPy | Matplotlib',
             fontsize=12, ha='center', va='center', color='#888888',
             fontfamily='monospace')
    
    pdf.savefig(fig)
    plt.close(fig)


# =============================================================================
# HÀM 2: THÊM TRANG MÔ TẢ THUẬT TOÁN
# =============================================================================
def them_trang_mo_ta_thuat_toan(pdf):
    """
    Thêm các trang mô tả thuật toán vào báo cáo PDF.
    """
    # ---- Trang mô tả Histogram ----
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor('#f5f5f5')
    
    fig.text(0.5, 0.92, 'NHIEM VU 1: HISTOGRAM', fontsize=22, fontweight='bold',
             ha='center', color='#e94560', fontfamily='monospace')
    
    mo_ta_histogram = """
1. CHUYEN ANH XAM:
   - Dung cong thuc: Gray = 0.299*R + 0.587*G + 0.114*B
   - Mat nguoi nhay cam voi mau xanh la (G) nhat.

2. HISTOGRAM:
   - Bieu do dem so luong pixel cho moi muc xam (0-255).
   - Truc X = muc xam, Truc Y = so pixel.

3. CAN BANG HISTOGRAM:
   - Lam cho phan bo muc xam trai deu tren [0, 255].
   - Tang do tuong phan cua anh.
   - Buoc: Tinh histogram -> CDF -> Chuan hoa CDF -> Anh xa pixel.

4. THU HEP MUC XAM [30, 120]:
   - Anh xa tuyen tinh: pixel_moi = (pixel_cu - min) * (120-30) / (max-min) + 30
   - Tat ca muc xam duoc nen lai trong khoang [30, 120].
"""
    fig.text(0.08, 0.82, mo_ta_histogram, fontsize=12,
             ha='left', va='top', color='#333333',
             fontfamily='monospace', linespacing=1.6)
    
    pdf.savefig(fig)
    plt.close(fig)
    
    # ---- Trang mô tả Tích chập và Lọc trung vị ----
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor('#f5f5f5')
    
    fig.text(0.5, 0.92, 'NHIEM VU 2: TICH CHAP & LOC TRUNG VI', fontsize=22,
             fontweight='bold', ha='center', color='#e94560', fontfamily='monospace')
    
    mo_ta_conv = """
1. TICH CHAP (CONVOLUTION):
   - Truot kernel qua anh, nhan tung phan tu roi tinh tong.
   - Kernel trung binh: lam mo anh, giam nhieu.
   - I1: kernel 3x3 (1/9), padding=1   -> giu nguyen kich thuoc
   - I2: kernel 5x5 (1/25), padding=2  -> giu nguyen kich thuoc
   - I3: kernel 7x7 (1/49), padding=3, stride=2  -> giam kich thuoc

2. LOC TRUNG VI (MEDIAN FILTER):
   - Sap xep pixel lan can, lay gia tri o giua.
   - Loai bo nhieu muoi tieu (salt-and-pepper) hieu qua.
   - I4 = median(I3, cua so 3x3)
   - I5 = median(I1, cua so 5x5)

3. ANH NGUONG I6:
   - Neu I4(x,y) > I5(x,y): I6(x,y) = 0 (den)
   - Nguoc lai: I6(x,y) = I5(x,y)
   - Lam noi bat vung I5 >= I4.
"""
    fig.text(0.08, 0.82, mo_ta_conv, fontsize=12,
             ha='left', va='top', color='#333333',
             fontfamily='monospace', linespacing=1.6)
    
    pdf.savefig(fig)
    plt.close(fig)
    
    # ---- Trang mô tả LBP ----
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor('#f5f5f5')
    
    fig.text(0.5, 0.92, 'NHIEM VU 3: LOCAL BINARY PATTERNS (LBP)', fontsize=22,
             fontweight='bold', ha='center', color='#e94560', fontfamily='monospace')
    
    mo_ta_lbp = """
1. THUAT TOAN LBP:
   - Chon P diem tren duong tron ban kinh R quanh pixel trung tam.
   - So sanh: lan can >= trung tam -> bit = 1, nguoc lai bit = 0.
   - Ghep P bit thanh chuoi nhi phan -> chuyen sang thap phan.
   - Dung noi suy bilinear khi toa do khong nguyen.

2. P = 8:
   - Chuoi nhi phan 8 bit -> 1 gia tri thap phan (0-255).

3. P = 16:
   - Chuoi nhi phan 16 bit -> tach 2 phan 8 bit.
   - Chuyen tung phan sang thap phan, lay MAX.
   - Ly do: 2^16 = 65536 > 255, khong hien thi duoc tren anh xam.

4. P = 24:
   - Chuoi nhi phan 24 bit -> tach 3 phan 8 bit.
   - Chuyen tung phan sang thap phan, lay MAX.
   - Ly do: 2^24 = 16 trieu > 255, phai chia nho de hien thi.

5. CAU HINH:
   P=8,R=1 | P=8,R=2 | P=16,R=2 | P=16,R=3 | P=24,R=3
"""
    fig.text(0.08, 0.82, mo_ta_lbp, fontsize=12,
             ha='left', va='top', color='#333333',
             fontfamily='monospace', linespacing=1.6)
    
    pdf.savefig(fig)
    plt.close(fig)


# =============================================================================
# HÀM 3: THÊM TRANG KẾT QUẢ HISTOGRAM CHO MỘT ẢNH
# =============================================================================
def them_trang_histogram(pdf, ten_anh, ket_qua_hist):
    """
    Thêm trang hiển thị kết quả histogram cho một ảnh.
    
    Hiển thị:
        - Ảnh xám gốc + H1
        - Ảnh sau cân bằng + H2
        - Ảnh sau thu hẹp [30,120] + H3
    
    Tham số:
        pdf: đối tượng PdfPages.
        ten_anh: tên ảnh gốc (string).
        ket_qua_hist: dict chứa đường dẫn các file kết quả.
    """
    fig, axes = plt.subplots(3, 2, figsize=(11.69, 8.27))
    fig.patch.set_facecolor('#ffffff')
    fig.suptitle(f'HISTOGRAM - {ten_anh}', fontsize=18, fontweight='bold',
                 color='#e94560', fontfamily='monospace', y=0.98)
    
    # Hàng 1: Ảnh xám + H1
    _hien_thi_anh(axes[0, 0], ket_qua_hist['duong_dan_anh_xam'], 'Anh xam goc')
    _hien_thi_anh(axes[0, 1], ket_qua_hist['duong_dan_h1'], 'Histogram H1')
    
    # Hàng 2: Ảnh cân bằng + H2
    _hien_thi_anh(axes[1, 0], ket_qua_hist['duong_dan_can_bang'], 'Sau can bang')
    _hien_thi_anh(axes[1, 1], ket_qua_hist['duong_dan_h2'], 'Histogram H2')
    
    # Hàng 3: Ảnh thu hẹp + H3
    _hien_thi_anh(axes[2, 0], ket_qua_hist['duong_dan_thu_hep'], 'Thu hep [30,120]')
    _hien_thi_anh(axes[2, 1], ket_qua_hist['duong_dan_h3'], 'Histogram thu hep')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


# =============================================================================
# HÀM 4: THÊM TRANG KẾT QUẢ TÍCH CHẬP CHO MỘT ẢNH
# =============================================================================
def them_trang_tich_chap(pdf, ten_anh, ket_qua_conv):
    """
    Thêm trang hiển thị kết quả tích chập và lọc trung vị.
    
    Hiển thị: I1, I2, I3, I4, I5, I6
    """
    fig, axes = plt.subplots(2, 3, figsize=(11.69, 8.27))
    fig.patch.set_facecolor('#ffffff')
    fig.suptitle(f'TICH CHAP & LOC TRUNG VI - {ten_anh}', fontsize=18,
                 fontweight='bold', color='#e94560', fontfamily='monospace', y=0.98)
    
    # Hàng 1: I1, I2, I3
    _hien_thi_anh(axes[0, 0], ket_qua_conv['duong_dan']['I1'], 'I1 (conv 3x3)')
    _hien_thi_anh(axes[0, 1], ket_qua_conv['duong_dan']['I2'], 'I2 (conv 5x5)')
    _hien_thi_anh(axes[0, 2], ket_qua_conv['duong_dan']['I3'], 'I3 (conv 7x7, s=2)')
    
    # Hàng 2: I4, I5, I6
    _hien_thi_anh(axes[1, 0], ket_qua_conv['duong_dan']['I4'], 'I4 (median I3)')
    _hien_thi_anh(axes[1, 1], ket_qua_conv['duong_dan']['I5'], 'I5 (median I1)')
    _hien_thi_anh(axes[1, 2], ket_qua_conv['duong_dan']['I6'], 'I6 (nguong)')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


# =============================================================================
# HÀM 5: THÊM TRANG KẾT QUẢ LBP CHO MỘT ẢNH
# =============================================================================
def them_trang_lbp(pdf, ten_anh, ket_qua_lbp, anh_xam_path):
    """
    Thêm trang hiển thị kết quả LBP cho một ảnh.
    
    Hiển thị: ảnh gốc + 5 cấu hình LBP.
    """
    fig, axes = plt.subplots(2, 3, figsize=(11.69, 8.27))
    fig.patch.set_facecolor('#ffffff')
    fig.suptitle(f'LOCAL BINARY PATTERNS - {ten_anh}', fontsize=18,
                 fontweight='bold', color='#e94560', fontfamily='monospace', y=0.98)
    
    # Ảnh gốc (xám)
    _hien_thi_anh(axes[0, 0], anh_xam_path, 'Anh xam goc')
    
    # 5 cấu hình LBP
    cau_hinh_keys = ['P8_R1', 'P8_R2', 'P16_R2', 'P16_R3', 'P24_R3']
    tieu_de = ['LBP P=8,R=1', 'LBP P=8,R=2', 'LBP P=16,R=2', 
               'LBP P=16,R=3', 'LBP P=24,R=3']
    vi_tri = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    
    for key, title, (r, c) in zip(cau_hinh_keys, tieu_de, vi_tri):
        if key in ket_qua_lbp:
            _hien_thi_anh(axes[r, c], ket_qua_lbp[key]['duong_dan'], title)
        else:
            axes[r, c].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


# =============================================================================
# HÀM 6: THÊM TRANG NHẬN XÉT
# =============================================================================
def them_trang_nhan_xet(pdf):
    """
    Thêm trang nhận xét tổng kết.
    """
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor('#f5f5f5')
    
    fig.text(0.5, 0.92, 'NHAN XET TONG KET', fontsize=22, fontweight='bold',
             ha='center', color='#e94560', fontfamily='monospace')
    
    nhan_xet = """
1. HISTOGRAM:
   - Can bang histogram giup tang do tuong phan ro ret, dac biet voi
     anh bi toi hoac bi sang qua muc.
   - Thu hep muc xam ve [30, 120] lam giam do tuong phan nhung huu ich
     khi can kiem soat dai muc xam dau ra.

2. TICH CHAP:
   - Kernel trung binh lam mo anh, giam nhieu nhung mat chi tiet.
   - Kernel lon hon (5x5, 7x7) lam mo nhieu hon kernel nho (3x3).
   - Stride = 2 giam kich thuoc anh di mot nua.

3. LOC TRUNG VI:
   - Hieu qua trong loai bo nhieu muoi tieu (salt-and-pepper).
   - Giu bien tot hon loc trung binh.
   - Cua so lon hon lam mo hon nhung loai nhieu tot hon.

4. LBP:
   - Ban kinh R lon hon nhan dien pattern o quy mo lon hon.
   - P lon hon cho chi tiet hon nhung tinh toan nhieu hon.
   - LBP la dac trung texture manh, ung dung trong nhan dang khuon mat,
     phan loai texture.

5. ANH I6:
   - Ket hop thong tin tu hai muc loc khac nhau.
   - Lam noi bat vung anh co gia tri loc trung vi I1 >= loc trung vi I3.
   - Co the dung de phat hien vung co dac diem khac biet.
"""
    fig.text(0.08, 0.82, nhan_xet, fontsize=12,
             ha='left', va='top', color='#333333',
             fontfamily='monospace', linespacing=1.6)
    
    pdf.savefig(fig)
    plt.close(fig)


# =============================================================================
# HÀM HỖ TRỢ: HIỂN THỊ ẢNH TRONG SUBPLOT
# =============================================================================
def _hien_thi_anh(ax, duong_dan, tieu_de):
    """
    Hiển thị ảnh trong một subplot của matplotlib.
    
    Tham số:
        ax: axes object của matplotlib.
        duong_dan: đường dẫn file ảnh (string).
        tieu_de: tiêu đề subplot (string).
    """
    try:
        anh = mpimg.imread(duong_dan)
        if len(anh.shape) == 2:
            ax.imshow(anh, cmap='gray')
        else:
            ax.imshow(anh)
    except Exception as e:
        ax.text(0.5, 0.5, f'Khong doc duoc\n{str(e)}',
                ha='center', va='center', fontsize=8, color='red')
    
    ax.set_title(tieu_de, fontsize=10, fontweight='bold', fontfamily='monospace')
    ax.axis('off')


# =============================================================================
# HÀM 7: TẠO BÁO CÁO PDF HOÀN CHỈNH
# =============================================================================
def tao_bao_cao_pdf(duong_dan_pdf, ket_qua_tat_ca):
    """
    Tạo báo cáo PDF hoàn chỉnh từ kết quả xử lý tất cả ảnh.
    
    Tham số:
        duong_dan_pdf: đường dẫn file PDF đầu ra (string).
        ket_qua_tat_ca: list các dict, mỗi dict chứa kết quả cho 1 ảnh.
            Mỗi dict có:
                - 'ten_anh': tên file ảnh gốc
                - 'histogram': dict kết quả histogram
                - 'convolution': dict kết quả tích chập
                - 'lbp': dict kết quả LBP
    """
    print(f"\n{'='*60}")
    print(f"DANG TAO BAO CAO PDF: {duong_dan_pdf}")
    print(f"{'='*60}")
    
    with PdfPages(duong_dan_pdf) as pdf:
        # 1. Trang bìa
        them_trang_tieu_de(pdf)
        print("  [PDF] Da tao trang bia.")
        
        # 2. Trang mô tả thuật toán
        them_trang_mo_ta_thuat_toan(pdf)
        print("  [PDF] Da tao trang mo ta thuat toan.")
        
        # 3. Kết quả cho từng ảnh
        for i, ket_qua in enumerate(ket_qua_tat_ca):
            ten_anh = ket_qua['ten_anh']
            print(f"  [PDF] Dang them ket qua cho {ten_anh} ({i+1}/{len(ket_qua_tat_ca)})...")
            
            # Trang histogram
            if 'histogram' in ket_qua and ket_qua['histogram'] is not None:
                them_trang_histogram(pdf, ten_anh, ket_qua['histogram'])
            
            # Trang tích chập
            if 'convolution' in ket_qua and ket_qua['convolution'] is not None:
                them_trang_tich_chap(pdf, ten_anh, ket_qua['convolution'])
            
            # Trang LBP
            if 'lbp' in ket_qua and ket_qua['lbp'] is not None:
                anh_xam_path = ket_qua['histogram']['duong_dan_anh_xam']
                them_trang_lbp(pdf, ten_anh, ket_qua['lbp'], anh_xam_path)
        
        # 4. Trang nhận xét
        them_trang_nhan_xet(pdf)
        print("  [PDF] Da tao trang nhan xet.")
    
    print(f"\n>>> BAO CAO PDF DA DUOC TAO THANH CONG!")
    print(f">>> Duong dan: {duong_dan_pdf}")
