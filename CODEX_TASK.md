Bạn là coding agent Python chuyên xử lý ảnh. Tôi có một project bài tập cuối kỳ đang làm dở vì agent trước bị hết token giữa chừng.

Project nằm tại:

D:\XLA\image_processing_final

Hãy chỉ làm việc trong thư mục này. Trước khi sửa, hãy kiểm tra đúng thư mục bằng các lệnh như `dir`, `cd image_processing_final` nếu cần. Không sửa nhầm thư mục `D:\XLA` bên ngoài.

Mục tiêu của bạn:

* Đọc toàn bộ project hiện có.
* Kiểm tra file nào đã có, file nào thiếu, phần nào sai.
* Không viết lại từ đầu nếu không cần thiết.
* Sửa hoặc bổ sung để project chạy được bằng:

```bash
python main.py
```

* Code cần đơn giản, dễ hiểu, có comment tiếng Việt để sinh viên có thể học vấn đáp.

Bài tập gốc:

Có 10 ảnh màu kích thước n x m. Chương trình cần đọc ảnh từ thư mục `input_images/`, xử lý từng ảnh, lưu kết quả vào `output/`, và tạo báo cáo PDF.

Thông tin nhóm cần đưa vào báo cáo PDF:

* Nhóm 10
* 23133054 - Nguyễn Hồ Phát
* 23133061 - Phan Trọng Quí
* 23133056 - Phan Trọng Phú
* 23133035 - Trần Minh Khánh

Yêu cầu chi tiết:

1. Xử lý ảnh đầu vào

* Đọc toàn bộ ảnh trong thư mục `input_images/`.
* Hỗ trợ định dạng `.jpg`, `.jpeg`, `.png`, `.bmp`.
* Mỗi ảnh màu I phải được chuyển sang ảnh xám trước khi xử lý.
* Nếu thiếu thư mục `input_images/` hoặc không có ảnh, chương trình phải báo lỗi rõ ràng.
* Không được xóa hoặc ghi đè ảnh gốc.

2. Phần Histogram

Với mỗi ảnh xám:

* Vẽ histogram ảnh xám ban đầu, gọi là H1.
* Cân bằng histogram ảnh xám, tạo ảnh cân bằng H2.
* Vẽ histogram của ảnh sau cân bằng H2.
* Hiệu chỉnh hoặc thu hẹp mức xám của ảnh H2 vào khoảng `[30, 120]`.
* Vẽ histogram sau khi thu hẹp.
* Lưu các kết quả gồm:

  * Ảnh xám ban đầu.
  * Histogram H1.
  * Ảnh sau cân bằng histogram.
  * Histogram H2.
  * Ảnh sau khi thu hẹp mức xám về `[30, 120]`.
  * Histogram sau khi thu hẹp.

3. Phần Convolution, Median Filter và tạo I6

Với mỗi ảnh xám:

* Tạo I1 bằng convolution:

  * Kernel trung bình 3x3.
  * Mỗi phần tử kernel = 1/9.
  * Padding = 1.
  * Stride = 1.

* Tạo I2 bằng convolution:

  * Kernel trung bình 5x5.
  * Mỗi phần tử kernel = 1/25.
  * Padding = 2.
  * Stride = 1.

* Tạo I3 bằng convolution:

  * Kernel trung bình 7x7.
  * Mỗi phần tử kernel = 1/49.
  * Padding = 3.
  * Stride = 2 theo cả chiều ngang và chiều dọc.

* Tạo I4:

  * Lọc trung vị ảnh I3 với lân cận 3x3.

* Tạo I5:

  * Lọc trung vị ảnh I1 với lân cận 5x5.

* Tạo I6 theo công thức:

```text
Nếu I4(x, y) > I5(x, y) thì I6(x, y) = 0
Ngược lại I6(x, y) = I5(x, y)
```

* Vì I4 và I5 có thể khác kích thước, phải xử lý để chúng cùng kích thước trước khi tạo I6.
* Ưu tiên padding để làm cùng kích thước. Nếu dùng resize thì phải giải thích rõ.
* Lưu I1, I2, I3, I4, I5, I6 vào thư mục output phù hợp.

4. Phần LBP — Local Binary Patterns

Với mỗi ảnh xám, tính LBP cho đủ 5 trường hợp:

* P = 8, R = 1.
* P = 8, R = 2.
* P = 16, R = 2.
* P = 16, R = 3.
* P = 24, R = 3.

Yêu cầu tính LBP:

* Ưu tiên viết hàm LBP thủ công, không chỉ gọi sẵn `skimage.feature.local_binary_pattern`, để sinh viên có thể giải thích khi vấn đáp.
* Với mỗi pixel trung tâm:

  * So sánh pixel lân cận với pixel trung tâm.
  * Nếu pixel lân cận >= pixel trung tâm thì bit = 1.
  * Ngược lại bit = 0.

Với P = 8:

* Tạo chuỗi nhị phân 8 bit.
* Đổi chuỗi 8 bit sang số thập phân.
* Gán giá trị đó cho pixel trung tâm.

Với P = 16:

* Tạo chuỗi nhị phân 16 bit.
* Tách thành 2 nhóm, mỗi nhóm 8 bit.
* Đổi từng nhóm 8 bit sang số thập phân.
* Lấy giá trị lớn nhất trong 2 nhóm để gán cho pixel trung tâm.

Với P = 24:

* Tạo chuỗi nhị phân 24 bit.
* Tách thành 3 nhóm, mỗi nhóm 8 bit.
* Đổi từng nhóm 8 bit sang số thập phân.
* Lấy giá trị lớn nhất trong 3 nhóm để gán cho pixel trung tâm.

Nếu điểm lân cận không nằm đúng tọa độ pixel nguyên, có thể dùng bilinear interpolation. Nếu dùng cách khác đơn giản hơn, phải đảm bảo logic đúng và giải thích rõ.

Lưu ảnh LBP tương ứng với từng cặp P, R.

5. Báo cáo PDF

Cần tự động tạo file:

```text
output/report.pdf
```

Báo cáo PDF cần có:

* Tên nhóm.
* MSSV và họ tên sinh viên.
* Mô tả ngắn gọn các thuật toán:

  * Chuyển ảnh xám.
  * Histogram.
  * Cân bằng histogram.
  * Thu hẹp mức xám về `[30, 120]`.
  * Convolution.
  * Padding.
  * Stride.
  * Median filter.
  * LBP.
* Kết quả xử lý cho từng ảnh.
* Ảnh minh họa trước/sau.
* Histogram tương ứng.
* Nhận xét ngắn gọn cho từng nhóm kết quả.
* Không cần báo cáo quá đẹp, nhưng phải đầy đủ, rõ ràng và mở được.

6. Cấu trúc project mong muốn

Nếu project hiện tại đã có cấu trúc khác nhưng hợp lý, có thể giữ lại. Tuy nhiên cấu trúc ưu tiên là:

```text
image_processing_final/
├── input_images/
├── output/
│   ├── histograms/
│   ├── convolution/
│   ├── lbp/
│   └── report.pdf
├── main.py
├── histogram_processing.py
├── convolution_processing.py
├── lbp_processing.py
├── report_generator.py
├── requirements.txt
└── README.md
```

7. Quy trình làm việc bắt buộc

Bước 1: Kiểm tra project

Trước khi sửa code, hãy đọc các file hiện có và lập bảng ngắn gồm:

* File hoặc chức năng.
* Đã có chưa.
* Đúng yêu cầu chưa.
* Thiếu hoặc sai gì.
* Cần sửa gì.

Cần kiểm tra tối thiểu:

* `main.py` có điều phối toàn bộ pipeline chưa.
* Có đọc ảnh từ `input_images/` chưa.
* Có chuyển ảnh màu sang ảnh xám chưa.
* Có xử lý histogram H1, H2, thu hẹp `[30,120]` chưa.
* Có tạo I1, I2, I3 đúng kernel, padding, stride chưa.
* Có tạo I4, I5 bằng median filter chưa.
* Có xử lý khác kích thước I4/I5 trước khi tạo I6 chưa.
* Có tạo I6 đúng công thức chưa.
* Có LBP đủ 5 trường hợp chưa.
* Với P=16, P=24 có tách chuỗi thành nhóm 8 bit rồi lấy max chưa.
* Có lưu kết quả rõ ràng chưa.
* Có tạo `output/report.pdf` chưa.
* Có `requirements.txt` chưa.
* Có `README.md` chưa.

Bước 2: Chạy thử project

Hãy chạy:

```bash
python main.py
```

Nếu thiếu thư viện, hãy tạo hoặc sửa `requirements.txt`.

Có thể dùng các thư viện sau nếu cần:

```text
opencv-python
numpy
matplotlib
Pillow
reportlab
scikit-image
```

Chỉ thêm thư viện thật sự cần thiết.

Bước 3: Sửa lỗi và bổ sung

Khi sửa:

* Không viết lại toàn bộ nếu phần cũ đã đúng.
* Không xóa code đúng.
* Nếu một file sai nhiều, có thể thay thế file đó bằng bản hoàn chỉnh.
* Nếu thiếu file, hãy tạo file mới.
* Nếu đổi cấu trúc, phải giải thích lý do.
* Không làm code quá phức tạp.
* Ưu tiên dễ hiểu hơn tối ưu hiệu năng.

Bước 4: Kiểm tra sau khi sửa

Sau khi sửa, hãy chạy lại:

```bash
python main.py
```

Đảm bảo:

* Chương trình không lỗi.
* Tự tạo thư mục `output/` nếu chưa có.
* Lưu đầy đủ ảnh kết quả.
* Tạo được `output/report.pdf`.
* File PDF mở được.
* README có hướng dẫn chạy.

Bước 5: Báo cáo lại cho tôi

Sau khi hoàn thành, hãy trả lời theo cấu trúc:

1. Đã kiểm tra thấy project ban đầu thiếu/sai gì.

2. Đã sửa hoặc thêm những file nào.

3. Cách chạy project.

4. Kết quả đầu ra nằm ở đâu.

5. Các lưu ý khi vấn đáp.

6. Câu hỏi vấn đáp cần thêm vào README hoặc phần giải thích

Hãy thêm phần ôn vấn đáp gồm các câu ngắn gọn:

* Histogram là gì?
* Cân bằng histogram dùng để làm gì?
* Vì sao cần thu hẹp mức xám về `[30, 120]`?
* Padding là gì?
* Stride là gì?
* Kernel là gì?
* Convolution hoạt động như thế nào?
* I1, I2, I3 khác nhau ở đâu?
* Median filter là gì?
* I4 và I5 được tạo như thế nào?
* Vì sao I4 và I5 có thể khác kích thước?
* I6 được tạo theo quy tắc nào?
* LBP là gì?
* P và R trong LBP là gì?
* Vì sao P=16 phải tách thành 2 nhóm 8 bit?
* Vì sao P=24 phải tách thành 3 nhóm 8 bit?
* Bilinear interpolation trong LBP dùng để làm gì?
* Pipeline xử lý một ảnh diễn ra như thế nào?

9. Nếu gần hết token

Nếu bị giới hạn token, không được dừng giữa chừng một file code. Hãy hoàn thành trọn vẹn từng file theo thứ tự ưu tiên:

1. `main.py`
2. `histogram_processing.py`
3. `convolution_processing.py`
4. `lbp_processing.py`
5. `report_generator.py`
6. `requirements.txt`
7. `README.md`

Nếu phải dừng, hãy ghi rõ:

```text
Phần đã hoàn thành:
...

Phần tiếp theo cần làm:
...
```

Yêu cầu quan trọng cuối cùng:

* Không bỏ qua report PDF.
* Không bỏ qua LBP P=16 và P=24.
* Không bỏ qua xử lý khác kích thước giữa I4 và I5.
* Không bỏ qua kiểm tra chạy `python main.py`.
* Code phải chạy được và dễ giải thích khi vấn đáp.
