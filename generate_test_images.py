"""
Script tạo 10 ảnh mẫu để test chương trình.
Chạy: python generate_test_images.py
"""
import numpy as np
import cv2
import os

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input_images")
os.makedirs(output_dir, exist_ok=True)

np.random.seed(42)

for i in range(1, 11):
    # Tạo ảnh màu ngẫu nhiên với kích thước nhỏ để chạy nhanh
    h, w = np.random.randint(100, 201), np.random.randint(100, 201)
    
    if i <= 3:
        # Ảnh gradient
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for c in range(3):
            base = np.random.randint(30, 200)
            grad = np.linspace(0, base, w).astype(np.uint8)
            img[:, :, c] = np.tile(grad, (h, 1))
        # Thêm nhiễu
        noise = np.random.randint(-20, 20, (h, w, 3))
        img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
    elif i <= 6:
        # Ảnh pattern hình học
        img = np.random.randint(40, 180, (h, w, 3), dtype=np.uint8)
        # Vẽ hình chữ nhật
        cx, cy = w // 2, h // 2
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        cv2.rectangle(img, (cx - 30, cy - 30), (cx + 30, cy + 30), color, -1)
        cv2.circle(img, (cx, cy), 20, (255, 255, 255), -1)
    else:
        # Ảnh ngẫu nhiên với vùng sáng/tối
        img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        # Thêm vùng tối ở góc
        img[:h//3, :w//3] = img[:h//3, :w//3] // 4
        # Thêm vùng sáng ở giữa
        img[h//3:2*h//3, w//3:2*w//3] = np.clip(
            img[h//3:2*h//3, w//3:2*w//3].astype(int) + 100, 0, 255
        ).astype(np.uint8)
    
    path = os.path.join(output_dir, f"image{i}.jpg")
    cv2.imwrite(path, img)
    print(f"Da tao: {path} ({h}x{w})")

print(f"\nDa tao 10 anh mau trong: {output_dir}")
