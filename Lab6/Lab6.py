import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# Hàm hỗ trợ hiển thị ảnh
def show_images(images, titles, rows, cols, figsize=(15, 10)):
    plt.figure(figsize=figsize)
    for i in range(len(images)):
        plt.subplot(rows, cols, i + 1)
        # Nếu là ảnh xám (2 chiều), dùng cmap='gray'
        if len(images[i].shape) == 2:
            plt.imshow(images[i], cmap='gray')
        else:
            # OpenCV dùng BGR, Matplotlib dùng RGB
            plt.imshow(cv2.cvtColor(images[i].astype(np.float32), cv2.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# --- CÁC HÀM AUGMENTATION ---
def augment_pipeline(img, rotate_limit=15, brightness_limit=0.2, flip=True, noise=False, crop=False):
    aug_img = img.copy()
    
    # 1. Flip
    if flip and random.random() > 0.5:
        aug_img = cv2.flip(aug_img, 1)
        
    # 2. Rotation
    angle = random.uniform(-rotate_limit, rotate_limit)
    h, w = aug_img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    aug_img = cv2.warpAffine(aug_img, M, (w, h))
    
    # 3. Brightness
    # Chuyển sang HSV để thay đổi độ sáng (V channel)
    hsv = cv2.cvtColor(aug_img, cv2.COLOR_BGR2HSV)
    h_v, s_v, v_v = cv2.split(hsv)
    ratio = 1.0 + random.uniform(-brightness_limit, brightness_limit)
    v_v = np.clip(v_v * ratio, 0, 255).astype(np.uint8)
    aug_img = cv2.merge((h_v, s_v, v_v))
    aug_img = cv2.cvtColor(aug_img, cv2.COLOR_HSV2BGR)

    # 4. Gaussian Noise
    if noise:
        gauss = np.random.normal(0, 15, aug_img.shape).astype(np.uint8)
        aug_img = cv2.add(aug_img, gauss)

    # 5. Random Crop & Zoom
    if crop:
        start_x = random.randint(0, int(w * 0.1))
        start_y = random.randint(0, int(h * 0.1))
        end_x = w - random.randint(0, int(w * 0.1))
        end_y = h - random.randint(0, int(h * 0.1))
        aug_img = aug_img[start_y:end_y, start_x:end_x]
        aug_img = cv2.resize(aug_img, (w, h))

    return aug_img

# =================================================================
# Bài 1: Chuẩn bị ảnh căn hộ / mặt tiền
# =================================================================
print("\n===== BÀI 1 =====")
# Giả định đường dẫn ảnh, bạn hãy thay đổi cho đúng thư mục của mình
path_b1 = r'D:\FPT\ITA105\Lab6\data\apartments' 
if os.path.exists(path_b1):
    img_list = [os.path.join(path_b1, f) for f in os.listdir(path_b1)[:5]]
    originals = []
    augmented = []

    for p in img_list:
        img = cv2.imread(p)
        if img is None: continue
        
        # Resize
        img = cv2.resize(img, (224, 224))
        originals.append(img)
        
        # Augmentation
        aug = augment_pipeline(img, rotate_limit=15, brightness_limit=0.2, flip=True)
        
        # RGB -> Grayscale
        aug_gray = cv2.cvtColor(aug, cv2.COLOR_BGR2GRAY)
        
        # Chuẩn hóa [0-1]
        aug_norm = aug_gray / 255.0
        augmented.append(aug_norm)

    # Hiển thị 5 gốc và 5 đã aug
    show_images(originals + augmented, 
                ["Original"]*5 + ["Augmented Gray"]*5, 2, 5)
else:
    print("Đường dẫn Bài 1 không tồn tại.")

# =================================================================
# Bài 2: Chuẩn bị ảnh xe ô tô / xe máy
# =================================================================
print("\n===== BÀI 2 =====")
path_b2 = r'D:\FPT\ITA105\Lab6\data\vehicles'
if os.path.exists(path_b2):
    img_p = os.path.join(path_b2, os.listdir(path_b2)[0])
    img = cv2.imread(img_p)
    
    # Resize
    img = cv2.resize(img, (224, 224))
    
    # Augmentation: Gaussian noise, brightness ±15%, rotate ±10°
    aug = augment_pipeline(img, rotate_limit=10, brightness_limit=0.15, flip=False, noise=True)
    
    # Tùy chọn Grayscale
    aug_gray = cv2.cvtColor(aug, cv2.COLOR_BGR2GRAY)
    
    # Chuẩn hóa [0-1]
    aug_norm = aug_gray / 255.0
    print("Bài 2: Đã hoàn thành resize, noise, brightness, rotation và normalization.")
    plt.imshow(aug_norm, cmap='gray')
    plt.title("Bài 2: Vehicle Augmented")
    plt.show()

# =================================================================
# Bài 3: Chuẩn bị ảnh trái cây / nông sản
# =================================================================
print("\n===== BÀI 3 =====")
path_b3 = r'D:\FPT\ITA105\Lab6\data\fruits'
if os.path.exists(path_b3):
    img_p = os.path.join(path_b3, os.listdir(path_b3)[0])
    img = cv2.imread(img_p)
    img = cv2.resize(img, (224, 224))
    
    aug_grid = []
    for i in range(9):
        # Aug: flip, random crop, zoom, rotation
        aug = augment_pipeline(img, rotate_limit=30, flip=True, crop=True)
        # Chuẩn hóa [0-1]
        aug_norm = aug / 255.0
        aug_grid.append(aug_norm)
        
    # Hiển thị grid 3x3
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(cv2.cvtColor(aug_grid[i].astype(np.float32), cv2.COLOR_BGR2RGB))
        plt.axis('off')
    plt.suptitle("Bài 3: 3x3 Grid Augmentation Trái cây")
    plt.show()

# =================================================================
# Bài 4: Chuẩn bị ảnh phòng / nội thất
# =================================================================
print("\n===== BÀI 4 =====")
path_b4 = r'D:\FPT\ITA105\Lab6\data\interiors'
if os.path.exists(path_b4):
    img_p = os.path.join(path_b4, os.listdir(path_b4)[0])
    img = cv2.imread(img_p)
    img_res = cv2.resize(img, (224, 224))
    
    results = [img_res]
    titles = ["Original"]
    
    for i in range(3):
        # Aug: xoay ±15°, horizontal flip, brightness ±20%
        aug = augment_pipeline(img_res, rotate_limit=15, brightness_limit=0.2, flip=True)
        # RGB -> Gray
        aug_gray = cv2.cvtColor(aug, cv2.COLOR_BGR2GRAY)
        # Chuẩn hóa [0-1]
        aug_norm = aug_gray / 255.0
        results.append(aug_norm)
        titles.append(f"Augmented {i+1}")
        
    # Hiển thị ảnh gốc và 3 ảnh aug
    show_images(results, titles, 1, 4)

print("\n--- HOÀN THÀNH LAB 6 ---")

show_images()
augment_pipeline()
