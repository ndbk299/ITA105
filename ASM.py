import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Seed để tái lập kết quả
random.seed(42)
np.random.seed(42)

# Các giá trị giả lập
districts = ["Cầu Giấy", "Thanh Xuân", "Hoàn Kiếm", "Nam Từ Liêm", "Tây Hồ",
             "Đống Đa", "Ba Đình", "Hà Đông", "Long Biên", "Hoàng Mai"]
conditions = ["mới", "đã qua sử dụng", "cần sửa"]
facilities = [["siêu thị", "trường học"], ["bệnh viện"], ["trường học", "bệnh viện"],
              ["siêu thị"], ["trường học", "siêu thị"], ["bệnh viện"],
              ["trường học", "bệnh viện", "siêu thị"], ["siêu thị"],
              ["trường học"], ["siêu thị", "bệnh viện"]]

descriptions = [
    "Căn hộ mới xây, gần Vincom, nội thất cao cấp",
    "Nhà nhỏ, tiện nghi, gần bến xe",
    "Nhà phố cổ, vị trí trung tâm, cần cải tạo",
    "Căn hộ chung cư, view sân vận động Mỹ Đình",
    "Biệt thự mini, gần Hồ Tây, sang trọng",
    "Nhà tập thể, gần ga Hà Nội",
    "Biệt thự cao cấp, gần Lăng Bác, nội thất gỗ",
    "Nhà cấp 4, giá rẻ, cần cải tạo",
    "Nhà liền kề, gần cầu Chương Dương",
    "Chung cư giá rẻ, gần hồ Linh Đàm"
]

# Sinh dữ liệu
n = 10
data = {
    "id": range(1, n+1),
    "price": [round(random.uniform(1.5e9, 7e9), -6) for _ in range(n)],
    "area_m2": [random.randint(35, 150) for _ in range(n)],
    "num_rooms": [random.randint(1, 4) for _ in range(n)],
    "num_bathrooms": [random.randint(1, 3) for _ in range(n)],
    "location_district": random.choices(districts, k=n),
    "house_condition": random.choices(conditions, k=n),
    "transaction_date": [(datetime(2025,1,1) + timedelta(days=random.randint(0,365))).date() for _ in range(n)],
    "description": random.choices(descriptions, k=n),
    "image_path": [f"images/house_{i:03d}.jpg" for i in range(1, n+1)],
    "year_built": [random.randint(1985, 2024) for _ in range(n)],
    "floor": [random.randint(1, 20) for _ in range(n)],
    "has_parking": [random.choice([True, False]) for _ in range(n)],
    "nearby_facilities": random.choices(facilities, k=n),
    "drone_image_path": [f"drone/house_{i:03d}.jpg" for i in range(1, n+1)]
}

df = pd.DataFrame(data)

print(df.head())

# Giai đoạn 1 Hiểu dữ liệu thực tế, phát hiện vấn đề, làm sạch, chuẩn hóa, xử lý dữ liệu rời rạc và text.

# 1. Khám phá dữ liệu đa dạng
# - Phân tích thống kê: mean, median, std, min, max, missing values, duplicate.
print("\n--- Thống kê mô tả ---")
print(df.describe())
print("\n--- Thống kê phân loại ---")
print(df['location_district'].value_counts())
print(df['house_condition'].value_counts())
print(df['num_rooms'].value_counts())
print(df['num_bathrooms'].value_counts())
print(df['has_parking'].value_counts())
print(df['nearby_facilities'].apply(lambda x: ','.join(x) if isinstance(x, list) else x).value_counts())
print("\n--- Kiểm tra giá trị thiếu ---")
print(df.isnull().sum())
print("\n--- Kiểm tra giá trị trùng lặp ---")
print(df.duplicated().sum())

# - Vẽ biểu đồ histogram, boxplot, violin plot cho numerical.
import matplotlib.pyplot as plt
import seaborn as sns

numerical_cols = ['price', 'area_m2', 'num_rooms', 'num_bathrooms', 'year_built', 'floor']
for col in numerical_cols:
    plt.figure(figsize=(12, 5)) 
    plt.subplot(1, 3, 1)
    sns.histplot(df[col], kde=True)
    plt.title(f'Histogram of {col}')
    plt.subplot(1, 3, 2)
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.subplot(1, 3, 3)
    sns.violinplot(x=df[col])
    plt.title(f'Violin Plot of {col}')
    plt.show()

# - Phân tích phân phối categorical và text.
categorical_cols = ['location_district', 'house_condition', 'num_rooms', 'num_bathrooms', 'has_parking']
for col in categorical_cols:    
    plt.figure(figsize=(8, 4))
    sns.countplot(x=df[col])
    plt.title(f'Count Plot of {col}')
    plt.xticks(rotation=45)
    plt.show()

# 2. Xử lý dữ liệu bẩn
# - Điền missing values tùy loại cột (mean/median/mode/forward/backward).
# Giả sử có dữ liệu thiếu (điền mẫu cho các cột quan trọng)
df['price'] = df['price'].fillna(df['price'].median())
df['area_m2'] = df['area_m2'].fillna(df['area_m2'].mean())
df['num_rooms'] = df['num_rooms'].fillna(df['num_rooms'].mode()[0])
df['location_district'] = df['location_district'].fillna(method='ffill')
df['house_condition'] = df['house_condition'].fillna("không xác định")
# Xử lý cột list (nearby_facilities)
df['nearby_facilities'] = df['nearby_facilities'].apply(lambda x: x if isinstance(x, list) else [])

# - Xử lý dữ liệu không hợp lệ: giá âm, số phòng = 0, typo trong categorical.
df = df[df['price'] > 0]
df = df[df['area_m2'] > 20]
df = df[df['num_rooms'] > 0]
df = df[df['num_bathrooms'] > 0]
df = df[df['year_built'] >= 1900]
df = df[df['floor'] > 0]

# 3. Chuẩn hóa dữ liệu (Scaling)
from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler_minmax = MinMaxScaler()
df['price_minmax'] = scaler_minmax.fit_transform(df[['price']])

scaler_zscore = StandardScaler()
df['area_zscore'] = scaler_zscore.fit_transform(df[['area_m2']])

# - Loại bỏ hoặc merge duplicate records.
df = df.drop_duplicates()
print("\n--- Dữ liệu sau khi làm sạch ---")
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.head())
