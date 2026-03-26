import pandas as pd

# Đọc dữ liệu từ file CSV
df = pd.read_csv("ITA105_Lab_1.csv")

# I.Khám phá dữ liệu

# 1. Kiểm tra kích thước dữ liệu
print("Kích thước dữ liệu:", df.shape)  # (số dòng, số cột)

# 2. Thống kê mô tả cho các cột số
print("\nThống kê mô tả:")
print(df.describe())

# 3. Kiểm tra giá trị thiếu trong các cột
print("\nGiá trị thiếu theo từng cột:")
print(df.isnull().sum())

# 4. Kiểm tra các giá trị bất thường (ví dụ StockQuantity âm)
print("\nCác dòng có StockQuantity âm:")
print(df[df["StockQuantity"] < 0])

# II.Xử lý dữ liệu thiếu

# 1. Phát hiện giá trị thiếu
print("Giá trị thiếu theo từng cột:")
print(df.isnull().sum())

# 2. Điền giá trị thiếu bằng mean/median/mode
# Điền bằng mean cho Price và StockQuantity
df_mean = df.copy()
df_mean["Price"] = df_mean["Price"].fillna(df_mean["Price"].mean())
df_mean["StockQuantity"] = df_mean["StockQuantity"].fillna(df_mean["StockQuantity"].mean())

# Điền bằng median cho Price và StockQuantity
df_median = df.copy()
df_median["Price"] = df_median["Price"].fillna(df_median["Price"].median())
df_median["StockQuantity"] = df_median["StockQuantity"].fillna(df_median["StockQuantity"].median())

# Điền bằng mode cho Category
df_mode = df.copy()
df_mode["Category"] = df_mode["Category"].fillna(df_mode["Category"].mode()[0])

print("\nSau khi điền bằng mean:")
print(df_mean.isnull().sum())

print("\nSau khi điền bằng median:")
print(df_median.isnull().sum())

print("\nSau khi điền bằng mode:")
print(df_mode.isnull().sum())

# 3. So sánh với phương pháp dropna()
df_dropna = df.dropna()
print("\nKích thước dữ liệu gốc:", df.shape)
print("Kích thước sau khi dropna:", df_dropna.shape)

# III.Xử lý dữ liệu lỗi
# 1. Kiểm tra giá trị bất hợp lý trong Price và StockQuantity
print("Giá trị Price nhỏ nhất:", df["Price"].min())
print("Giá trị StockQuantity nhỏ nhất:", df["StockQuantity"].min())

# Lọc các dòng có Price <= 0 hoặc StockQuantity < 0
invalid_price = df[df["Price"] <= 0]
invalid_stock = df[df["StockQuantity"] < 0]

print("\nCác dòng có Price không hợp lệ:")
print(invalid_price)

print("\nCác dòng có StockQuantity không hợp lệ:")
print(invalid_stock)

# Xử lý: thay thế giá trị bất hợp lý bằng NaN để sau đó điền giá trị hợp lý
df.loc[df["Price"] <= 0, "Price"] = None
df.loc[df["StockQuantity"] < 0, "StockQuantity"] = None

# 2. Lọc các giá trị không hợp lệ trong Rating
print("\nPhân bố Rating:")
print(df["Rating"].value_counts())

# Rating hợp lệ chỉ từ 1 đến 5 (giả sử thang điểm 1–5)
invalid_rating = df[~df["Rating"].between(1, 5)]
print("\nCác dòng có Rating không hợp lệ:")
print(invalid_rating)

# Xử lý: thay thế các giá trị Rating ngoài khoảng 1–5 bằng NaN
df.loc[~df["Rating"].between(1, 5), "Rating"] = None

# 3. Kiểm tra lại dữ liệu sau khi xử lý
print("\nGiá trị thiếu sau khi xử lý:")
print(df.isnull().sum())

# IV.Làm mượt dữ liệu nhiễu

import matplotlib.pyplot as plt

# 1. Áp dụng Moving Average (cửa sổ = 5)
df["Price_MA"] = df["Price"].rolling(window=5, min_periods=1).mean()

# 2. Vẽ biểu đồ line trước và sau khi làm mượt
plt.figure(figsize=(12,6))

# Biểu đồ gốc
plt.plot(df["ProductID"], df["Price"], label="Giá gốc", color="blue", alpha=0.6)

# Biểu đồ sau khi làm mượt
plt.plot(df["ProductID"], df["Price_MA"], label="Giá sau Moving Average", color="red", linewidth=2)

plt.title("Biểu đồ Price trước và sau khi làm mượt (Moving Average)")
plt.xlabel("ProductID")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()

# V.Chuyển hóa dữ liệu

# 1. Chuyển tất cả giá trị trong cột Category thành chữ thường
df["Category"] = df["Category"].str.lower()

# 2. Loại bỏ ký tự thừa trong cột Description
# Ví dụ: bỏ dấu "!" hoặc "?" lặp lại
df["Description"] = df["Description"].str.replace(r"[!?]+", "", regex=True).str.strip()

# 3. Chuyển đổi đơn vị giá từ USD sang VND
# Giả sử tỷ giá 1 USD = 25,000 VND
usd_to_vnd = 25000
df["Price_VND"] = df["Price"] * usd_to_vnd

# Kiểm tra kết quả
print(df[["Category", "Description", "Price", "Price_VND"]].head())