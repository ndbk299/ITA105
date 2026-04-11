import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Bài 1: Doanh thu siêu thị
print("\n===== BÀI 1 =====")

Supermarket = pd.read_csv(r'D:/FPT/ITA105/Lab5/ITA105_Lab_5_Supermarket.csv')

# - Chuyển cột ngày về datetime, đặt index.
Supermarket['date'] = pd.to_datetime(Supermarket['date'], format='mixed', dayfirst=True, errors='coerce')
Supermarket.set_index('date', inplace=True)
print(Supermarket.head())

# - Kiểm tra missing values, điền bằng Forward Fill / Backward Fill / Interpolation.
print(Supermarket.isnull().sum())
Supermarket['revenue'] = Supermarket['revenue'].interpolate(method='linear')
print(Supermarket.isnull().sum())

# - Tạo đặc trưng: năm, tháng, quý, ngày trong tuần, weekend/weekday.
Supermarket["year"]= Supermarket.index.year
Supermarket["month"] = Supermarket.index.month
Supermarket["quarter"] = Supermarket.index.quarter
Supermarket["day_of_week"] = Supermarket.index.dayofweek
Supermarket["is_weekend"] = Supermarket["day_of_week"].isin([5, 6]).astype(int)
Supermarket_monthly = Supermarket['revenue'].resample('ME').sum().to_frame()
Supermarket_monthly.head()

# - Vẽ biểu đồ tổng doanh thu theo tháng, tuần.
plt.figure(figsize=(12, 6))
plt.plot(Supermarket_monthly.index, Supermarket_monthly['revenue'], marker='o')
plt.title('Monthly Revenue')
plt.xlabel('Month')
plt.ylabel('Revenue')
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(Supermarket.index, Supermarket['revenue'], marker='o')
plt.title('Daily Revenue')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.show()

# - Phát hiện xu hướng dài hạn và mùa vụ (trend & seasonality) bằng rolling mean hoặc decomposition.
Supermarket_monthly['rolling_mean_3'] = Supermarket_monthly['revenue'].rolling(window=3).mean()
plt.figure(figsize=(12, 6))
plt.plot(Supermarket_monthly.index, Supermarket_monthly['revenue'], label='Revenue', marker='o')
plt.plot(Supermarket_monthly.index, Supermarket_monthly['rolling_mean_3'], label='Rolling Mean (3 months)', color='orange')
plt.title('Monthly Revenue with Rolling Mean')
plt.xlabel('Month')
plt.ylabel('Revenue')
plt.legend()
plt.show()

decomposition = seasonal_decompose(Supermarket_monthly['revenue'], model='additive', period=12)
plt.figure(figsize=(12, 8)) 
decomposition.plot()
plt.show()

# Bài 2: Lưu lượng truy cập website
print("\n===== BÀI 2 =====")

Website = pd.read_csv(r'D:/FPT/ITA105/Lab5/ITA105_Lab_5_Web_traffic.csv')

# - Đặt lại tần suất dữ liệu (hourly).
Website['datetime'] = pd.to_datetime(Website['datetime'], format='mixed', dayfirst=True, errors='coerce')
Website.set_index('datetime', inplace=True)
Website = Website.resample('h').mean()
print(Website.head())

# - Xử lý dữ liệu thiếu (nếu có) bằng nội suy tuyến tính.
print(Website.isnull().sum())
Website['visits'] = Website['visits'].interpolate(method='linear')
print(Website.isnull().sum())

# - Tạo biến đặc trưng: giờ trong ngày, ngày trong tuần.
Website['hour'] = Website.index.hour
Website['day_of_week'] = Website.index.dayofweek
print(Website.head())

# - Vẽ biểu đồ lưu lượng theo giờ, phát hiện xu hướng peak/trough trong ngày.
plt.figure(figsize=(12, 6))
plt.plot(Website.index, Website['visits'], marker='o')
plt.title('Hourly Visitors')
plt.xlabel('Hour')
plt.ylabel('Visitors')
plt.show()

# - Phân tích seasonality hàng ngày và hàng tuần.
Website_hourly = Website.resample('h').mean()
decomposition = seasonal_decompose(Website_hourly['visits'], model='additive', period=24)
plt.figure(figsize=(12, 8)) 
decomposition.plot()
plt.show()

# Bài 3: Giá cổ phiếu
print("\n===== BÀI 3 =====")

Stock = pd.read_csv(r'D:/FPT/ITA105/Lab5/ITA105_Lab_5_Stock.csv')

# - Chuyển cột ngày về datetime, đặt index.
Stock['date'] = pd.to_datetime(Stock['date'], format='mixed', dayfirst=True, errors='coerce')
Stock.set_index('date', inplace=True)
print(Stock.head())

# - Kiểm tra missing values (ngày nghỉ giao dịch), điền bằng forward fill.
print(Stock.isnull().sum())
Stock = Stock.ffill()
print(Stock.isnull().sum())

# - Vẽ biểu đồ line plot giá đóng cửa.
plt.figure(figsize=(12, 6))
plt.plot(Stock.index, Stock['close_price'], marker='o')
plt.title('Close Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# - Tạo rolling mean 7 ngày, 30 ngày để nhận diện trend.
Stock['rolling_mean_7'] = Stock['close_price'].rolling(window=7).mean()
Stock['rolling_mean_30'] = Stock['close_price'].rolling(window=30).mean()
plt.figure(figsize=(12, 6))
plt.plot(Stock.index, Stock['close_price'], label='Close Price', marker='o')
plt.plot(Stock.index, Stock['rolling_mean_7'], label='Rolling Mean (7 days)', color='orange')
plt.plot(Stock.index, Stock['rolling_mean_30'], label='Rolling Mean (30 days)', color='green')
plt.title('Close Price with Rolling Means')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# - Phân tích seasonality theo tháng, tìm pattern lặp lại.
Stock_monthly = Stock.resample('ME').mean()
decomposition = seasonal_decompose(Stock_monthly['close_price'], model='additive', period=12)
plt.figure(figsize=(12, 8)) 
decomposition.plot()
plt.show()

# Bài 4: Sản xuất công nghiệp
print("\n===== BÀI 4 =====")

Industrial = pd.read_csv(r'D:/FPT/ITA105/Lab5/ITA105_Lab_5_Production.csv')

# - Kiểm tra và điền missing values.
Industrial['week_start'] = pd.to_datetime(Industrial['week_start'], format='mixed', dayfirst=True, errors='coerce')
Industrial.set_index('week_start', inplace=True)
print(Industrial.isnull().sum())
Industrial['production'] = Industrial['production'].interpolate(method='linear')
print(Industrial.isnull().sum())

# - Tạo đặc trưng: tuần, quý, năm.
Industrial['week'] = Industrial.index.isocalendar().week
Industrial['quarter'] = Industrial.index.quarter
Industrial['year'] = Industrial.index.year
print(Industrial.head())

# - Phát hiện trend dài hạn bằng rolling mean.
Industrial['rolling_mean_12'] = Industrial['production'].rolling(window=12).mean()
plt.figure(figsize=(12, 6))
plt.plot(Industrial.index, Industrial['production'], label='Production', marker='o')
plt.plot(Industrial.index, Industrial['rolling_mean_12'], label='Rolling Mean (12 months)', color='orange')
plt.title('Industrial Production with Rolling Mean')
plt.xlabel('Week Start')
plt.ylabel('Production')
plt.legend()
plt.show()

# - Phân tích seasonality theo quý.
Industrial_quarterly = Industrial.resample('QE').mean()
decomposition = seasonal_decompose(Industrial_quarterly['production'], model='additive', period=4)
plt.figure(figsize=(12, 8)) 
decomposition.plot()
plt.show()

# - Vẽ biểu đồ decomposition (trend + seasonality + residuals) bằng statsmodels.tsa.seasonal_decompose.
decomposition = seasonal_decompose(Industrial['production'], model='additive', period=12)
plt.figure(figsize=(12, 8)) 
decomposition.plot()
plt.show()