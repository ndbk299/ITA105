# =========================================
# BÀI 1: DOANH THU SIÊU THỊ (FULL COLAB)
# =========================================

# =========================
# 1. IMPORT THƯ VIỆN
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score

from statsmodels.tsa.seasonal import seasonal_decompose

# =========================
# 2. TẠO DATASET GIẢ LẬP
# =========================
# Dataset gồm:
# - date: ngày
# - revenue: doanh thu

np.random.seed(42)

dates = pd.date_range(start="2023-01-01", end="2024-12-31", freq="D")

data = pd.DataFrame({
    "date": dates
})

# Tạo dữ liệu:
# trend: xu hướng tăng dần
# seasonality: chu kỳ theo tuần
# noise: nhiễu ngẫu nhiên
data["trend"] = np.linspace(100, 200, len(data))
data["seasonality"] = 20 * np.sin(2 * np.pi * data.index / 7)
data["noise"] = np.random.normal(0, 5, len(data))

data["revenue"] = data["trend"] + data["seasonality"] + data["noise"]

# Tạo missing values
missing_index = np.random.choice(data.index, 20)
data.loc[missing_index, "revenue"] = np.nan

# Giữ lại 2 cột chính
data = data[["date", "revenue"]]

print("=== DATA BAN ĐẦU ===")
print(data.head())


# =========================
# 3. TIỀN XỬ LÝ
# =========================

# Chuyển sang datetime và set index
data["date"] = pd.to_datetime(data["date"])
data.set_index("date", inplace=True)

# Xử lý missing values bằng Interpolation
data["revenue"] = data["revenue"].interpolate()

# =========================
# 4. FEATURE ENGINEERING
# =========================

# Tạo các đặc trưng thời gian
data["year"] = data.index.year
data["month"] = data.index.month
data["dayofweek"] = data.index.dayofweek  # 0 = Monday

# Weekend = Thứ 7, CN
data["is_weekend"] = data["dayofweek"].isin([5, 6]).astype(int)

print("\n=== DATA SAU KHI XỬ LÝ ===")
print(data.head())


# =========================
# 5. VẼ BIỂU ĐỒ
# =========================

# Doanh thu theo tháng
monthly = data["revenue"].resample("ME").sum()
monthly.plot(title="Doanh thu theo tháng", figsize=(10,5))
plt.show()

# Doanh thu theo tuần
weekly = data["revenue"].resample("W").sum()
weekly.plot(title="Doanh thu theo tuần", figsize=(10,5))
plt.show()


# =========================
# 6. PHÂN TÍCH TREND & SEASONALITY
# =========================

# Decomposition để tách:
# - Trend (xu hướng dài hạn)
# - Seasonal (chu kỳ)
# - Residual (nhiễu)
result = seasonal_decompose(data["revenue"], model="additive", period=7)
result.plot()
plt.show()


# =========================
# 7. LINEAR REGRESSION (DỰ ĐOÁN DOANH THU)
# =========================

features = ["year", "month", "dayofweek", "is_weekend"]

X = data[features]
y = data["revenue"]

# Chia train/test (giữ thứ tự thời gian)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Huấn luyện model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Dự đoán
y_pred = lr_model.predict(X_test)

# Đánh giá
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\n=== LINEAR REGRESSION ===")
print("MSE:", mse)
print("R-squared:", r2)

plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted', color='red')
plt.legend()
plt.show()


# =========================
# 8. CLASSIFICATION (LOGISTIC REGRESSION)
# =========================

# Tạo nhãn:
# 1 = doanh thu cao
# 0 = doanh thu thấp
threshold = data["revenue"].mean()
data["high_revenue"] = (data["revenue"] > threshold).astype(int)

X = data[features]
y = data["high_revenue"]

# Train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Model classification
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Dự đoán
y_pred = clf.predict(X_test)

# Đánh giá
acc = accuracy_score(y_test, y_pred)
print("\n=== LOGISTIC REGRESSION ===")
print("Accuracy:", acc)


# =========================
# 9. ROLLING MEAN (XU HƯỚNG DÀI HẠN)
# =========================

data["rolling_mean_30"] = data["revenue"].rolling(window=30).mean()

data[["revenue", "rolling_mean_30"]].plot(
    title="Xu hướng dài hạn (Rolling Mean 30 ngày)",
    figsize=(10,5)
)
plt.show()


# =========================
# 10. KẾT LUẬN
# =========================

print("""
KẾT LUẬN:

1. Linear Regression:
   - Dự đoán doanh thu (giá trị liên tục)
   - Phù hợp phân tích xu hướng

2. Logistic Regression:
   - Phân loại ngày doanh thu cao/thấp
   - Hỗ trợ ra quyết định kinh doanh

3. Time Series:
   - Seasonal Decomposition: tìm chu kỳ
   - Rolling Mean: xu hướng dài hạn
""")