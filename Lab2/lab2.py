import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Bài 1 Khám phá dữ liệu Housing
# 1 Nạp dữ liệu, kiểm tra shape, missing values.
housing_data = pd.read_csv(r"d:\FPT\ITA105\Lab2\ITA105_Lab_2_Housing.csv")
print (f"Shape: {housing_data.shape}")
print (f"Missing values:\n{housing_data.isnull().sum()}")

# 2 In thống kê mô tả (mean, median, std, min, max) và nhận xét sơ bộ về dữ liệu.
print (f"Descriptive statistics:\n{housing_data.describe()}")

# 3. Vẽ boxplot cho từng biến numeric và đánh dấu các điểm ngoại lệ.
numeric_cols = housing_data.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=housing_data[col])
    plt.title(f'Boxplot for {col}')
    plt.show()

# 4. Vẽ scatterplot diện tích và giá để phát hiện điểm lạc lõng.
plt.figure(figsize=(8, 6))
sns.scatterplot(x=housing_data["dien_tich"], y=housing_data["gia"])
plt.title("Scatterplot diện tích vs giá")
plt.show()

# 5. Tính IQR, xác định ngoại lệ theo công thức.
Q1=housing_data["gia"].quantile(0.25)
Q3=housing_data["gia"].quantile(0.75)
IQR=Q3-Q1
lower_bound=Q1-1.5*IQR
upper_bound=Q3+1.5*IQR
outliers=housing_data[(housing_data["gia"] < lower_bound) | (housing_data["gia"] > upper_bound)]
print (f"Outliers:\n{outliers}")

# 6. Tính Z-score cho từng biến numeric và xác định ngoại lệ (|Z| >3).
z_scores=stats.zscore(housing_data[numeric_cols])
outliers_z=housing_data[abs(z_scores)>3].dropna()
print (f"Outliers based on Z-score:\n{outliers_z}")

# 7. So sánh số lượng ngoại lệ phát hiện bằng IQR, Z-score và boxplot.
print (f"Number of outliers (IQR): {len(outliers)}")
print (f"Number of outliers (Z-score): {len(outliers_z)}")
print (f"Number of outliers (boxplot): {len(outliers)}")

# 8. Phân tích nguyên nhân: ngoại lệ do lỗi nhập liệu hay đặc thù thực tế?
print ("Phần lớn là lỗi nhập liệu hoặc dữ liệu test.")

# 9. Áp dụng một phương pháp xử lý ngoại lệ: giữ, loại bỏ hoặc điều chỉnh (clip / log-transform).
housing_data_clipped = housing_data.copy()
housing_data_clipped["gia"] = housing_data_clipped["gia"].clip(lower=lower_bound, upper=upper_bound)
print (f"Descriptive statistics after clipping:\n{housing_data_clipped.describe()}")

# 10. Vẽ lại boxplot sau xử lý và nhận xét ảnh hưởng.
plt.figure(figsize=(8, 6))
sns.boxplot(x=housing_data_clipped["gia"])
plt.title("Boxplot for gia after clipping")
plt.show()
print ("Boxplot sau khi clipping đã loại bỏ các điểm ngoại lệ, giúp dữ liệu trở nên cân đối hơn và dễ phân tích hơn.")  


# Bài 2 Phát hiện ngoại lệ trong dữ liệu IoT / Sensor
# 1. Load dữ liệu, set timestamp làm index, kiểm tra missing values.
iot_data = pd.read_csv(r"d:\FPT\ITA105\Lab2\ITA105_Lab_2_IoT.csv")
iot_data["timestamp"] = pd.to_datetime(iot_data["timestamp"])
iot_data.set_index("timestamp", inplace=True)
print (f"Missing values:\n{iot_data.isnull().sum()}")

# 2. Vẽ line plot temperature theo thời gian cho từng sensor.
plt.figure(figsize=(12, 6))
for sensor in iot_data["sensor_id"].unique():
    plt.plot(iot_data[iot_data["sensor_id"] == sensor].index, iot_data[iot_data["sensor_id"] == sensor]["temperature"], label=f"Sensor {sensor}")
plt.title("Temperature over time for each sensor")
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.legend()
plt.show()

# 3. Phát hiện ngoại lệ bằng rolling mean ± 3 × std (window = 10).
window_size = 10
for sensor in iot_data["sensor_id"].unique():
    sensor_data = iot_data[iot_data["sensor_id"] == sensor]["temperature"]
    rolling_mean = sensor_data.rolling(window=window_size).mean()
    rolling_std = sensor_data.rolling(window=window_size).std()
    upper_bound = rolling_mean + 3 * rolling_std
    lower_bound = rolling_mean - 3 * rolling_std
    outliers_rolling = sensor_data[(sensor_data > upper_bound) | (sensor_data < lower_bound)]
    print (f"Outliers for Sensor {sensor} based on rolling mean:\n{outliers_rolling}")

# 4. Tính Z-score cho từng sensor và xác định ngoại lệ (|Z| > 3).
for sensor in iot_data["sensor_id"].unique():
    sensor_data = iot_data[iot_data["sensor_id"] == sensor]["temperature"]
    z_scores = stats.zscore(sensor_data.dropna())
    outliers_z = sensor_data[abs(z_scores) > 3]
    print (f"Outliers for Sensor {sensor} based on Z-score:\n{outliers_z}")
    
# 5. Vẽ boxplot và scatter plot giữa các biến (temperature vs pressure, pressure vs humidity) highlight điểm bất thường.
plt.figure(figsize=(8, 6))
sns.boxplot(x=iot_data["temperature"])
plt.title("Boxplot for temperature")
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(x=iot_data["temperature"], y=iot_data["pressure"])
plt.title("Scatterplot for temperature vs pressure")
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(x=iot_data["pressure"], y=iot_data["humidity"])
plt.title("Scatterplot for pressure vs humidity")
plt.show()

# 6. So sánh số lượng ngoại lệ phát hiện bởi rolling mean, Z-score, box plot, scatter plot.
print (f"Number of outliers (rolling mean): {len(outliers_rolling)}")
print (f"Number of outliers (Z-score): {len(outliers_z)}")
print (f"Number of outliers (boxplot): {len(outliers_rolling)}")
print (f"Number of outliers (scatter plot): {len(outliers_rolling)}")

# 7. Xử lý ngoại lệ bằng interpolation hoặc clip giá trị và vẽ lại dữ liệu.
iot_data_interpolated = iot_data.copy()
iot_data_interpolated["temperature"] = iot_data_interpolated["temperature"].interpolate(method='linear')
plt.figure(figsize=(12, 6))
for sensor in iot_data_interpolated["sensor_id"].unique():
    plt.plot(iot_data_interpolated[iot_data_interpolated["sensor_id"] == sensor].index, iot_data_interpolated[iot_data_interpolated["sensor_id"] == sensor]["temperature"], label=f"Sensor {sensor}")
plt.title("Temperature over time for each sensor after interpolation")

#  Bài 3 Ngoại lệ trong giao dịch E-commerce
# 1. Load dữ liệu, kiểm tra missing values, thống kê mô tả.
ecommerce_data = pd.read_csv(r"d:\FPT\ITA105\Lab2\ITA105_Lab_2_Ecommerce.csv")
print (f"Missing values:\n{ecommerce_data.isnull().sum()}")
print (f"Descriptive statistics:\n{ecommerce_data.describe()}")

# 2. Vẽ boxplot cho price, quantity, rating.
plt.figure(figsize=(8, 6))
sns.boxplot(x=ecommerce_data["price"])
plt.title("Boxplot for price")
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x=ecommerce_data["quantity"])
plt.title("Boxplot for quantity")
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x=ecommerce_data["rating"])
plt.title("Boxplot for rating")
plt.show()

# 3. Tính IQR và Z-score cho các biến numeric, xác định ngoại lệ.
Q1_price=ecommerce_data["price"].quantile(0.25)
Q3_price=ecommerce_data["price"].quantile(0.75)
IQR_price=Q3_price-Q1_price
lower_bound_price=Q1_price-1.5*IQR_price
upper_bound_price=Q3_price+1.5*IQR_price
outliers_price=ecommerce_data[(ecommerce_data["price"] < lower_bound_price) | (ecommerce_data["price"] > upper_bound_price)]
print (f"Outliers for price:\n{outliers_price}")
z_scores_price=stats.zscore(ecommerce_data["price"].dropna())
outliers_z_price=ecommerce_data[abs(z_scores_price) > 3]
print (f"Outliers for price based on Z-score:\n{outliers_z_price}")


# 4. Vẽ scatterplot price vs quantity và đánh dấu các ngoại lệ.
plt.figure(figsize=(8, 6))
sns.scatterplot(x=ecommerce_data["price"], y=ecommerce_data["quantity"])
plt.title("Scatterplot for price vs quantity")
plt.show()

# 5. Phân tích nguyên nhân: giá trị 0, rating > 5, số lượng sản phẩm bất thường, category hiếm.
print ("Giá = 0 hoặc rating > 5 → lỗi nhập liệu.")
print ("Số lượng lớn (50, 100) → có thể là đơn bulk order.")

# 6. Thực hiện xử lý ngoại lệ:
# - Loại bỏ các giá trị do lỗi nhập liệu
# - Giữ các ngoại lệ hợp lý (ví dụ: sản phẩm premium, số lượng lớn thực tế)
# - Điều chỉnh giá trị lệch (clip / log-transform).
ecommerce_data_cleaned = ecommerce_data.copy()
ecommerce_data_cleaned = ecommerce_data_cleaned[(ecommerce_data_cleaned["price"] > 0) & (ecommerce_data_cleaned["rating"] <= 5)]
print (f"Descriptive statistics after cleaning:\n{ecommerce_data_cleaned.describe()}")

# 7. Vẽ lại box plot và scatter plot sau xử lý, nhận xét tác động.
plt.figure(figsize=(8, 6))
sns.boxplot(x=ecommerce_data_cleaned["price"])
plt.title("Boxplot for price after cleaning")
plt.show()

# Bài 4 Multivariate Outlier
# 1. Xác định các điểm ngoại lệ multivariate bằng cách kết hợp 2–3 biến:
# - Housing: diện tích + giá
sns.scatterplot(x=housing_data["dien_tich"], y=housing_data["gia"])
plt.title("Scatterplot diện tích vs giá")
plt.show()

# - IoT: nhiệt độ + độ ẩm
sns.scatterplot(x=iot_data["temperature"], y=iot_data["humidity"])
plt.title("Scatterplot temperature vs humidity")
plt.show()

# - E-commerce: giá + số lượng + rating
sns.scatterplot(x=ecommerce_data["price"], y=ecommerce_data["quantity"], hue=ecommerce_data["rating"])
plt.title("Scatterplot price vs quantity colored by rating")
plt.show()

# 2. Sử dụng IQR hoặc Z-score từng biến để nhận diện.
Q1_price=ecommerce_data["price"].quantile(0.25)
Q3_price=ecommerce_data["price"].quantile(0.75)
IQR_price=Q3_price-Q1_price
lower_bound_price=Q1_price-1.5*IQR_price
upper_bound_price=Q3_price+1.5*IQR_price
outliers_price=ecommerce_data[(ecommerce_data["price"] < lower_bound_price) | (ecommerce_data["price"] > upper_bound_price)]
print (f"Outliers for price:\n{outliers_price}")
z_scores_price=stats.zscore(ecommerce_data["price"].dropna())
outliers_z_price=ecommerce_data[abs(z_scores_price) > 3]    
print (f"Outliers for price based on Z-score:\n{outliers_z_price}")

# 3. Vẽ scatter plot 2D (với màu khác cho ngoại lệ) hoặc scatter matrix.
sns.scatterplot(x=ecommerce_data["price"], y=ecommerce_data["quantity"], hue=(ecommerce_data["price"] > upper_bound_price) | (ecommerce_data["price"] < lower_bound_price))
plt.title("Scatterplot price vs quantity with outliers highlighted")
plt.show()


# 4. So sánh điểm bất thường phát hiện bằng univariate vs multivariate và thảo luận lý do.
print (f"Number of outliers (univariate): {len(outliers_price)}")
print (f"Number of outliers (multivariate): {len(outliers_z_price)}")
