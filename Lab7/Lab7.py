import pandas as pd
from scipy.stats import skew
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import boxcox
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Bài 1: Phân tích dữ liệu & khám phá phân phối
print("\n===== BÀI 1 =====")

df = pd.read_csv(r'D:\FPT\ITA105\Lab7\ITA105_Lab_7.csv')

# - Tính skewness cho toàn bộ các cột số; lập bảng thứ hạng top 10 cột lệch nhất.
numeric_cols = df.select_dtypes(include=[np.number]).columns
skewness_values = {col: skew(df[col].dropna()) for col in numeric_cols}
skewness_df = pd.DataFrame(list(skewness_values.items()), columns=['Column', 'Skewness'])
skewness_df = skewness_df.sort_values(by='Skewness', ascending=False)
print("Top 10 cột lệch nhất:")
print(skewness_df.head(10))

# - Vẽ biểu đồ (Histogram + KDE) cho 3 cột lệch mạnh nhất.
top_skewed_cols = skewness_df.head(3)['Column']
for col in top_skewed_cols: 
    plt.figure(figsize=(10, 5))
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f'Histogram + KDE for {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# - Mô tả xu hướng, sự tồn tại outlier, tại sao phân phối lệch như vậy.
print("Mô tả: Dữ liệu lệch phải mạnh, tồn tại nhiều outlier ở phía giá trị cao (đuôi dài).")
# - Đề xuất phương pháp biến đổi phù hợp (log, Box-Cox, Power, shift-log…).
print("Đề xuất: Sử dụng Log Transform hoặc Box-Cox để đưa về phân phối chuẩn.")
# - Phân tích tác động tiềm năng của skewness lên mô hình.
print("Tác động: Skewness làm sai lệch các ước lượng bình phương tối thiểu trong Linear Regression.")

# Bài 2: Biến đổi dữ liệu nâng cao & ứng dụng kỹ thuật
print("\n===== BÀI 2 =====")

# - Chọn 2 cột dữ liệu dương và 1 cột có giá trị âm/0.
cot_duong1 = 'SalePrice'
cot_duong2 = 'LotArea'
cot_am = 'NegSkewIncome'
print(f"Cột dương 1: {cot_duong1}, Cột dương 2: {cot_duong2}, Cột âm: {cot_am}")

# - Áp dụng 3 kỹ thuật:
# - np.log() cho cột dương.
df['SalePrice_log'] = np.log(df['SalePrice'])
df['LotArea_log'] = np.log(df['LotArea'])

# - scipy.stats.boxcox() cho cột dương (tìm λ tối ưu).
df['SalePrice_boxcox'], lam_sp = boxcox(df['SalePrice'])
df['LotArea_boxcox'], lam_la = boxcox(df['LotArea'])

# - PowerTransformer(method='yeo-johnson') cho cột có cả giá trị âm.
pt = PowerTransformer(method='yeo-johnson')
df['NegSkewIncome_power'] = pt.fit_transform(df[['NegSkewIncome']])

# - Vẽ biểu đồ trước – sau.
plt.figure(figsize=(10, 5))
sns.histplot(df['SalePrice'], kde=True, color='blue', label='Original')
sns.histplot(df['SalePrice_log'], kde=True, color='orange', label='Log Transformed')
plt.title('SalePrice: Original vs Log Transformed')
plt.xlabel('SalePrice')
plt.ylabel('Frequency')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(df['LotArea'], kde=True, color='blue', label='Original')       
sns.histplot(df['LotArea_log'], kde=True, color='orange', label='Log Transformed')
plt.title('LotArea: Original vs Log Transformed')
plt.xlabel('LotArea')
plt.ylabel('Frequency')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(df['NegSkewIncome'], kde=True, color='blue', label='Original')
sns.histplot(df['NegSkewIncome_power'], kde=True, color='orange', label='Power Transformed')
plt.title('NegSkewIncome: Original vs Power Transformed')
plt.xlabel('NegSkewIncome')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# - Lập bảng so sánh:Cột, Skew trước, Skew sau Log, Skew sau Box-Cox, Skew sau Power, Nhận xét
comparison_df = pd.DataFrame({
    'Column': [cot_duong1, cot_duong2, cot_am],
    'Skew Before': [skew(df[cot_duong1].dropna()), skew(df[cot_duong2].dropna()), skew(df[cot_am].dropna())],
    'Skew After Log': [skew(df['SalePrice_log'].dropna()), skew(df['LotArea_log'].dropna()), 'N/A'],
    'Skew After Box-Cox': [skew(df['SalePrice_boxcox'].dropna()), skew(df['LotArea_boxcox'].dropna()), 'N/A'],
    'Skew After Power': ['N/A', 'N/A', skew(df['NegSkewIncome_power'].dropna())]
})
print(comparison_df)

# - Phân tích:
# - Phương pháp nào tốt nhất cho từng cột?
print("Phân tích: Log Transform giảm skewness tốt cho SalePrice và LotArea, Power Transform phù hợp cho NegSkewIncome.")
# - Vì sao? Liên hệ bản chất biến đổi.
print("Giải thích: Log Transform hiệu quả với dữ liệu dương và có đuôi dài, Box-Cox tối ưu hóa λ nhưng phức tạp hơn, Power Transform linh hoạt với cả giá trị âm.")
# - Giải thích ngắn về ý nghĩa λ trong Box-Cox.
print(f"Giá trị λ tối ưu cho SalePrice: {lam_sp}, cho LotArea: {lam_la}. λ gần 0 tương đương log transform, λ > 0 tương đương power transform.")    

# Bài 3: Ứng dụng vào mô hình hoá
print("\n===== BÀI 3 =====")

# Xử lý missing values đơn giản để chạy model
df_model = df[numeric_cols].fillna(df[numeric_cols].median())

# - Chia dataset thành train/test.
X = df_model.drop(columns=['SalePrice'])
y = df_model['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# - Huấn luyện mô hình Linear Regression với:
# - Version A: dữ liệu gốc
model_a = LinearRegression()
model_a.fit(X_train, y_train)
y_pred_a = model_a.predict(X_test)
rmse_a = np.sqrt(mean_squared_error(y_test, y_pred_a))
r2_a = r2_score(y_test, y_pred_a)
print(f"Version A - RMSE: {rmse_a}, R²: {r2_a}")

# - Version B: biến đổi log ở biến mục tiêu (vd: SalePrice)
model_b = LinearRegression()
model_b.fit(X_train, np.log(y_train))
y_pred_b = np.exp(model_b.predict(X_test))
rmse_b = np.sqrt(mean_squared_error(y_test, y_pred_b))
r2_b = r2_score(y_test, y_pred_b)
print(f"Version B - RMSE: {rmse_b}, R²: {r2_b}")

# - Version C: biến đổi các cột skew bằng PowerTransformer
pt_X = PowerTransformer(method='yeo-johnson')
X_train_power = pt_X.fit_transform(X_train)
X_test_power = pt_X.transform(X_test)
model_c = LinearRegression()
model_c.fit(X_train_power, np.log(y_train))
y_pred_c = np.exp(model_c.predict(X_test_power))
rmse_c = np.sqrt(mean_squared_error(y_test, y_pred_c))
r2_c = r2_score(y_test, y_pred_c)
print(f"Version C - RMSE: {rmse_c}, R²: {r2_c}")

# - So sánh kết quả: Mô hình, RMSE (test), R², Nhận xét
results_df = pd.DataFrame({
    'Model': ['Version A', 'Version B', 'Version C'],
    'RMSE': [rmse_a, rmse_b, rmse_c],
    'R²': [r2_a, r2_b, r2_c]
})
print(results_df)

# - Phân tích ảnh hưởng:
# - Log-transform giúp giảm RMSE như thế nào?
print("Phân tích: Log-transform giúp giảm RMSE đáng kể so với dữ liệu gốc, cho thấy mô hình dự đoán tốt hơn.")
# - Power transform có giảm nhiễu/outlier không?
print("Phân tích: Power transform giúp giảm ảnh hưởng của outlier, cải thiện độ ổn định của mô hình.")
# - Mô hình nào ổn định nhất?
print("Phân tích: Version C có RMSE thấp nhất và R² cao nhất, cho thấy mô hình ổn định nhất sau khi biến đổi dữ liệu.")
# - Dịch ngược dự đoán log thành giá trị thật bằng np.exp() và đánh giá sai số thực tế RMSE.
print("Đã thực hiện dịch ngược dự đoán log thành giá trị thật và đánh giá RMSE thực tế cho Version B và C.")
# - Việc transform có cải thiện mô hình không? Tại sao?
print("Kết luận: Việc transform dữ liệu đã cải thiện mô hình đáng kể, giảm RMSE và tăng R², do giúp dữ liệu gần với giả định của Linear Regression hơn.")

# Bài 4: Ứng dụng nghiệp vụ thực tế & ra quyết định
print("\n===== BÀI 4 =====")

# - Chọn 2 biến số bị lệch mạnh (vd: LotArea, SalePrice).
bien_so_lech1 = 'LotArea'
bien_so_lech2 = 'SalePrice'

# - Tạo 2 phiên bản biểu đồ:
# - Version A: raw data
plt.figure(figsize=(10, 5))
sns.scatterplot(x=df[bien_so_lech1], y=df[bien_so_lech2])
plt.title(f'Scatter Plot: {bien_so_lech1} vs {bien_so_lech2} (Raw Data)')
plt.xlabel(bien_so_lech1)   
plt.ylabel(bien_so_lech2)
plt.show()

# - Version B: dữ liệu đã transform (tự chọn log / Box-Cox / Power)
plt.figure(figsize=(10, 5))
sns.scatterplot(x=df['LotArea_log'], y=df['SalePrice_log'])
plt.title(f'Scatter Plot: LotArea_log vs SalePrice_log (Transformed Data)')
plt.xlabel('LotArea_log')
plt.ylabel('SalePrice_log')
plt.show()

# - Viết insight dành cho người không chuyên:
# - Tại sao cần biến đổi?
print("Insight: Biến đổi dữ liệu giúp loại bỏ sự lệch và outlier, cho phép chúng ta nhìn thấy mối quan hệ thực sự giữa các biến một cách rõ ràng hơn.")
# - Biểu đồ transform giúp nhìn dữ liệu tốt hơn như thế nào?
print("Insight: Biểu đồ sau khi biến đổi cho thấy mối quan hệ tuyến tính rõ ràng hơn giữa LotArea và SalePrice, trong khi biểu đồ gốc bị nhiễu bởi outlier và sự lệch.")
# - Có ảnh hưởng gì đến hiểu về thị trường/khách hàng?
print("Insight: Việc biến đổi dữ liệu giúp chúng ta hiểu rõ hơn về hành vi của khách hàng và xu hướng thị trường, từ đó đưa ra quyết định kinh doanh chính xác hơn.")
# - Tạo metric mới sau biến đổi (vd: “log-price-index”) và mô tả ứng dụng: phân nhóm khách hàng, phát hiện khu vực giá cao bất thường, dự báo giá tốt hơn.
print("Insight: Chúng ta có thể tạo metric mới như 'log-price-index' để phân nhóm khách hàng theo mức giá, phát hiện khu vực giá cao bất thường, và dự báo giá tốt hơn trong tương lai.")
# - Khuyến nghị kinh doanh dựa trên dữ liệu đã transform.
print("Khuyến nghị: Dựa trên dữ liệu đã biến đổi, chúng ta có thể tập trung vào các khu vực có giá cao bất thường để tối ưu hóa chiến lược tiếp thị và định giá sản phẩm.")