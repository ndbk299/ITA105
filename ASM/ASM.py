import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


print("\n===== ASM =====")

data1 = pd.DataFrame({
    'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'price': [3200000000, 1500000000, 2800000000, 5000000000, 950000000, 2100000000, 3500000000, 1200000000, 4500000000, 1800000000],
    'area': [85, 45, 70, 120, 30, 60, 95, 40, 110, 55],
    'rooms': [3, 2, 3, 4, 1, 2, 3, 2, 4, 2],
    'bathrooms': [2, 1, 2, 3, 1, 2, 2, 1, 3, 1],
    'status': ['new', 'used', 'renovated', 'new', 'used', 'new', 'renovated', 'used', 'new', 'used'],
    'location': ['Cầu Giấy', 'Thanh Xuân', 'Đống Đa', 'Tây Hồ', 'Hà Đông', 'Cầu Giấy', 'Ba Đình', 'Hoàng Mai', 'Tây Hồ', 'Thanh Xuân'],
    'description': ["Căn hộ luxury gần công viên, nội thất cao cấp",
                    "Nhà cozy, gần trường học, tiện ích đầy đủ",
                    "Nhà đã cải tạo, thiết kế hiện đại, gần trung tâm",
                    "Biệt thự ven hồ, không gian thoáng, sang trọng",
                    "Nhà nhỏ gọn, phù hợp gia đình trẻ, giá rẻ",
                    "Căn hộ mới, gần Vincom, tiện nghi hiện đại",       
                    "Nhà phố cải tạo, phong cách châu Âu, sang trọng",
                    "Nhà gần bến xe, giá hợp lý, cozy",
                    "Biệt thự cao cấp, luxury, gần trường quốc tế",
                    "Nhà tiện nghi, gần siêu thị, cozy"],
    'transaction_date': ['2024-05-12', '2024-06-20', '2024-07-15', '2024-08-01', '2024-09-10', '2024-10-05', '2024-11-12', '2024-12-01', '2025-01-18', '2025-02-25'],
    'image_path': ['house_001.jpg', 'house_002.jpg', 'house_003.jpg', 'house_004.jpg', 'house_005.jpg', 'house_006.jpg', 'house_007.jpg', 'house_008.jpg', 'house_009.jpg', 'house_010.jpg']
})

print(data1)

print("\n===== GĐ 1 =====")

# 1. Khám phá dữ liệu đa dạng

# - Phân tích thống kê: mean, median, std, min, max, missing values, duplicate.
print(data1.describe())
print(data1.isnull().sum())
print(data1.duplicated().sum())

# - Vẽ biểu đồ histogram, boxplot, violin plot cho numerical.
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
sns.histplot(data1['price'], kde=True)
plt.title('Histogram of Price')
plt.subplot(1, 3, 2)
sns.boxplot(x=data1['price'])
plt.title('Boxplot of Price')
plt.subplot(1, 3, 3)
sns.violinplot(x=data1['price'])
plt.title('Violin Plot of Price')
plt.show()

# - Phân tích phân phối categorical và text.
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.countplot(x=data1['status'])
plt.title('Count of Status')
plt.subplot(1, 2, 2)
sns.countplot(x=data1['location'])
plt.title('Count of Location')
plt.show()

# 2. Xử lý dữ liệu bẩn

# - Điền missing values tùy loại cột (mean/median/mode/forward/backward).
data1['price'] = data1['price'].fillna(data1['price'].mean())
data1['area'] = data1['area'].fillna(data1['area'].median())
data1['status'] = data1['status'].fillna(data1['status'].mode()[0])
data1['location'] = data1['location'].ffill()
data1['description'] = data1['description'].bfill()
data1['transaction_date'] = data1['transaction_date'].fillna(data1['transaction_date'].mode()[0])
data1['image_path'].fillna('no_image.jpg', inplace=True)

# - Xử lý dữ liệu không hợp lệ: giá âm, số phòng = 0, typo trong categorical.
data1 = data1[data1['price'] > 0]
data1 = data1[data1['rooms'] > 0]
data1['status'] = data1['status'].replace({'neww': 'new', 'usedd': 'used', 'renovatedd': 'renovated'})
print(data1)

# - Loại bỏ hoặc merge duplicate records.
data1.drop_duplicates(inplace=True)
print(data1)

# 3. Outliers & skew cơ bản

# - Phát hiện outlier bằng IQR, Z-score.
Q1 = data1['price'].quantile(0.25)
Q3 = data1['price'].quantile(0.75)
IQR = Q3 - Q1
Lower_bound = Q1 - 1.5 * IQR
Upper_bound = Q3 + 1.5 * IQR
outliers = data1[(data1['price']< Lower_bound) | (data1['price'] > Upper_bound)]
print("Outliers based on IQR:")
print(outliers)
data1['price_zscore'] = data1['price'].apply(lambda x: ( x- data1['price'].mean())/data1['price'].std())
outliers_zscore = data1[(data1['price_zscore'] < -3) | (data1['price_zscore'] > 3)]
print("Outliers based on Z-score:")
print(outliers_zscore)

# - Chọn chiến lược xử lý: loại bỏ, capping, hoặc biến đổi.
data1 =  data1[(data1['price'] >= Lower_bound) & (data1['price'] <= Upper_bound)]
data1['price'] = data1['price'].clip(lower=Lower_bound, upper=Upper_bound)
print(data1)

# 4. Chuẩn hóa số & biến đổi categorical

# - Scaling numerical bằng Min-Max, Z-score.
data1['price_scaled'] = (data1['price'] - data1['price'].min()) / (data1['price'].max() - data1['price'].min())
data1['area_scaled'] = (data1['area'] - data1['area'].min()) / (data1['area'].max() - data1['area'].min())
print(data1[['price_scaled', 'area_scaled']])
data1['price_zscore'] = (data1['price'] - data1['price'].mean()) / data1['price'].std()
data1['area_zscore'] = (data1['area'] - data1['area'].mean()) / data1['area'].std()
print(data1[['price_zscore', 'area_zscore']])

# - One-hot encoding / Label encoding categorical.
data1_encoded = pd.get_dummies(data1, columns=['status', 'location'], drop_first=True)
print(data1_encoded.head()) 

# - TF-IDF hoặc embedding text.
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(data1['description'])
print(tfidf_matrix.toarray())

# 5. Phát hiện duplicate thông tin dựa trên text similarity trong mô tả nhà. Gợi ý merge các bản ghi trùng lặp.
from sklearn.metrics.pairwise import cosine_similarity
similarity_matrix = cosine_similarity(tfidf_matrix)
print(similarity_matrix)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import datetime

# =========================
# 1. Load dữ liệu
# =========================
data = data1.copy()

# =========================
# 2. Feature Engineering
# =========================

# Biến đổi thời gian
data['transaction_date'] = pd.to_datetime(data['transaction_date'])
data['year'] = data['transaction_date'].dt.year
data['month'] = data['transaction_date'].dt.month
data['quarter'] = data['transaction_date'].dt.quarter
data['season'] = data['month'] % 12 // 3 + 1  # mùa

# Feature từ text
data['desc_word_count'] = data['description'].apply(lambda x: len(x.split()))
data['luxury_flag'] = data['description'].str.contains("luxury", case=False).astype(int)
data['cozy_flag'] = data['description'].str.contains("cozy", case=False).astype(int)

# Target log-transform
data['price_log'] = np.log1p(data['price'])

# =========================
# 3. Chuẩn bị dữ liệu
# =========================
X = data[['area','rooms','bathrooms','status','location','year','month','quarter','season',
          'desc_word_count','luxury_flag','cozy_flag','description']]
y = data['price_log']

# Cột numerical, categorical, text
num_features = ['area','rooms','bathrooms','year','month','quarter','season','desc_word_count','luxury_flag','cozy_flag']
cat_features = ['status','location']
text_features = 'description'

# Transformers
num_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

cat_transformer = OneHotEncoder(handle_unknown='ignore')

text_transformer = TfidfVectorizer(max_features=50)

# ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features),
        ('text', text_transformer, text_features)
    ]
)

# =========================
# 4. Pipeline mô hình
# =========================
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results[name] = {"RMSE": rmse, "R2": r2}

# =========================
# 5. Kết quả
# =========================
print("Kết quả mô hình:")
for model, metrics in results.items():
    print(f"{model}: RMSE={metrics['RMSE']:.2f}, R2={metrics['R2']:.2f}")
