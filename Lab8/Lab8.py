import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, PowerTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv(r'D:\FPT\ITA105\Lab8\ITA105_Lab_8.csv')

# Bài 1: Xây dựng Pipeline tổng quát cho dữ liệu đa dạng
# - Xây dựng ColumnTransformer cho từng nhóm cột:
# - Số: Impute → Outlier removal → Scaling → Log/Power transform
# - Categorical: Impute → One-hot
# # - Text: Tokenization → Stopword → TF-IDF
# - Time series: Extract (month, quarter…) → Interpolate missing
# - Tạo pipeline tổng hợp bằng Pipeline() + ColumnTransformer().
# - Xuất ra schema cuối cùng (feature names sau encoding TF-IDF và one-hot).
# - Kiểm tra pipeline chạy được trên 10 dòng dữ liệu demo (smoke test).
# Bài 2: Kiểm thử Pipeline & Kiểm định chất lượng dữ liệu
# - Chạy pipeline với 5 bộ dữ liệu kiểm thử:
# - Dữ liệu đầy đủ
# - Dữ liệu có missing nhiều
# - Dữ liệu bị lệch phân phối
# - Dữ liệu có categorical chưa từng thấy (unseen categories)
# - Dữ liệu sai định dạng (string thay vì số)
# - Kiểm thử theo checklist:
# - Có lỗi nào xuất hiện?
# - Outlier removal hoạt động đúng không?
# - Encoding có xử lý unseen category không?
# - Output cuối cùng có dạng numeric matrix chưa?
# - Hình dạng (shape) dữ liệu có nhất quán hay biến động?
# - So sánh phân phối dữ liệu trước/sau pipeline (histogram + mô tả thống kê).
# - Viết báo cáo lỗi và cách sửa.
# Bài 3: Tích hợp pipeline vào mô hình dự báo
# - Tích hợp model vào pipeline từ Bài 1:
# - Linear Regression
# - RandomForest
# - XGBoost (tuỳ chọn)
# - Chạy Cross-validation với pipeline:
# - 5-fold hoặc 10-fold
# - So sánh giữa mô hình dùng pipeline và mô hình dùng dữ
# liệu xử lý thủ công
# - So sánh kết quả: Mô hình, RMSE, MAE, R², Mức ổn định (var giữa folds)
# - Đánh giá:
# - Pipeline giúp giảm lỗi thủ công như thế nào?
# - Vì sao CV trong pipeline chuẩn hơn CV ngoài pipeline?
# - Tại sao pipeline giúp “không leak dữ liệu”?
# - Xuất reports: normalized feature importances, ảnh hưởng của text features (nếu có TF-IDF)
# Bài 4: Triển khai pipeline thành sản phẩm cho dự báo thực tế.
# - Tạo pipeline inference:
# - Nhận input từ file CSV mới
# - Chạy toàn bộ pipeline transform
# - Xuất ra dự đoán giá nhà
# - Kiểm thử với dữ liệu mới chưa từng có trong tập train.
# - Xây dựng hàm predict_price(new_data) dùng pipeline đã lưu bằng joblib.
# - Tạo tài liệu mô tả:
# - Pipeline gồm những bước gì?
# - Đầu vào/đầu ra dạng gì?
# - Những rủi ro khi dùng dữ liệu mới? (unseen category, drift, format)
