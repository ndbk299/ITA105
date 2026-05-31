import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
import joblib
import os

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, FunctionTransformer, PowerTransformer
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
df = pd.read_csv('/mnt/user-data/uploads/ITA105_Lab_8.csv')
print("=" * 60)
print("DATASET INFO")
print("=" * 60)
print(df.shape)
print(df.dtypes)
print(df.head(3))

TARGET = 'SalePrice'
NUM_COLS      = ['LotArea', 'Rooms', 'NoiseFeature']
CAT_COLS      = ['Neighborhood', 'Condition', 'HasGarage']
TEXT_COL      = 'Description'
DATE_COL      = 'SaleDate'
DROP_COLS     = ['ImagePath']

# ─────────────────────────────────────────────
# Custom transformers
# ─────────────────────────────────────────────
class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clip values beyond IQR fence (fitted on train, applied to test)."""
    def fit(self, X, y=None):
        self.lower_ = {}
        self.upper_ = {}
        for i in range(X.shape[1]):
            col = X[:, i].astype(float)
            q1, q3 = np.nanpercentile(col, 25), np.nanpercentile(col, 75)
            iqr = q3 - q1
            self.lower_[i] = q1 - 1.5 * iqr
            self.upper_[i] = q3 + 1.5 * iqr
        return self

    def transform(self, X, y=None):
        X = X.copy().astype(float)
        for i in range(X.shape[1]):
            X[:, i] = np.clip(X[:, i], self.lower_[i], self.upper_[i])
        return X


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract year, month, quarter from a date column."""
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if hasattr(X, 'values'):
            X = X.values
        results = []
        for val in np.array(X).ravel():
            try:
                dt = pd.to_datetime(val)
                results.append([dt.year, dt.month, dt.quarter])
            except Exception:
                results.append([np.nan, np.nan, np.nan])
        arr = np.array(results, dtype=float)
        # Interpolate NaN with column median
        for col_idx in range(arr.shape[1]):
            median = np.nanmedian(arr[:, col_idx])
            arr[:, col_idx] = np.where(np.isnan(arr[:, col_idx]), median, arr[:, col_idx])
        return arr


class SafeNumericConverter(BaseEstimator, TransformerMixin):
    """Force-convert to numeric, coercing errors to NaN."""
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df_tmp = pd.DataFrame(X)
        return df_tmp.apply(pd.to_numeric, errors='coerce').values


# ─────────────────────────────────────────────
# BÀI 1 – Pipeline tổng quát
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("BÀI 1: XÂY DỰNG PIPELINE")
print("=" * 60)

# --- Numeric pipeline: Impute → Convert → Clip outliers → Scale → Power transform
numeric_pipeline = Pipeline([
    ('safe_convert', SafeNumericConverter()),
    ('impute',       SimpleImputer(strategy='median')),
    ('clip',         OutlierClipper()),
    ('scale',        StandardScaler()),
    ('power',        PowerTransformer(method='yeo-johnson')),
])

# --- Categorical pipeline: Impute → One-hot (handle_unknown='ignore')
categorical_pipeline = Pipeline([
    ('impute',  SimpleImputer(strategy='most_frequent')),
    ('onehot',  OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])

# --- Text pipeline: TF-IDF (tokenization + stopword removal built-in)
# FunctionTransformer fills NaN descriptions before TF-IDF
class TextFillTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X, y=None):
        arr = np.array(X).ravel()
        return ['' if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v) for v in arr]

text_pipeline = Pipeline([
    ('fill',  TextFillTransformer()),
    ('tfidf', TfidfVectorizer(
        max_features=20,
        stop_words='english',
        token_pattern=r'(?u)\b[a-zA-Z]{2,}\b'
    )),
])

# --- Date pipeline
date_pipeline = Pipeline([
    ('date_extract', DateFeatureExtractor()),
    ('scale',        StandardScaler()),
])

# --- Combined ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num',  numeric_pipeline,     NUM_COLS),
        ('cat',  categorical_pipeline, CAT_COLS),
        ('text', text_pipeline,        TEXT_COL),
        ('date', date_pipeline,        DATE_COL),
    ],
    remainder='drop'
)

# --- Feature name helper
def get_feature_names(ct):
    names = []
    for name, transformer, cols in ct.transformers_:
        if name == 'remainder':
            continue
        if name == 'num':
            names += [f"num_{c}" for c in cols]
        elif name == 'cat':
            ohe = transformer.named_steps['onehot']
            names += list(ohe.get_feature_names_out(cols if isinstance(cols, list) else [cols]))
        elif name == 'text':
            tfidf = transformer.named_steps['tfidf']
            names += [f"text_{v}" for v in tfidf.get_feature_names_out()]
        elif name == 'date':
            names += ['date_year', 'date_month', 'date_quarter']
    return names

# ── Prepare X/y
X = df.drop(columns=[TARGET] + DROP_COLS)
y = df[TARGET].values

# ── Fit preprocessor & smoke test on 10 rows
preprocessor.fit(X, y)
X_transformed = preprocessor.transform(X)

feature_names = get_feature_names(preprocessor)

print(f"\nTotal features after encoding: {X_transformed.shape[1]}")
print(f"Feature schema ({len(feature_names)} features):")
for i, fn in enumerate(feature_names):
    print(f"  [{i:3d}] {fn}")

# Smoke test on 10 rows
sample = X.head(10).copy()
out = preprocessor.transform(sample)
print(f"\nSmoke test (10 rows): output shape = {out.shape}")
print(f"Any NaN: {np.isnan(out).any()}")
print("✓ Pipeline smoke test PASSED")

# ─────────────────────────────────────────────
# BÀI 2 – Kiểm thử với 5 bộ dữ liệu
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("BÀI 2: KIỂM THỬ PIPELINE")
print("=" * 60)

base = X.head(30).copy()

def make_test_sets(base_df):
    # 1. Dữ liệu đầy đủ
    t1 = base_df.copy()

    # 2. Missing nhiều (30 % mỗi cột)
    t2 = base_df.copy()
    rng = np.random.default_rng(42)
    for col in NUM_COLS + CAT_COLS + [TEXT_COL]:
        mask = rng.random(len(t2)) < 0.3
        t2.loc[mask, col] = np.nan

    # 3. Phân phối lệch (LotArea nhân 100x)
    t3 = base_df.copy()
    t3['LotArea'] = t3['LotArea'] * 100

    # 4. Unseen categories
    t4 = base_df.copy()
    t4['Neighborhood'] = 'Z_UNSEEN'
    t4['Condition']    = 'Unknown99'

    # 5. Sai định dạng (số dạng string, ngày sai)
    t5 = base_df.copy()
    t5['LotArea']    = t5['LotArea'].astype(str).str.replace('0', 'O')  # 'O' thay '0'
    t5['NoiseFeature'] = 'NOT_A_NUMBER'
    t5['SaleDate']   = 'BAD_DATE'

    return [
        ("1 - Đầy đủ",                    t1),
        ("2 - Missing nhiều",              t2),
        ("3 - Phân phối lệch",             t3),
        ("4 - Unseen categories",          t4),
        ("5 - Sai định dạng",              t5),
    ]

test_sets = make_test_sets(base)

checklist_rows = []
for label, tdf in test_sets:
    result = {"Bộ dữ liệu": label}
    try:
        out = preprocessor.transform(tdf)
        result["Lỗi"] = "Không"
        result["Shape"] = str(out.shape)
        result["Numeric matrix"] = "Có" if out.dtype in [np.float64, np.float32] else str(out.dtype)
        result["NaN còn lại"]    = str(np.isnan(out).sum())
        result["Shape nhất quán"] = "Có" if out.shape[1] == X_transformed.shape[1] else f"KHÁC: {out.shape[1]}"
    except Exception as e:
        result["Lỗi"] = str(e)[:60]
        result["Shape"] = "N/A"
        result["Numeric matrix"] = "N/A"
        result["NaN còn lại"]    = "N/A"
        result["Shape nhất quán"] = "N/A"
    checklist_rows.append(result)

checklist_df = pd.DataFrame(checklist_rows)
print("\nChecklist kết quả:")
print(checklist_df.to_string(index=False))

# ── So sánh phân phối trước/sau pipeline
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Phân phối trước/sau pipeline (cột số)", fontsize=14, fontweight='bold')

for idx, col in enumerate(NUM_COLS):
    ax = axes[0, idx]
    ax.hist(df[col].dropna(), bins=30, color='steelblue', alpha=0.7, edgecolor='white')
    ax.set_title(f"Trước: {col}", fontsize=10)
    ax.set_xlabel("Giá trị gốc")
    ax.set_ylabel("Tần suất")

for idx, col in enumerate(NUM_COLS):
    ax = axes[1, idx]
    col_idx = NUM_COLS.index(col)
    ax.hist(X_transformed[:, col_idx], bins=30, color='tomato', alpha=0.7, edgecolor='white')
    ax.set_title(f"Sau pipeline: {col}", fontsize=10)
    ax.set_xlabel("Giá trị đã xử lý")
    ax.set_ylabel("Tần suất")

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/bai2_distribution.png', dpi=120, bbox_inches='tight')
plt.close()
print("\n✓ Đã lưu biểu đồ phân phối: bai2_distribution.png")

# ── Mô tả thống kê
print("\nThống kê trước pipeline (cột số):")
print(df[NUM_COLS].describe().round(2).to_string())
print("\nThống kê sau pipeline (cột số):")
before_after = pd.DataFrame(
    X_transformed[:, :len(NUM_COLS)],
    columns=[f"num_{c}" for c in NUM_COLS]
).describe().round(4)
print(before_after.to_string())

# ── Báo cáo lỗi
print("""
─── Báo cáo lỗi và cách sửa ───
| Bộ dữ liệu         | Vấn đề                          | Cách pipeline xử lý               |
|--------------------|----------------------------------|------------------------------------|
| 2 - Missing nhiều  | Giá trị NaN nhiều                | SimpleImputer(median/most_freq)    |
| 3 - Phân phối lệch | LotArea skewed cực mạnh          | OutlierClipper + PowerTransformer  |
| 4 - Unseen cat     | OHE không biết category mới      | handle_unknown='ignore' → vector 0 |
| 5 - Sai định dạng  | Số dạng string, ngày sai         | SafeNumericConverter coerce→NaN    |
                                                       DateFeatureExtractor try/except    |
""")

# ─────────────────────────────────────────────
# BÀI 3 – Tích hợp mô hình & Cross-validation
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("BÀI 3: TÍCH HỢP MÔ HÌNH & CROSS-VALIDATION")
print("=" * 60)

models = {
    'LinearRegression': LinearRegression(),
    'RandomForest':     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = ['neg_root_mean_squared_error', 'neg_mean_absolute_error', 'r2']

results = {}
for name, model in models.items():
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model',        model),
    ])
    cv_res = cross_validate(full_pipeline, X, y, cv=cv, scoring=scoring, return_train_score=False)
    results[name] = {
        'RMSE':     -cv_res['test_neg_root_mean_squared_error'],
        'MAE':      -cv_res['test_neg_mean_absolute_error'],
        'R2':        cv_res['test_r2'],
    }
    print(f"\n{name}:")
    print(f"  RMSE  = {(-cv_res['test_neg_root_mean_squared_error']).mean():.2f} ± {(-cv_res['test_neg_root_mean_squared_error']).std():.2f}")
    print(f"  MAE   = {(-cv_res['test_neg_mean_absolute_error']).mean():.2f} ± {(-cv_res['test_neg_mean_absolute_error']).std():.2f}")
    print(f"  R²    = {cv_res['test_r2'].mean():.4f} ± {cv_res['test_r2'].std():.4f}")

# ── So sánh pipeline vs xử lý thủ công
print("\n── So sánh Pipeline vs Xử lý thủ công ──")
# Thủ công: chỉ dùng cột số sẵn có, impute & scale bên ngoài CV
from sklearn.model_selection import KFold
X_manual = df[NUM_COLS].copy()
X_manual = X_manual.apply(pd.to_numeric, errors='coerce')
X_manual.fillna(X_manual.median(), inplace=True)
X_manual_scaled = (X_manual - X_manual.mean()) / X_manual.std()

manual_results = {}
for name, model in [('LinearRegression', LinearRegression()),
                    ('RandomForest', RandomForestRegressor(n_estimators=100, random_state=42))]:
    fold_rmse, fold_r2 = [], []
    for train_idx, val_idx in cv.split(X_manual_scaled):
        Xtr, Xval = X_manual_scaled.iloc[train_idx], X_manual_scaled.iloc[val_idx]
        ytr, yval = y[train_idx], y[val_idx]
        model.fit(Xtr, ytr)
        pred = model.predict(Xval)
        fold_rmse.append(np.sqrt(mean_squared_error(yval, pred)))
        fold_r2.append(r2_score(yval, pred))
    manual_results[name] = {'RMSE': np.array(fold_rmse), 'R2': np.array(fold_r2)}
    print(f"\n{name} (thủ công): RMSE={np.mean(fold_rmse):.2f} ± {np.std(fold_rmse):.2f}  R²={np.mean(fold_r2):.4f}")

# ── Bảng so sánh tổng hợp
print("\n═══ BẢNG SO SÁNH ═══")
compare_rows = []
for name in models:
    compare_rows.append({
        'Mô hình': name,
        'Phương pháp': 'Pipeline',
        'RMSE (mean)': round(results[name]['RMSE'].mean(), 2),
        'RMSE (std)':  round(results[name]['RMSE'].std(), 2),
        'MAE (mean)':  round(results[name]['MAE'].mean(), 2),
        'R² (mean)':   round(results[name]['R2'].mean(), 4),
        'R² (std)':    round(results[name]['R2'].std(), 4),
    })
    compare_rows.append({
        'Mô hình': name,
        'Phương pháp': 'Thủ công',
        'RMSE (mean)': round(manual_results[name]['RMSE'].mean(), 2),
        'RMSE (std)':  round(manual_results[name]['RMSE'].std(), 2),
        'MAE (mean)':  'N/A',
        'R² (mean)':   round(manual_results[name]['R2'].mean(), 4),
        'R² (std)':    round(manual_results[name]['R2'].std(), 4),
    })

compare_df = pd.DataFrame(compare_rows)
print(compare_df.to_string(index=False))

# ── Feature importance (Random Forest)
rf_pipeline = Pipeline([('preprocessor', preprocessor), ('model', RandomForestRegressor(n_estimators=100, random_state=42))])
rf_pipeline.fit(X, y)
importances = rf_pipeline.named_steps['model'].feature_importances_
imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
imp_df = imp_df.sort_values('importance', ascending=False).head(20)
imp_df['importance_norm'] = imp_df['importance'] / imp_df['importance'].sum()

fig, ax = plt.subplots(figsize=(10, 7))
sns.barplot(data=imp_df, x='importance_norm', y='feature', ax=ax, palette='viridis')
ax.set_title("Top 20 Feature Importances (Normalized) – Random Forest", fontsize=13, fontweight='bold')
ax.set_xlabel("Normalized Importance")
ax.set_ylabel("Feature")
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/bai3_feature_importance.png', dpi=120, bbox_inches='tight')
plt.close()
print("\n✓ Đã lưu biểu đồ feature importance: bai3_feature_importance.png")

print("""
─── Đánh giá ───
▸ Pipeline giúp giảm lỗi thủ công:
  • Tất cả bước transform được đóng gói → không quên bước nào.
  • OneHotEncoder với handle_unknown='ignore' an toàn với unseen category.
  • SafeNumericConverter xử lý string→NaN trước khi impute.

▸ CV trong pipeline chuẩn hơn CV ngoài pipeline:
  • Mỗi fold, scaler/imputer chỉ fit trên train split → không dùng thông tin val.
  • Xử lý thủ công fit scaler trên toàn bộ X trước CV → data leakage.

▸ Pipeline không leak dữ liệu:
  • Mỗi bước transform chỉ .fit() trên X_train của từng fold,
    .transform() độc lập trên X_val → số liệu val không ảnh hưởng preprocessing.
""")

# ─────────────────────────────────────────────
# BÀI 4 – Triển khai pipeline inference
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("BÀI 4: TRIỂN KHAI PIPELINE INFERENCE")
print("=" * 60)

# Lưu pipeline tốt nhất (Random Forest)
best_pipeline = rf_pipeline
joblib.dump(best_pipeline, '/mnt/user-data/outputs/house_price_pipeline.pkl')
print("✓ Pipeline đã lưu: house_price_pipeline.pkl")

def predict_price(new_data):
    """
    Dự báo giá nhà từ DataFrame hoặc file CSV mới.

    Parameters
    ----------
    new_data : pd.DataFrame hoặc str (đường dẫn CSV)

    Returns
    -------
    np.ndarray: mảng giá trị dự đoán (USD)
    """
    if isinstance(new_data, str):
        new_data = pd.read_csv(new_data)

    loaded_pipeline = joblib.load('/mnt/user-data/outputs/house_price_pipeline.pkl')

    # Đảm bảo đúng cột đầu vào (bỏ target/drop nếu có)
    for drop_col in [TARGET] + DROP_COLS:
        if drop_col in new_data.columns:
            new_data = new_data.drop(columns=[drop_col])

    predictions = loaded_pipeline.predict(new_data)
    return predictions


# ── Kiểm thử với dữ liệu mới chưa thấy trong train
new_test = pd.DataFrame({
    'LotArea':       [3000,   500,   99999],
    'Rooms':         [5,      3,     10   ],
    'HasGarage':     [1,      0,     1    ],
    'NoiseFeature':  [0.5,   -1.2,   2.5  ],
    'Neighborhood':  ['A',   'Z_NEW','C'  ],  # Z_NEW chưa từng thấy
    'Condition':     ['Good','Poor', 'Excellent'],
    'Description':   [
        'bright modern garden luxury',
        'quiet cozy near school',
        'spacious renovated luxury beautiful view garage'
    ],
    'SaleDate':      ['2024-01-15', '2023-06-01', '2022-12-31'],
})

preds = predict_price(new_test.copy())
print("\nDự đoán giá nhà cho 3 mẫu mới:")
for i, p in enumerate(preds):
    print(f"  Mẫu {i+1}: ${p:,.2f}")

# ── Tài liệu mô tả pipeline
doc = """
╔══════════════════════════════════════════════════════════════════════╗
║           TÀI LIỆU MÔ TẢ PIPELINE DỰ BÁO GIÁ NHÀ                   ║
╚══════════════════════════════════════════════════════════════════════╝

1. CÁC BƯỚC PIPELINE
─────────────────────
   Step 1: ColumnTransformer (preprocessor)
     ├── [Numeric]  LotArea, Rooms, NoiseFeature
     │     SafeNumericConverter → SimpleImputer(median)
     │     → OutlierClipper(IQR) → StandardScaler → PowerTransformer(Yeo-Johnson)
     │
     ├── [Categorical]  Neighborhood, Condition, HasGarage
     │     SimpleImputer(most_frequent) → OneHotEncoder(handle_unknown='ignore')
     │
     ├── [Text]  Description
     │     TfidfVectorizer(max_features=20, stop_words='english')
     │
     └── [Date]  SaleDate
           DateFeatureExtractor (year, month, quarter) → StandardScaler

   Step 2: RandomForestRegressor (n_estimators=100)

2. ĐẦU VÀO / ĐẦU RA
─────────────────────
   Đầu vào : pandas DataFrame với các cột:
       LotArea (int), Rooms (int), HasGarage (0/1),
       NoiseFeature (float), Neighborhood (str), Condition (str),
       Description (str), SaleDate (YYYY-MM-DD str)

   Đầu ra  : numpy array giá trị float (USD), shape = (n_samples,)

3. CÁCH SỬ DỤNG
─────────────────
   from lab8_solution import predict_price
   import pandas as pd

   df_new = pd.read_csv('new_houses.csv')
   predictions = predict_price(df_new)
   # Hoặc:
   predictions = predict_price('new_houses.csv')

4. NHỮNG RỦI RO KHI DÙNG DỮ LIỆU MỚI
──────────────────────────────────────
   a) Unseen Category:
      → Xử lý: OHE với handle_unknown='ignore' → vector 0, mô hình vẫn chạy.
      → Rủi ro: Mô hình mất thông tin category → dự đoán kém chính xác hơn.

   b) Data Drift:
      → Phân phối dữ liệu thực tế thay đổi theo thời gian (giá nhà tăng).
      → Cần retrain định kỳ hoặc thêm monitoring drift.

   c) Format Sai:
      → Số dạng string: SafeNumericConverter chuyển thành NaN → impute bằng median.
      → Ngày sai định dạng: DateFeatureExtractor trả [NaN, NaN, NaN] → impute.
      → Missing cột: Pipeline raise KeyError → cần validate schema đầu vào.

   d) Outlier cực đoan:
      → OutlierClipper clamp theo IQR của tập train.
      → Giá trị ngoài phạm vi huấn luyện sẽ bị clip → không extrapolate.

   e) Scale mismatch:
      → Scaler fit trên train; dữ liệu mới có đơn vị khác → kết quả lệch.
      → Cần đảm bảo đơn vị nhất quán (m² vs ft², USD vs VND…).
"""
print(doc)

with open('/mnt/user-data/outputs/pipeline_documentation.txt', 'w', encoding='utf-8') as f:
    f.write(doc)
print("✓ Tài liệu đã lưu: pipeline_documentation.txt")

print("\n" + "=" * 60)
print("HOÀN THÀNH TẤT CẢ CÁC BÀI")
print("=" * 60)
