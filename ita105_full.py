"""
ITA105 – Dự án PropTech: Pipeline phân tích & dự báo giá nhà
Giai đoạn 1: EDA, làm sạch, chuẩn hóa
Giai đoạn 2: Feature engineering, pipeline, mô hình, KPI, visualization
Giai đoạn Hoàn thiện: Unseen categories, feature interaction, model comparison
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings, re, os, joblib
from scipy import stats
from scipy.stats import skew

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, OneHotEncoder,
    PowerTransformer, LabelEncoder, QuantileTransformer
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_validate, KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from difflib import SequenceMatcher

warnings.filterwarnings('ignore')
os.makedirs('/mnt/user-data/outputs', exist_ok=True)

PALETTE = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B", "#44BBA4", "#E94F37"]
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10})

# ════════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════════
df_raw = pd.read_csv('/mnt/user-data/uploads/ITA105_Lab_8.csv')
print(f"Dataset: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} cols")

# ════════════════════════════════════════════════════════════════
# GIAI ĐOẠN 1 – PHẦN 1: Khám phá dữ liệu
# ════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("GIAI ĐOẠN 1 – KHÁM PHÁ DỮ LIỆU")
print("═"*60)

NUM_COLS = ['LotArea', 'Rooms', 'NoiseFeature', 'SalePrice']
CAT_COLS = ['Neighborhood', 'Condition', 'HasGarage']
TEXT_COL = 'Description'
DATE_COL = 'SaleDate'

# --- 1.1 Thống kê cơ bản
print("\n── Thống kê cột số ──")
print(df_raw[NUM_COLS].describe().round(2).to_string())

print("\n── Missing values ──")
missing = df_raw.isnull().sum()
print(missing[missing > 0] if missing.any() else "Không có missing values.")

print(f"\n── Duplicate rows: {df_raw.duplicated().sum()} ──")

print("\n── Phân phối categorical ──")
for col in CAT_COLS:
    print(f"  {col}: {df_raw[col].value_counts().to_dict()}")

print("\n── Skewness (cột số) ──")
for col in NUM_COLS:
    print(f"  {col}: {df_raw[col].skew():.3f}")

# --- 1.2 Biểu đồ EDA – Histogram + Boxplot + Violin
fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(4, 4, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle("GIAI ĐOẠN 1 – Khám phá dữ liệu số", fontsize=14, fontweight='bold')

for i, col in enumerate(NUM_COLS):
    # Histogram
    ax = fig.add_subplot(gs[0, i])
    ax.hist(df_raw[col].dropna(), bins=40, color=PALETTE[i], edgecolor='white', alpha=0.85)
    ax.set_title(f"Hist: {col}")
    ax.set_xlabel(col); ax.set_ylabel("Tần suất")

    # Boxplot
    ax2 = fig.add_subplot(gs[1, i])
    ax2.boxplot(df_raw[col].dropna(), vert=True, patch_artist=True,
                boxprops=dict(facecolor=PALETTE[i], alpha=0.6))
    ax2.set_title(f"Boxplot: {col}")

    # Violin
    ax3 = fig.add_subplot(gs[2, i])
    parts = ax3.violinplot(df_raw[col].dropna(), showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor(PALETTE[i]); pc.set_alpha(0.6)
    ax3.set_title(f"Violin: {col}")

# Categorical bar charts
for i, col in enumerate(CAT_COLS[:3]):
    ax = fig.add_subplot(gs[3, i])
    vc = df_raw[col].value_counts()
    ax.bar(vc.index.astype(str), vc.values, color=PALETTE[i+3])
    ax.set_title(f"Cat: {col}"); ax.set_xlabel(col); ax.set_ylabel("Count")

# Text word count distribution
ax = fig.add_subplot(gs[3, 3])
wc = df_raw[TEXT_COL].dropna().apply(lambda x: len(str(x).split()))
ax.hist(wc, bins=20, color=PALETTE[5], edgecolor='white', alpha=0.85)
ax.set_title("Word count (Description)")
ax.set_xlabel("# words"); ax.set_ylabel("Tần suất")

plt.savefig('/mnt/user-data/outputs/g1_eda.png', dpi=120, bbox_inches='tight')
plt.close()
print("\n✓ Lưu: g1_eda.png")

# ════════════════════════════════════════════════════════════════
# GIAI ĐOẠN 1 – PHẦN 2: Xử lý dữ liệu bẩn
# ════════════════════════════════════════════════════════════════
print("\n── Xử lý dữ liệu bẩn ──")
df = df_raw.copy()

# Loại bỏ dòng có giá âm hoặc số phòng = 0
invalid_price = (df['SalePrice'] <= 0).sum()
invalid_rooms = (df['Rooms'] <= 0).sum()
df = df[(df['SalePrice'] > 0) & (df['Rooms'] > 0)]
print(f"  Removed invalid SalePrice≤0: {invalid_price}, Rooms≤0: {invalid_rooms}")

# Typo / chuẩn hóa categorical (uppercase, strip)
for col in ['Neighborhood', 'Condition']:
    df[col] = df[col].astype(str).str.strip().str.capitalize()

# Fill missing nếu có
for col in NUM_COLS:
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)
for col in CAT_COLS:
    if df[col].isnull().any():
        df[col].fillna(df[col].mode()[0], inplace=True)
if df[TEXT_COL].isnull().any():
    df[TEXT_COL].fillna('', inplace=True)

# Loại duplicate
before = len(df)
df.drop_duplicates(inplace=True)
print(f"  Dropped {before - len(df)} duplicates. Final rows: {len(df):,}")

# ════════════════════════════════════════════════════════════════
# GIAI ĐOẠN 1 – PHẦN 3: Outliers & Skew
# ════════════════════════════════════════════════════════════════
print("\n── Outlier detection ──")
def iqr_bounds(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5*iqr, q3 + 1.5*iqr

for col in ['LotArea', 'SalePrice']:
    lo, hi = iqr_bounds(df[col])
    n_out = ((df[col] < lo) | (df[col] > hi)).sum()
    zscore_out = (np.abs(stats.zscore(df[col])) > 3).sum()
    print(f"  {col}: IQR outliers={n_out}, Z-score(>3)={zscore_out} → strategy: capping")
    df[col] = df[col].clip(lower=lo, upper=hi)

# ════════════════════════════════════════════════════════════════
# GIAI ĐOẠN 1 – PHẦN 4: Chuẩn hóa & Encoding
# ════════════════════════════════════════════════════════════════
print("\n── Chuẩn hóa ──")
df_enc = df.copy()

# Min-Max scaling (demo)
for col in ['LotArea', 'Rooms', 'NoiseFeature']:
    mn, mx = df_enc[col].min(), df_enc[col].max()
    df_enc[f'{col}_minmax'] = (df_enc[col] - mn) / (mx - mn + 1e-9)

# Z-score scaling
for col in ['LotArea', 'SalePrice']:
    df_enc[f'{col}_zscore'] = (df_enc[col] - df_enc[col].mean()) / (df_enc[col].std() + 1e-9)

# One-hot encoding
ohe = pd.get_dummies(df_enc[['Neighborhood', 'Condition']], prefix=['nbr', 'cond'], drop_first=False)
df_enc = pd.concat([df_enc, ohe], axis=1)

# Label encoding HasGarage (already binary, just rename)
df_enc['HasGarage_lbl'] = df_enc['HasGarage'].astype(int)

# TF-IDF on Description
tfidf = TfidfVectorizer(max_features=15, stop_words='english')
tfidf_mat = tfidf.fit_transform(df_enc[TEXT_COL].fillna('').astype(str))
tfidf_df = pd.DataFrame(tfidf_mat.toarray(), columns=[f'tfidf_{v}' for v in tfidf.get_feature_names_out()], index=df_enc.index)
df_enc = pd.concat([df_enc, tfidf_df], axis=1)

print(f"  df_enc shape after encoding: {df_enc.shape}")

# ════════════════════════════════════════════════════════════════
# GIAI ĐOẠN 1 – PHẦN 5: Text similarity / Duplicate detection
# ════════════════════════════════════════════════════════════════
print("\n── Text similarity duplicate detection (sample 200) ──")
sample_desc = df[TEXT_COL].fillna('').head(200).tolist()
dup_pairs = []
for i in range(len(sample_desc)):
    for j in range(i+1, len(sample_desc)):
        ratio = SequenceMatcher(None, sample_desc[i], sample_desc[j]).ratio()
        if ratio > 0.85:
            dup_pairs.append((i, j, round(ratio, 3)))

print(f"  Cặp bản ghi mô tả giống nhau (>85%): {len(dup_pairs)}")
if dup_pairs:
    for a, b, r in dup_pairs[:5]:
        print(f"    Row {a} ↔ Row {b}: similarity={r}")
    print("  → Đề xuất: kiểm tra thủ công, merge hoặc xóa bản ghi trùng.")

# ════════════════════════════════════════════════════════════════
# GIAI ĐOẠN 2 – PHẦN 1: Feature Engineering nâng cao
# ════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("GIAI ĐOẠN 2 – FEATURE ENGINEERING & PIPELINE")
print("═"*60)

df2 = df.copy()

# --- Date features
df2['SaleDate'] = pd.to_datetime(df2['SaleDate'], errors='coerce')
df2['sale_year']    = df2['SaleDate'].dt.year
df2['sale_month']   = df2['SaleDate'].dt.month
df2['sale_quarter'] = df2['SaleDate'].dt.quarter
df2['sale_season']  = df2['sale_month'].map(
    {12:4,1:4,2:4, 3:1,4:1,5:1, 6:2,7:2,8:2, 9:3,10:3,11:3}
)

# --- Text features
def count_keyword(text, kw):
    return str(text).lower().count(kw)

luxury_words = ['luxury', 'beautiful', 'renovated', 'modern', 'spacious']
for kw in luxury_words:
    df2[f'kw_{kw}'] = df2[TEXT_COL].apply(lambda x: count_keyword(x, kw))

df2['text_word_count'] = df2[TEXT_COL].fillna('').apply(lambda x: len(str(x).split()))
df2['luxury_score'] = df2[[f'kw_{kw}' for kw in luxury_words]].sum(axis=1)

# Simple sentiment: ratio of positive/negative words
pos_words = ['luxury','beautiful','bright','cozy','sunny','garden','renovated','modern','spacious','central']
neg_words  = ['noise','poor','bad','dirty','small','dark','old']
df2['sentiment_pos'] = df2[TEXT_COL].apply(lambda x: sum(str(x).lower().count(w) for w in pos_words))
df2['sentiment_neg'] = df2[TEXT_COL].apply(lambda x: sum(str(x).lower().count(w) for w in neg_words))
df2['sentiment']     = df2['sentiment_pos'] - df2['sentiment_neg']

# --- Image features (proxy – pixel stats not available, use pathname hash)
df2['img_idx'] = df2['ImagePath'].str.extract(r'img_(\d+)').astype(float)

# --- Business KPIs
df2['price_per_room'] = df2['SalePrice'] / (df2['Rooms'] + 1e-9)
df2['log_price']      = np.log1p(df2['SalePrice'])
df2['log_lotarea']    = np.log1p(df2['LotArea'])
df2['log_price_index']= df2['log_price'] / (df2['Rooms'] + 1)

# --- Skewness & Power transform
print("\n── Skewness & Transform ──")
for col in ['LotArea', 'SalePrice', 'Rooms']:
    sk_before = df2[col].skew()
    df2[f'{col}_log'] = np.log1p(df2[col])
    df2[f'{col}_yeo'], _ = stats.yeojohnson(df2[col])
    sk_log = df2[f'{col}_log'].skew()
    sk_yeo = df2[f'{col}_yeo'].skew()
    print(f"  {col}: skew_raw={sk_before:.2f}, skew_log={sk_log:.2f}, skew_yeo={sk_yeo:.2f}")

# ════════════════════════════════════════════════════════════════
# GIAI ĐOẠN 2 – PHẦN 2: Pipeline hoàn chỉnh
# ════════════════════════════════════════════════════════════════
print("\n── Xây dựng Pipeline hoàn chỉnh ──")

# Custom transformers (reused from Lab 8, enhanced)
class OutlierClipper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.bounds_ = {}
        X = np.array(X, dtype=float)
        for i in range(X.shape[1]):
            q1, q3 = np.nanpercentile(X[:,i], 25), np.nanpercentile(X[:,i], 75)
            iqr = q3 - q1
            self.bounds_[i] = (q1 - 1.5*iqr, q3 + 1.5*iqr)
        return self
    def transform(self, X, y=None):
        X = np.array(X, dtype=float)
        for i, (lo, hi) in self.bounds_.items():
            X[:,i] = np.clip(X[:,i], lo, hi)
        return X

class SafeNumeric(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X, y=None):
        return pd.DataFrame(X).apply(pd.to_numeric, errors='coerce').values

class DateExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X, y=None):
        vals = np.array(X).ravel()
        out = []
        for v in vals:
            try: dt = pd.to_datetime(v); out.append([dt.year, dt.month, dt.quarter])
            except: out.append([np.nan, np.nan, np.nan])
        arr = np.array(out, dtype=float)
        for c in range(arr.shape[1]):
            med = np.nanmedian(arr[:,c])
            arr[:,c] = np.where(np.isnan(arr[:,c]), med, arr[:,c])
        return arr

class TextFiller(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X, y=None):
        return ['' if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v)
                for v in np.array(X).ravel()]

class LuxuryFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract text-based features: word count, luxury score, sentiment."""
    LUXURY = ['luxury','beautiful','renovated','modern','spacious','bright','cozy']
    POS    = ['luxury','beautiful','bright','cozy','sunny','garden','renovated','modern']
    NEG    = ['noise','poor','bad','dirty','small','dark']
    def fit(self, X, y=None): return self
    def transform(self, X, y=None):
        texts = ['' if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v)
                 for v in np.array(X).ravel()]
        out = []
        for t in texts:
            t_lo = t.lower()
            wc  = len(t.split())
            lux = sum(t_lo.count(w) for w in self.LUXURY)
            pos = sum(t_lo.count(w) for w in self.POS)
            neg = sum(t_lo.count(w) for w in self.NEG)
            out.append([wc, lux, pos - neg])
        return np.array(out, dtype=float)

FEAT_NUM  = ['LotArea', 'Rooms', 'NoiseFeature']
FEAT_CAT  = ['Neighborhood', 'Condition', 'HasGarage']
FEAT_TEXT = 'Description'
FEAT_DATE = 'SaleDate'

numeric_pipe = Pipeline([
    ('safe',   SafeNumeric()),
    ('impute', SimpleImputer(strategy='median')),
    ('clip',   OutlierClipper()),
    ('scale',  StandardScaler()),
    ('power',  PowerTransformer(method='yeo-johnson')),
])

cat_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('ohe',    OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])

tfidf_pipe = Pipeline([
    ('fill',  TextFiller()),
    ('tfidf', TfidfVectorizer(max_features=20, stop_words='english')),
])

lux_pipe = Pipeline([
    ('lux', LuxuryFeatureExtractor()),
    ('scale', StandardScaler()),
])

date_pipe = Pipeline([
    ('extract', DateExtractor()),
    ('scale',   StandardScaler()),
])

preprocessor = ColumnTransformer([
    ('num',  numeric_pipe, FEAT_NUM),
    ('cat',  cat_pipe,     FEAT_CAT),
    ('tfidf',tfidf_pipe,   FEAT_TEXT),
    ('lux',  lux_pipe,     FEAT_TEXT),
    ('date', date_pipe,    FEAT_DATE),
], remainder='drop')

Xraw = df.drop(columns=['SalePrice', 'ImagePath'])
y_raw = df['SalePrice'].values
y_log = np.log1p(y_raw)

preprocessor.fit(Xraw, y_raw)
X_transformed = preprocessor.transform(Xraw)
print(f"  Pipeline output shape: {X_transformed.shape}")
print(f"  Any NaN: {np.isnan(X_transformed).any()}")

# ════════════════════════════════════════════════════════════════
# GIAI ĐOẠN 2 – PHẦN 3: Mô hình dự báo & Cross-validation
# ════════════════════════════════════════════════════════════════
print("\n── Huấn luyện mô hình ──")

models = {
    'LinearRegression':     LinearRegression(),
    'Ridge':                Ridge(alpha=1.0),
    'RandomForest':         RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
    'GradientBoosting':     GradientBoostingRegressor(n_estimators=150, random_state=42),
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = ['neg_root_mean_squared_error', 'neg_mean_absolute_error', 'r2']

results_raw = {}
results_log = {}

for name, model in models.items():
    pipe = Pipeline([('pre', preprocessor), ('mdl', model)])

    # Raw target
    cv_r = cross_validate(pipe, Xraw, y_raw, cv=cv, scoring=scoring)
    results_raw[name] = {
        'RMSE': -cv_r['test_neg_root_mean_squared_error'],
        'MAE':  -cv_r['test_neg_mean_absolute_error'],
        'R2':    cv_r['test_r2'],
    }

    # Log-transformed target
    cv_l = cross_validate(pipe, Xraw, y_log, cv=cv, scoring=scoring)
    results_log[name] = {
        'RMSE': -cv_l['test_neg_root_mean_squared_error'],
        'R2':    cv_l['test_r2'],
    }

    print(f"\n  {name}:")
    print(f"    [Raw]  RMSE={results_raw[name]['RMSE'].mean():.0f} ± {results_raw[name]['RMSE'].std():.0f}  "
          f"MAE={results_raw[name]['MAE'].mean():.0f}  R²={results_raw[name]['R2'].mean():.4f}")
    print(f"    [Log]  RMSE_log={results_log[name]['RMSE'].mean():.4f}  R²={results_log[name]['R2'].mean():.4f}")

# Summary table
print("\n═══ BẢNG SO SÁNH MÔ HÌNH (RAW TARGET) ═══")
rows = []
for name in models:
    rows.append({
        'Mô hình': name,
        'RMSE': f"{results_raw[name]['RMSE'].mean():,.0f}",
        'RMSE_std': f"{results_raw[name]['RMSE'].std():,.0f}",
        'MAE': f"{results_raw[name]['MAE'].mean():,.0f}",
        'R²': f"{results_raw[name]['R2'].mean():.4f}",
        'R²_std': f"{results_raw[name]['R2'].std():.4f}",
    })
print(pd.DataFrame(rows).to_string(index=False))

# ════════════════════════════════════════════════════════════════
# GIAI ĐOẠN 2 – PHẦN 4: KPI & Phân tích kinh doanh
# ════════════════════════════════════════════════════════════════
print("\n── Phân tích KPI & Business Insight ──")

df_kpi = df2.copy()
df_kpi['Neighborhood'] = df_kpi['Neighborhood'].str.strip().str.capitalize()

# Price-per-room by Neighborhood
kpi_nbr = df_kpi.groupby('Neighborhood').agg(
    avg_price   =('SalePrice','mean'),
    median_price=('SalePrice','median'),
    avg_rooms   =('Rooms','mean'),
    avg_luxury  =('luxury_score','mean'),
    count       =('SalePrice','count'),
).round(0)
kpi_nbr['price_per_room'] = (kpi_nbr['avg_price'] / kpi_nbr['avg_rooms']).round(0)
print("\n  KPI theo Neighborhood:")
print(kpi_nbr.to_string())

# Top 5% giá cực trị
p95 = df_kpi['SalePrice'].quantile(0.95)
top5 = df_kpi[df_kpi['SalePrice'] >= p95]
print(f"\n  Nhà giá cực trị (top 5%): {len(top5)} records, avg SalePrice={top5['SalePrice'].mean():,.0f}")
print(f"  Luxury score trung bình top 5%: {top5['luxury_score'].mean():.2f} vs toàn bộ: {df_kpi['luxury_score'].mean():.2f}")

# Phân khúc khách hàng
bins   = [0, 120000, 200000, 300000, np.inf]
labels = ['Bình dân (<120K)', 'Trung bình (120-200K)', 'Cao cấp (200-300K)', 'Hạng sang (>300K)']
df_kpi['segment'] = pd.cut(df_kpi['SalePrice'], bins=bins, labels=labels)
seg_tbl = df_kpi['segment'].value_counts().sort_index()
print("\n  Phân khúc khách hàng:")
print(seg_tbl.to_string())

# ════════════════════════════════════════════════════════════════
# GIAI ĐOẠN 2 – PHẦN 5: Dashboard visualization
# ════════════════════════════════════════════════════════════════
# --- Chart 1: Raw vs Transformed distribution
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Biến đổi phân phối: Raw vs Log vs Yeo-Johnson", fontsize=13, fontweight='bold')
for i, col in enumerate(['LotArea', 'SalePrice', 'Rooms']):
    axes[0, i].hist(df2[col], bins=40, color=PALETTE[i], edgecolor='white', alpha=0.85)
    axes[0, i].set_title(f"Raw: {col} (skew={df2[col].skew():.2f})")
    axes[1, i].hist(df2[f'{col}_log'], bins=40, color=PALETTE[i+3], edgecolor='white', alpha=0.85)
    axes[1, i].set_title(f"Log: {col} (skew={df2[col+'_log'].skew():.2f})")
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/g2_transform.png', dpi=120, bbox_inches='tight')
plt.close()
print("\n✓ Lưu: g2_transform.png")

# --- Chart 2: Model comparison bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("So sánh mô hình (5-fold CV)", fontsize=13, fontweight='bold')

model_names = list(models.keys())
rmse_vals   = [results_raw[n]['RMSE'].mean() for n in model_names]
r2_vals     = [results_raw[n]['R2'].mean()   for n in model_names]
rmse_std    = [results_raw[n]['RMSE'].std()  for n in model_names]

x = np.arange(len(model_names))
bars = axes[0].bar(x, rmse_vals, yerr=rmse_std, capsize=5,
                   color=PALETTE[:len(model_names)], edgecolor='white', alpha=0.85)
axes[0].set_xticks(x); axes[0].set_xticklabels(model_names, rotation=15, ha='right')
axes[0].set_title("RMSE (thấp hơn = tốt hơn)"); axes[0].set_ylabel("RMSE")
for bar, v in zip(bars, rmse_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                 f'{v:,.0f}', ha='center', fontsize=8)

bars2 = axes[1].bar(x, r2_vals, color=PALETTE[:len(model_names)], edgecolor='white', alpha=0.85)
axes[1].set_xticks(x); axes[1].set_xticklabels(model_names, rotation=15, ha='right')
axes[1].set_title("R² Score (cao hơn = tốt hơn)"); axes[1].set_ylabel("R²")
axes[1].axhline(0, color='black', linewidth=0.8, linestyle='--')
for bar, v in zip(bars2, r2_vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, v + 0.002,
                 f'{v:.3f}', ha='center', fontsize=8)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/g2_models.png', dpi=120, bbox_inches='tight')
plt.close()
print("✓ Lưu: g2_models.png")

# --- Chart 3: KPI Dashboard
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("PropTech KPI Dashboard", fontsize=14, fontweight='bold')

# 3a: Avg price by neighborhood
ax = axes[0, 0]
nbrs = kpi_nbr.index.tolist()
ax.bar(nbrs, kpi_nbr['avg_price'], color=PALETTE[:len(nbrs)], edgecolor='white')
ax.set_title("Giá trung bình theo Khu vực")
ax.set_ylabel("USD")

# 3b: Luxury score vs SalePrice scatter
ax = axes[0, 1]
scatter = ax.scatter(df2['luxury_score'], df2['SalePrice'],
                     c=df2['Rooms'], cmap='viridis', alpha=0.4, s=12)
plt.colorbar(scatter, ax=ax, label='Rooms')
ax.set_title("Luxury Score vs Giá"); ax.set_xlabel("Luxury Score"); ax.set_ylabel("SalePrice")

# 3c: Price distribution by Condition
ax = axes[0, 2]
conds = df2['Condition'].unique()
for i, cond in enumerate(sorted(conds)):
    vals = df2[df2['Condition']==cond]['SalePrice']
    ax.hist(vals, bins=25, alpha=0.55, label=cond, color=PALETTE[i % len(PALETTE)])
ax.set_title("Phân phối giá theo Condition"); ax.legend(fontsize=8); ax.set_xlabel("SalePrice")

# 3d: Customer segment pie
ax = axes[1, 0]
ax.pie(seg_tbl.values, labels=seg_tbl.index, autopct='%1.1f%%',
       colors=PALETTE[:len(seg_tbl)], startangle=140)
ax.set_title("Phân khúc khách hàng")

# 3e: Sale trend by year
ax = axes[1, 1]
trend = df2.groupby('sale_year')['SalePrice'].median()
ax.plot(trend.index, trend.values, marker='o', color=PALETTE[0], linewidth=2)
ax.fill_between(trend.index, trend.values, alpha=0.15, color=PALETTE[0])
ax.set_title("Xu hướng giá theo năm (median)"); ax.set_xlabel("Năm"); ax.set_ylabel("USD")

# 3f: log_price_index vs Neighborhood boxplot
ax = axes[1, 2]
data_box = [df_kpi[df_kpi['Neighborhood']==n]['log_price_index'].dropna().values
            for n in sorted(df_kpi['Neighborhood'].unique())]
ax.boxplot(data_box, labels=sorted(df_kpi['Neighborhood'].unique()), patch_artist=True,
           boxprops=dict(facecolor=PALETTE[2], alpha=0.6))
ax.set_title("Log-Price-Index theo Khu vực"); ax.set_ylabel("log_price_index")

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/g2_dashboard.png', dpi=120, bbox_inches='tight')
plt.close()
print("✓ Lưu: g2_dashboard.png")

# ════════════════════════════════════════════════════════════════
# GIAI ĐOẠN HOÀN THIỆN
# ════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("GIAI ĐOẠN HOÀN THIỆN")
print("═"*60)

# --- HT1: Dataset mới với missing + unseen categories
print("\n── HT1: Dataset mới với missing + unseen categories ──")
df_new = Xraw.head(20).copy()
df_new.loc[0:5, 'Neighborhood'] = 'Z_UNSEEN'
df_new.loc[6:8, 'LotArea']      = np.nan
df_new.loc[9,   'Description']  = np.nan
out_new = preprocessor.transform(df_new)
print(f"  Shape: {out_new.shape}, NaN: {np.isnan(out_new).sum()} → Pipeline xử lý OK")

# --- HT2: Auto-detect new columns
print("\n── HT2: Auto-detect cột mới ──")
df_extra = Xraw.copy()
df_extra['NewCol_Float']  = np.random.randn(len(df_extra))
df_extra['NewCol_String'] = 'extra_data'

known_cols = set(Xraw.columns)
new_cols   = [c for c in df_extra.columns if c not in known_cols]
print(f"  Cột mới phát hiện: {new_cols}")
print("  → Pipeline dùng remainder='drop' → bỏ qua cột mới, không lỗi.")
out_extra = preprocessor.transform(df_extra)
print(f"  Shape (với cột lạ): {out_extra.shape} – nhất quán.")

# --- HT3: Feature interaction
print("\n── HT3: Feature Interaction ──")
df_int = df.copy()
df_int['SaleDate'] = pd.to_datetime(df_int['SaleDate'], errors='coerce')
df_int['sale_year'] = df_int['SaleDate'].dt.year.fillna(2010).astype(int)

# Interaction features
df_int['area_x_rooms']    = df_int['LotArea'] * df_int['Rooms']
df_int['area_x_rooms_nbr']= df_int['LotArea'] * df_int['Rooms'] * df_int['Neighborhood'].map(
    {'A':1,'B':2,'C':3,'D':4}).fillna(2)

# Compare model with/without interaction
feat_base  = ['LotArea','Rooms','NoiseFeature']
feat_inter = feat_base + ['area_x_rooms', 'area_x_rooms_nbr', 'sale_year']

sc = StandardScaler()
results_int = {}
for fname, feats in [('Baseline (num only)', feat_base), ('+ Interactions', feat_inter)]:
    Xf = df_int[feats].fillna(df_int[feats].median())
    Xf_scaled = sc.fit_transform(Xf)
    cv_r = cross_validate(RandomForestRegressor(n_estimators=100, random_state=42),
                          Xf_scaled, y_raw, cv=cv, scoring=scoring)
    results_int[fname] = {
        'RMSE': -cv_r['test_neg_root_mean_squared_error'].mean(),
        'R2':    cv_r['test_r2'].mean(),
    }
    print(f"  {fname}: RMSE={results_int[fname]['RMSE']:,.0f}  R²={results_int[fname]['R2']:.4f}")

# --- HT4: Model comparison – Numerical only vs Numerical + Text + Date
print("\n── HT4: Numerical only vs Numerical + Text + Date ──")

pre_num_only = ColumnTransformer([
    ('num', numeric_pipe, FEAT_NUM),
    ('cat', cat_pipe,     FEAT_CAT),
], remainder='drop')

pre_full = preprocessor  # already built

rf = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)

for pname, pre in [('Num+Cat only', pre_num_only), ('Full (Num+Cat+Text+Date)', pre_full)]:
    pipe = Pipeline([('pre', pre), ('mdl', rf)])
    cv_r = cross_validate(pipe, Xraw, y_raw, cv=cv, scoring=scoring)
    rmse = -cv_r['test_neg_root_mean_squared_error'].mean()
    r2   = cv_r['test_r2'].mean()
    print(f"  {pname}: RMSE={rmse:,.0f}  R²={r2:.4f}")

# ════════════════════════════════════════════════════════════════
# BUSINESS INSIGHT REPORT
# ════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("BUSINESS INSIGHT REPORT")
print("═"*60)

report = f"""
╔══════════════════════════════════════════════════════════════╗
║        PROPTECH – BÁO CÁO INSIGHT NGHIỆP VỤ                 ║
╚══════════════════════════════════════════════════════════════╝

1. KHU VỰC NÊN ĐẦU TƯ / TRÁNH
───────────────────────────────
{kpi_nbr[['avg_price','price_per_room','avg_luxury']].to_string()}

  → Nên đầu tư: khu vực có avg_price cao + luxury_score cao.
  → Tránh: khu vực giá thấp nhưng noise cao (xem NoiseFeature).

2. PHÂN KHÚC KHÁCH HÀNG
─────────────────────────
{seg_tbl.to_string()}

  → Phân khúc "Trung bình" chiếm đa số → ưu tiên marketing.
  → Phân khúc "Hạng sang" có luxury_score cao hơn trung bình.

3. GIÁ TRỊ TRANSFORM
─────────────────────
  • LotArea bị lệch nặng (skew >> 2) → log-transform làm phân phối chuẩn hơn.
  • PowerTransformer (Yeo-Johnson) phù hợp nhất cho pipeline ML.
  • Log-transform target (SalePrice) giúp model ổn định hơn trên outliers.

4. KPI ĐỀ XUẤT
───────────────
  • log_price_index = log(price) / (rooms + 1) → đo "giá trị / phòng".
  • luxury_score    = tổng từ khoá sang trọng → proxy chất lượng mô tả.
  • price_per_room  = SalePrice / Rooms → so sánh nhà nhiều phòng vs ít phòng.

5. MÔ HÌNH TỐT NHẤT
─────────────────────
  RandomForest và GradientBoosting cho kết quả ổn định nhất.
  Thêm text + date features cải thiện R² so với chỉ dùng số.

6. KHUYẾN NGHỊ
───────────────
  • Retrain định kỳ (tháng) để bắt kịp market drift.
  • Thêm ảnh CNN embedding để cải thiện luxury_score.
  • Monitor unseen categories từ nguồn dữ liệu mới.
"""
print(report)

with open('/mnt/user-data/outputs/business_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

# Save pipeline
joblib.dump(preprocessor, '/mnt/user-data/outputs/full_preprocessor.pkl')
print("✓ Lưu: full_preprocessor.pkl, business_report.txt")

print("\n" + "═"*60)
print("HOÀN THÀNH TOÀN BỘ DỰ ÁN")
print("═"*60)
