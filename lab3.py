import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Bài 1 Thông số vận động viên

# - Nạp dữ liệu, kiểm tra missing values, thống kê mô tả.

sporter = pd.read_csv(r'd:\FPT\ITA105\LMS-ITA105\Lab3\Lab3\ITA105_Lab_3_Sports.csv', encoding='utf-8')
print(sporter.shape)
print(sporter.isnull().sum())
print(sporter.describe())

# - Vẽ histogram và boxplot cho từng biến

numerical_cols = ['chieu_cao_cm', 'can_nang_kg', 'toc_do_100m_s', 'so_ban_thang', 'so_phut_thi_dau']
for col in numerical_cols:
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(sporter[col].dropna(), kde=True)
    plt.title(f'Histogram of {col}')
    plt.subplot(1, 2, 2)
    sns.boxplot(x=sporter[col].dropna())
    plt.title(f'Boxplot of {col}')
    plt.show()

# - Chuẩn hóa từng biến bằng Min-Max Scaling → đưa về [0,1].
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
sporter[numerical_cols] = scaler.fit_transform(sporter[numerical_cols])
print(sporter.head())

# - Chuẩn hóa từng biến bằng Z-Score Normalization → mean = 0, std = 1.
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
sporter[numerical_cols] = scaler.fit_transform(sporter[numerical_cols])
print(sporter.head())

# - Vẽ biểu đồ so sánh phân phối dữ liệu trước và sau chuẩn hóa (Min-Max và Z-Score).
for col in numerical_cols:
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    sns.histplot(sporter[col].dropna(), kde=True)
    plt.title(f'Original Distribution of {col}')
    plt.subplot(1, 3, 2)
    sns.histplot(scaler.fit_transform(sporter[[col]]).flatten(), kde=True)
    plt.title(f'Min-Max Scaled Distribution of {col}')
    plt.subplot(1, 3, 3)
    sns.histplot(StandardScaler().fit_transform(sporter[[col]]).flatten(), kde=True)
    plt.title(f'Z-Score Normalized Distribution of {col}')
    plt.show()

# Bài 2 Chỉ số bệnh nhân

# - Khám phá dữ liệu, thống kê, trực quan hóa (histogram,boxplot).
patient = pd.read_csv(r'd:\FPT\ITA105\LMS-ITA105\Lab3\Lab3\ITA105_Lab_3_Patient.csv', encoding='utf-8')
print(patient.shape)
print(patient.isnull().sum())
print(patient.describe())

numerical_cols = ['tuoi', 'can_nang_kg', 'huyet_ap_mmHg', 'nhip_tim_bpm', 'muc_do_dau'] 
for col in numerical_cols:
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(patient[col].dropna(), kde=True)
    plt.title(f'Histogram of {col}')
    plt.subplot(1, 2, 2)
    sns.boxplot(x=patient[col].dropna())
    plt.title(f'Boxplot of {col}')
    plt.show()

# - Phát hiện ngoại lệ (bệnh nhân cực cao/nhỏ, huyết áp cực đoan).
for col in numerical_cols:
    plt.figure(figsize=(12, 5))
    sns.boxplot(x=patient[col].dropna())
    plt.title(f'Boxplot of {col} for Outlier Detection')
    plt.show()  

# - Chuẩn hóa bằng Min-Max và Z-Score.
scaler = MinMaxScaler()
patient[numerical_cols] = scaler.fit_transform(patient[numerical_cols])
print(patient.head())
scaler = StandardScaler()
patient[numerical_cols] = scaler.fit_transform(patient[numerical_cols])
print(patient.head())

# - So sánh phân phối trước và sau chuẩn hóa.
for col in numerical_cols:
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    sns.histplot(patient[col].dropna(), kde=True)
    plt.title(f'Original Distribution of {col}')
    plt.subplot(1, 3, 2)
    sns.histplot(scaler.fit_transform(patient[[col]]).flatten(), kde=True)
    plt.title(f'Min-Max Scaled Distribution of {col}')
    plt.subplot(1, 3, 3)
    sns.histplot(StandardScaler().fit_transform(patient[[col]]).flatten(), kde=True)
    plt.title(f'Z-Score Normalized Distribution of {col}')
    plt.show()

# - Nhận xét: biến nào bị ảnh hưởng nhiều bởi ngoại lệ, và phương pháp chuẩn hóa nào phù hợp hơn?
# Min-Max dễ bị ảnh hưởng bởi ngoại lệ (ví dụ huyết áp 250).
# Z-Score ổn định hơn khi có outlier.

# Bài 3 Chỉ số công ty

# - Khám phá dataset, vẽ boxplot để quan sát scale khác nhau và ngoại lệ (công ty cực lớn).
company = pd.read_csv(r'd:\FPT\ITA105\LMS-ITA105\Lab3\Lab3\ITA105_Lab_3_Company.csv', encoding='utf-8')
print(company.shape)
print(company.isnull().sum())
print(company.describe())

numerical_cols = ['doanh_thu_trieu_usd', 'loi_nhuan_trieu_usd', 'so_nhan_vien', 'gia_tri_tai_san_trieu_usd']
for col in numerical_cols:
    plt.figure(figsize=(12, 5))
    sns.boxplot(x=company[col].dropna())
    plt.title(f'Boxplot of {col} for Scale and Outlier Detection')
    plt.show()

# - Chuẩn hóa bằng Min-Max và Z-Score.
scaler = MinMaxScaler()
company[numerical_cols] = scaler.fit_transform(company[numerical_cols])
print(company.head())
scaler = StandardScaler()
company[numerical_cols] = scaler.fit_transform(company[numerical_cols])
print(company.head())

# - Vẽ scatterplot so sánh 2 biến trước và sau chuẩn hóa (ví dụ:Doanh thu và Lợi nhuận).
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)    
sns.scatterplot(x=company['doanh_thu_trieu_usd'], y=company['loi_nhuan_trieu_usd'])
plt.title('Scatterplot of Doanh thu vs Lợi nhuận (Original)')
plt.subplot(1, 2, 2)
sns.scatterplot(x=StandardScaler().fit_transform(company[['doanh_thu_trieu_usd']]).flatten(), y=StandardScaler().fit_transform(company[['loi_nhuan_trieu_usd']]).flatten())
plt.title('Scatterplot of Doanh thu vs Lợi nhuận (Z-Score Normalized)')
plt.show()

# - Nhận xét: dữ liệu có ngoại lệ lớn → Min-Max có phù hợp không?
# Min-Max bị kéo lệch mạnh bởi công ty cực lớn.

# - Thảo luận: chọn phương pháp chuẩn hóa cho mô hình dự đoán tài chính (Linear Regression, KNN).
# Z-Score phù hợp hơn cho mô hình dự đoán (Linear Regression, KNN).

# Bài 4 Người chơi trực tuyến

# - Khám phá dữ liệu, kiểm tra missing values, trực quan hóa phân phối.
gamer = pd.read_csv(r'd:\FPT\ITA105\LMS-ITA105\Lab3\Lab3\ITA105_Lab_3_Gamer.csv', encoding='utf-8')
print(gamer.shape)
print(gamer.isnull().sum())
print(gamer.describe())

numerical_cols = ['thoi_gian_choi_gio', 'so_lan_dang_nhap_trong_thang', 'so_ban_thang_trong_game', 'do_tuoi']
for col in numerical_cols:  
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(gamer[col].dropna(), kde=True)
    plt.title(f'Histogram of {col}')
    plt.subplot(1, 2, 2)
    sns.boxplot(x=gamer[col].dropna())
    plt.title(f'Boxplot of {col}')
    plt.show()

# - Chuẩn hóa bằng Min-Max và Z-Score.
scaler = MinMaxScaler()
gamer[numerical_cols] = scaler.fit_transform(gamer[numerical_cols])
print(gamer.head())
scaler = StandardScaler()
gamer[numerical_cols] = scaler.fit_transform(gamer[numerical_cols])
print(gamer.head())

# - Vẽ histogram so sánh phân phối trước và sau chuẩn hóa.
for col in numerical_cols:
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    sns.histplot(gamer[col].dropna(), kde=True)
    plt.title(f'Original Distribution of {col}')
    plt.subplot(1, 3, 2)
    sns.histplot(scaler.fit_transform(gamer[[col]]).flatten(), kde=True)
    plt.title(f'Min-Max Scaled Distribution of {col}')
    plt.subplot(1, 3, 3)
    sns.histplot(StandardScaler().fit_transform(gamer[[col]]).flatten(), kde=True)
    plt.title(f'Z-Score Normalized Distribution of {col}')
    plt.show()

# - Thảo luận: một số người chơi cực kỳ “cày cuốc” → ngoại lệ, phương pháp nào ổn hơn?
# Min-Max bị ảnh hưởng bởi người chơi cực đoan.
# Z-Score giữ được phân phối hợp lý hơn.

# - Chuẩn hóa dữ liệu để chuẩn bị cho mô hình clustering hoặc KNN (highlight lý do chọn phương pháp).
# Ứng dụng: chuẩn hóa để chuẩn bị cho clustering hoặc KNN.