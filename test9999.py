import pandas as pd
data = [10, pd.NA, 20, 1000, 30]
df = pd.DataFrame(data, columns=['Value'])
print(df)

# Xác định missing
missing_value = df.isnull().sum()
print(f"Số lượng missing values: \n{missing_value}")

# Xác định outlier
Q1 = df['Value'].quantile(0.25)
Q3 = df['Value'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['Value'] < Q1 - 1.5 * IQR) | (df['Value'] > Q3 + 1.5 * IQR)]
print(f"Số lượng outliers: \n{outliers}")

# Tính median
median_value = df['Value'].median()
print(f"Median: {median_value}")

# Đề xuất pipeline xử lý
def data_cleaning_pipeline(df):
    # Điền missing values bằng median
    df_clean = df.copy()
    df_clean['Value'] = df_clean['Value'].fillna(median_value)
    # Loại bỏ outliers
    df_clean = df_clean[(df_clean['Value'] >= Q1 - 1.5 * IQR) & (df_clean['Value'] <= Q3 + 1.5 * IQR)]
    return df_clean
cleaned_df = data_cleaning_pipeline(df)
print("Data sau khi được làm sạch:")
print(cleaned_df)
