import pandas as pd
import numpy as np
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

def data_cleaning_pipeline(df, median, q1, q3, iqr):
    df_clean = df.copy()
    df_clean['Value'] = df_clean['Value'].fillna(median)
    df_clean = df_clean[(df_clean['Value'] >= q1 - 1.5 * iqr) & (df_clean['Value'] <= q3 + 1.5 * iqr)]
    return df_clean

data = [5, pd.NA, 15, 200, 25]
df2 = pd.DataFrame(data, columns=['Value'])
print(df2)

missing_value2 = df2.isnull().sum()
print(f"Số lượng missing values: \n{missing_value2}")

Q1_2 = df2['Value'].quantile(0.25)
Q3_2 = df2['Value'].quantile(0.75)
IQR_2 = Q3_2 - Q1_2
outliers2 = df2[(df2['Value'] < Q1_2 - 1.5 * IQR_2) | (df2['Value'] > Q3_2 + 1.5 * IQR_2)]
print(f"Số lượng outliers: \n{outliers2}")
median_value2 = df2['Value'].median()
print(f"Median: {median_value2}")
cleaned_df2 = data_cleaning_pipeline(df2, median_value2, Q1_2, Q3_2, IQR_2)
print("Data sau khi được làm sạch:")
print(cleaned_df2)

data = [8, 10, pd.NA, 12, 500]
df3 = pd.DataFrame(data, columns=['Value'])
print(df3)
missing_value3 = df3.isnull().sum()
print(f"Số lượng missing values: \n{missing_value3}")
Q1_3=df3['Value'].quantile(0.25)
Q3_3=df3['Value'].quantile(0.75)
IQR=Q3_3 - Q1_3
outliers3 = df3[(df3['Value'] < Q1_3 - 1.5 * IQR) & (df3['Value'] > Q3_3 +1.5 * IQR)]
print(f"Số lượng outliers: \n{outliers3}")
meadian_value3 = df3['Value'].median()
print(f"Median: {meadian_value3}")
def data_cleaning_pipeline(df, median,Q1,Q3,IQR):
    df_clean = df.copy()
    df_clean['Value'] = df_clean['Value'].fillna(median)
    df_clean=df_clean[(df_clean['Value'] >= Q1_3 - 1.5 * IQR) & ( df_clean['Value']<= Q3_3 + 1.5 * IQR)]
    return df_clean
cleaned_df3 = data_cleaning_pipeline(df3, meadian_value3, Q1_3, Q3_3, IQR)
print("Data sau khi được làm sạch:")
print(cleaned_df3)