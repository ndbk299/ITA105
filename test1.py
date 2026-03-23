import pandas as pd
data = {"Name": ["Alice", "Bob", "Charlie"],
        "Age": [25, None, 35],
        "City": ["New York", "Abc", "Los Angeles"]
        }
df=pd.DataFrame(data)
df.to_csv("test.csv",index=False)
print("File created successfully.")
df=pd.read_csv("test.csv")
print(df)
missing=df.isnull().sum()
print("Missing values in each column:")
print(missing)
mean_age=df["Age"].mean()
df["Age"].fillna(mean_age,inplace=True)
print("DataFrame after filling missing values:")
print(df)
max_age=df["Age"].max()
print("Maximum age:",max_age)
