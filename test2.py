# Machine Learning
# import thư viện linear regression
from sklearn.linear_model import LinearRegression
# import numpy để nạp dữ liệu
import numpy as np
# tạo input (giờ học)
# reshape (-1,1) để tạo ma trận cột
hours = np.array([2,3,5,8]).reshape(-1, 1)
# output (điểm)
scores = np.array ([4,5,7,9])
# tạo model
model = LinearRegression()
# huấn luyện model
model.fit(hours, scores)
# dự đoán điểm khi học 6 giờ
prediction = model.predict(np.array([[6]]))
print(f"Kết quả khi học 6 giờ: {prediction[0]}")
#impory thu vien ve bieu do
import matplotlib.pyplot as plt
#ve bieu do
plt.scatter(hours,scores)
#tinh gia tri du doan cho cac diem
pred = model.predict(hours)
#ve bieu do du doan
plt.plot(hours,pred,color='red')
#nhãn trục
plt.xlabel("So gio hoc")
plt.ylabel("Diem")
#hien thi
plt.show()

#=====================ML du doan gia nha============================
#1. import thu vien
import pandas as pd #xu ly du lieu
from sklearn.model_selection import train_test_split #chia du lieu
from sklearn.linear_model import LinearRegression #tao model
import tensorflow as tf
import numpy as np

#2. Dataset
data = {
"Area":[50,60,80,100,120],
"Price": [100,120,150,200,230]
}
#tao DataFrame
df = pd.DataFrame(data)
# input
x = df.drop("Price",axis=1)
# output
y = df["Price"]
#model
model = LinearRegression()
#train
model.fit(x,y)
#du doan
pred = model.predict([[70]])
# Dự đoán giá nhà 70m2
print("Giá nhà dự đoán: ",pred)
#=====================deep learning==============
#2. tao du lieu
x= np.array([1,2,3,4,5])
y= np.array([2,4,6,8,10])
#3. tao model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1,input_shape=(1,)))
#4. train model
model.compile(optimizer='sgd',loss='mean_squared_error')
model.fit(x,y,epochs=100)
#5.du doan ket qua
pred = model.predict(np.array(([6])))
#6. in ket qua du doan 
print("Ket qua du doan cuar deep learning: ",pred)
#======================ML vẽ biểu đồ==============================