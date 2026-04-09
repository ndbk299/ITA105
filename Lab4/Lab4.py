import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

data={
  "Hours": [1,2,3,4,5,6,7,8],
  "Scores": [2,4,5,6,7,8,8.5,9]
 }

df = pd.DataFrame(data)
print(df)

X = df[["Hours"]]
y = df["Scores"]

model = LinearRegression()
model.fit(X, y)
print("Coefficient:", model.coef_)
print("Intercept:", model.intercept_)
y_pred = model.predict(X)
plt.scatter(X, y, color='blue', label='Actual Scores')
plt.plot(X, y_pred, color='red', label='Predicted Scores')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.legend()
plt.show()

new_hours = [[6]]
predicted_score = model.predict(new_hours)
print(f"Predicted score for {new_hours[0][0]} hours: {predicted_score[0]}")

new_data = [[4],[6],[9]]
predicted_scores = model.predict(new_data)
for i, score in enumerate(predicted_scores):
    print(f"Predicted score for {new_data[i][0]} hours: {score}")
    plt.scatter(new_data[i][0], score, color='green')
plt.show()  

plt.scatter(X, y, color='blue', label='Actual Scores')
plt.plot(X, y_pred, color='red', label='Predicted Scores')
plt.scatter(new_data, predicted_scores, color='green', label='New Predictions')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.title("Hours vs Scores with Linear Regression")
plt.legend()
plt.show()

y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)


