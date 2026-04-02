import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

hours = np.array ([1,2,3,4,5,6]).reshape(-1,1)
pass_exam = np.array([0,0,0,1,1,1])
model = LogisticRegression()

model.fit(hours, pass_exam)

prediction = model.predict([[4]])
print(prediction)

prob = model.predict_proba([[4]])
print(prob)

plt.scatter(hours, pass_exam)
plt.xlabel("Hours Studied")
plt.ylabel("Pass Exam")
plt.title("Logistic Regression: Hours vs Pass Exam")
plt.show()
