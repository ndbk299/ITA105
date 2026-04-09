import tensorflow as tf
import numpy as np

# tao dataset
X = np.array([[1], [2], [3], [4], [5]], dtype=float)
y = np.array([[2], [4], [6], [8], [10]], dtype=float)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])
model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(X, y, epochs=100)
print(model.predict(np.array([[6]])))