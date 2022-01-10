import tensorflow as tf
import pandas as pd

(ind_val, dep_val), _ = tf.keras.datasets.mnist.load_data()
ind_val = ind_val.reshape(60000, 28, 28, 1)
dep_val = pd.get_dummies(dep_val)

# 나의 로드맵
print(ind_val.shape, dep_val.shape) # (60000, 28, 28, 1) (60000, 10)

X = tf.keras.layers.Input(shape=[28, 28, 1])

HC1 = tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='swish')(X) 
HS2 = tf.keras.layers.MaxPool2D()(HC1)

HC3 = tf.keras.layers.Conv2D(filters=16, kernel_size=5, activation='swish')(HS2)
HS4 = tf.keras.layers.MaxPool2D()(HC3)

HF5 = tf.keras.layers.Flatten()(HS4)

HD6 = tf.keras.layers.Dense(units=120, activation='swish')(HF5)
HD7 = tf.keras.layers.Dense(units=84, activation='swish')(HF5)

Y = tf.keras.layers.Dense(units=10, activation='softmax')(HD7)

model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy', metrics='accuracy')

# 학습
model.fit(ind_val, dep_val, epochs=10)