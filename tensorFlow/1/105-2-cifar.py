import tensorflow as tf
import pandas as pd

# 비교를 위한...
# (ind_val, dep_val), _ = tf.keras.datasets.mnist.load_data()
# print(ind_val.shape, dep_val.shape) 
# mnist - (60000, 28, 28) (60000,)
(ind_val, dep_val), _ = tf.keras.datasets.cifar10.load_data()
print(ind_val.shape, dep_val.shape) 
    # >>> (50000, 32, 32, 3) (50000, 1)
try :
    dep_val = pd.get_dummies(dep_val) # raise ValueError("Data must be 1-dimensional")
except ValueError : 
    print('ValueError 발생 reshape후 > pd.get_dummies!')
    dep_val = dep_val.reshape(50000,)
    print(ind_val.shape, dep_val.shape)
        # >>> (50000, 32, 32, 3) (50000,)
    dep_val = pd.get_dummies(dep_val)
print(ind_val.shape, dep_val.shape)
    # >>> (50000, 32, 32, 3) (50000, 10)

# 나의 로드맵
print(ind_val.shape, dep_val.shape)

X = tf.keras.layers.Input(shape=[32, 32, 3])

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

pred = model.predict(ind_val[0:5])
print(pd.DataFrame(pred).round(2))
 
# 정답 확인
print(dep_val[0:5])
 
# 모델 확인
model.summary()