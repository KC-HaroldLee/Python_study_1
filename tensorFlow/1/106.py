import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

paths = glob.glob('./notMNIST_small/*/*.png')
# paths = glob.glob('tensorFlow/notMNIST_small/*/*.png')
paths = np.random.permutation(paths) 
print(paths[0])
    # tensorFlow/notMNIST_small\E\Q2FsdmVydCBNVCBMaWdodC50dGY=.png
ind_val = np.array([plt.imread(paths[i]) for i in range(len(paths))])
dep_val = np.array([paths[i].split('\\')[1] for i in range(len(paths))])

print(ind_val.shape, dep_val.shape)
    # (18724, 28, 28) (18724,)

# 변수 전처리
ind_val = ind_val.reshape(18724, 28, 28, 1)
# 종속 = pd.get_dummies(종속)
dep_val = pd.get_dummies(dep_val) # 다행히 1차원
# dep_val = dep_val.reshape(18724, 10)
print(ind_val.shape, dep_val.shape)
    # 

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

pred = model.predict(ind_val[0:5])
print(pd.DataFrame(pred).round(2))
 
# 정답 확인
print(dep_val[0:5])
 
# 모델 확인
model.summary()