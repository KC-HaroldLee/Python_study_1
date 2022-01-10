import tensorflow as tf
import pandas as pd

data_path = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/iris.csv'
iris = pd.read_csv(data_path)
print(iris.head())

# 원핫인코딩
iris = pd.get_dummies(iris)

ide_val = iris[['꽃잎길이', '꽃잎폭', '꽃받침길이', '꽃받침폭']]
dep_val = iris[['품종_setosa', '품종_versicolor', '품종_virginica']]
print(ide_val.shape, dep_val.shape)

X = tf.keras.layers.Input(shape=[4])
Y = tf.keras.layers.Dense(3, activation='softmax')(X)
model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy', metrics='accuracy')

model.fit(ide_val, dep_val, epochs=2500, verbose=0)
model.fit(ide_val, dep_val, epochs=10)

# 맨 처음 데이터 5개
print(model.predict(ide_val[:150]))
print(dep_val)

# 가중치
print(model.get_weights())