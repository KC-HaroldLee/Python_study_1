import tensorflow as tf
import pandas as pd

data_url = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/boston.csv'
boston = pd.read_csv(data_url)

ide_val = boston[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 
        'rad', 'tax', 'ptratio', 'b', 'lstat']]
dep_val = boston[['medv']]

print(ide_val.shape, dep_val.shape) # (506, 13) (506, 1)

X = tf.keras.layers.Input(shape=[13])
H = tf.keras.layers.Dense(10, activation='swish')(X) #
Y = tf.keras.layers.Dense(1)(H) #
model = tf.keras.models.Model(X, Y) 
model.compile(loss='mse')

# 모델 구조 확인
model.summary()

# 학습(FIT)
model.fit(ide_val, dep_val, epochs=1000)

print(model.predict(ide_val[:5]))
print(dep_val[:5])

print(model.get_weights())