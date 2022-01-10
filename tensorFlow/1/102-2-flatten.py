import tensorflow as tf
import pandas as pd

(ind_val, dep_val), _ = tf.keras.datasets.mnist.load_data()
print(ind_val.shape, dep_val.shape)

# ind_val = ind_val.reshape(60000, 784) # reshape는 하지 않는다.
dep_val = pd.get_dummies(dep_val)
print(ind_val.shape, dep_val.shape)

X = tf.keras.layers.Input(shape = [28, 28]) # 튜플이 아닌... 리스트(차원)
H1 = tf.keras.layers.Flatten()(X) # 이러면 알아서 됨
H2 = tf.keras.layers.Dense(units = 84, activation = 'swish')(H1) 
Y = tf.keras.layers.Dense(units = 10, activation = 'softmax')(H2)
model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy', metrics = 'accuracy')
# model.compile(loss='categorical_crossentropy', metrics='accuracy')

# 학습
model.fit(ind_val, dep_val, epochs = 10)

# 검증
pred = model.predict(ind_val[0:5])
print(pd.DataFrame(pred).round(2))
print(dep_val[0:5])