import tensorflow as tf
import pandas as pd
from tensorflow.python.eager.monitoring import Metric
from tensorflow.python.ops.gen_array_ops import shape

# 데이터 준비
(ind_val, dep_val), _ = tf.keras.datasets.mnist.load_data()
print(ind_val, dep_val)

ind_val = ind_val.reshape(60000, 28, 28, 1) # 1을 추가하기 위한
dep_val = pd.get_dummies(dep_val)
print(ind_val.shape, dep_val.shape) # 이걸 로드맵 삼으면 되겠구만.

X = tf.keras.layers.Input(shape = [28, 28, 1]) # 역시 추가
H1 = tf.keras.layers.Conv2D(3, kernel_size = 5, activation='swish')(X)
H2 = tf.keras.layers.Conv2D(6, kernel_size = 5, activation='swish')(H1)
H3 = tf.keras.layers.Flatten()(H2) # flatten도 Hidden

H4 = tf.keras.layers.Dense(84, activation = 'swish')(H3)
# Y = tf.keras.layers.Dense(10, activation = 'swish')(H4) # 여기가 로드맵의 마지막이다.
Y = tf.keras.layers.Dense(10, activation = 'softmax')(H4)

model = tf.keras.models.Model(X, Y)
model.compile(loss = 'categorical_crossentropy', metrics = 'accuracy')

model.fit(ind_val, dep_val, epochs=10)

pred = model.predict(ind_val[0:5])
print(pd.DataFrame(pred).round(2))

print(dep_val[0:5])

print(model.summary())