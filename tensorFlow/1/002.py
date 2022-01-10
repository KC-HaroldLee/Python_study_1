import pandas as pd
import tensorflow as tf

data_url = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/lemonade.csv'
data_set = pd.read_csv(data_url)
print(data_set.head())

# 데이터 나누기
ind_val = data_set[['온도']]
dep_val = data_set[['판매량']]

# 모델 만들기
X = tf.keras.layers.Input(shape=[1]) # 1은 칼럼 개수
Y = tf.keras.layers.Dense(1)(X)
model = tf.keras.models.Model(X, Y)
model.compile(loss='mse') # mse?

# 학습시키기
model.fit(ind_val, dep_val, epochs = 10000)

# 
print(model.predict(ind_val))
print(data_set )
print(model.predict([[15]]))
