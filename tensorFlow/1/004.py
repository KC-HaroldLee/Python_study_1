import pandas as pd
import tensorflow as tf

# 데이터 수집
data_url = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/boston.csv'
boston = pd.read_csv(data_url)
# print(boston.head())
columns_length = len(boston.columns)
print(len(boston.columns))

ind_val = boston[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 
            'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']] # medv는 뺌
dep_val = boston[['medv']]
print(ind_val.shape, dep_val.shape)

# 모델의 구조를 만든다.
X = tf.keras.layers.Input(shape=[13]) # 여긴 shape고
Y = tf.keras.layers.Dense(units=1)(X) # 여긴 units네
model = tf.keras.models.Model(X, Y)
model.compile(loss='mse')

# 학습한다.
model.fit(ind_val, dep_val, epochs=10)
model.fit(ind_val, dep_val, epochs=4980, verbose=0) # vebose는 로그가 나오게
model.fit(ind_val, dep_val, epochs=10)

# 모델을 이용
print(model.predict(ind_val[55:105]))
# 종속변수 확인
print(dep_val[55:105])

# 모델의 수식 확인
print(model.get_weights())