import tensorflow as tf
import pandas as pd
import time

(ind_val, dep_val), _ = tf.keras.datasets.mnist.load_data()
ind_val = ind_val.reshape(60000, 28, 28, 1)
dep_val = pd.get_dummies(dep_val)

# 나의 로드맵...
print(ind_val.shape, dep_val.shape) # (60000, 28, 28, 1) (60000, 10)

mkTimeStart = time.time()

# 모델 만들기
X = tf.keras.layers.Input(shape=[28, 28, 1])
# 컨볼루션
HC1 = tf.keras.layers.Conv2D(filters=3, kernel_size=5, activation='swish')(X)
HC2= tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='swish')(HC1)
# 플래튼
HF1 = tf.keras.layers.Flatten()(HC2)
# 덴스
HD1 = tf.keras.layers.Dense(units=84, activation='swish')(HF1)
Y = tf.keras.layers.Dense(units=10, activation='softmax')(HD1)

model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy', metrics='accuracy')
model.summary()
mkTimeEnd = time.time()

# # 최종 모델 만들기
# X = tf.keras.layers.Input(shape=[28, 28, 1])

# mkTimeStart = time.time()

# # 컨볼루션&풀링
# HC1 = tf.keras.layers.Conv2D(filters=3, kernel_size=5, activation='swish')(X)
# HP1 = tf.keras.layers.MaxPool2D()(HC1) # ??? 이게 끝?

# HC2= tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='swish')(HP1)
# HP2 = tf.keras.layers.MaxPool2D()(HC2)

# # 플래튼
# HF1 = tf.keras.layers.Flatten()(HC2)
# # 덴스
# HD1 = tf.keras.layers.Dense(units=84, activation='swish')(HF1)
# Y = tf.keras.layers.Dense(units=10, activation='softmax')(HD1)

# model = tf.keras.models.Model(X, Y)
# model.compile(loss='categorical_crossentropy', metrics='accuracy')

# model.summary()

# mkTimeEnd = time.time()


# 학습
fitTimeStart = time.time()
model.fit(ind_val, dep_val, epochs=10)
fitTimeEnd = time.time()
pred = model.predict(ind_val[0:5])
pd.DataFrame(pred).round(2)
 
# 정답 확인
print(dep_val[0:5])

print('====걸린시간====')
print('모델 생성 : ', mkTimeEnd - mkTimeStart)
print('학습 과정 : ', fitTimeEnd - fitTimeStart)
 