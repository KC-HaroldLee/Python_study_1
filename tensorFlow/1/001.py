import pandas as pd

# 파일로부터 데이터 읽어오기
file_url = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/lemonade.csv'
lemonade = pd.read_csv(file_url)

file_url= 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/boston.csv'
boston = pd.read_csv(file_url)

file_url = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/iris.csv'
iris = pd.read_csv(file_url)

# shape 확인
print(lemonade.shape)
print(boston.shape)
print(iris.shape)

# 칼럼확인
print(lemonade.columns)
print(boston.columns)
print(iris.columns)

# 독립변수와 종속변수 분리

val1 = lemonade[['온도']] # 왜 두개일까
val2 = lemonade[['판매량']]

print(val1)
print(val2)

# 데이터 확인?
print(lemonade.head()) # 5개만 나옴
print(iris) # 끝까지는 나옴 중간 ...처리
