import matplotlib.pyplot as plt

# Scatter 띄우기

bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]
plt.scatter(x=bream_length, y=bream_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show

smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]
plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)

plt.scatter(30, 600, marker='^') # 하나의 점

# plt.xlabel('length') # 어차피
# plt.ylabel('weight') # 중복이라.
plt.show()

# 리스트 합치기 # 일단 사이킷런이 기대하는 형태를 위해서
length = bream_length+smelt_length
weight = smelt_weight+smelt_weight

fish_data = [(l, w) for l ,w in zip(length, weight)]

# 정답 준비
fish_target = [1]*30 + [0]*14 # 도미 1 빙어 0

# k-최근접 이웃 (간단하다고 한다.)
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
kn.fit(fish_data, fish_target)
kn.score(fish_data, fish_target)

kn.predict([[30,600]]) # 가까운 5(기본값)개의 데이터를 본다.

kn49 = KNeighborsClassifier(n_neighbor=46)
kn49.score(fish_data, fish_target)