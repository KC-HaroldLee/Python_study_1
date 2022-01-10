import numpy as np

class Perceptron(object) :
    
    def __init__(self, eta=0.01, n_iter=50, random_state=1) :
        self.eta =  eta # 학습률
        self.n_iter = n_iter # 훈련 횟수
        self.random_state = random_state # 난수 생성기 시드

    def fit(self, X, y) :
        # X : 입력값 (대문자 2차원 배열)
        # y : 출력값 (소문자 1차원 배열)
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter) : # 그냥 반복하려고 썼구먼, while이런거 안쓰려고
            errors = 0
            for xi, target in zip(X, y): # 쌍을 맞춰서 반환해준다. (x1, y1) ....
                update = self.eta * (target - self.predict(xi)) # target = y, self.predict(xi) = y햇
                self.w_[1:] += update * xi # 가중치 업데이트
                self.w_[0] += update # 절편(얘는 마지막이 아니라 처음에 있는 것이 의미상으론 맞겠군)
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:], self.w_[0])

    def predict(self, X) :
        return np.where(self.net_input(X) >= 0.0, 1, -1)