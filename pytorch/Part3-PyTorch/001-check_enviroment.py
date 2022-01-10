import torch
import torch.nn as nn # 딥러닝 네크워의의 기본 구성 요소를 포함한 모듈
import torch.nn.functional as F # 딥러닝에서 자주 사용되는 함수가 포함한 모듈 
import torch.optim as optim # 가중치 추정에 필요한 최적화 알고리즘을 포함한 모듈
from torchvision import datasets, transforms # 딥러닝에서 자주 사용되는 데이터셋과 모델 구조, 이미지 변환기술을 포함한 모듈
from matplotlib import pyplot as plt # 시각화!

import os
os.chdir('./Part3-PyTorch')
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# 분석 환경 지정

is_cuda = torch.cuda.is_available()
# device = torch.device('cuda' if is_cuda else 'cpu')
device = torch.device('cpu')

print ('현재 device : {}.'.format(device))

# HyperParameter 지정

# 모델을 설계하기 전에 HyperParameter를 사전에 정의한다.
# batch_size : 모델 가중치를 한 번 업데이트 시킬 때 사용되는 샘플 단위 개수(미니 배치 사이즈)
# epoch_num : 학습 데이터를 모두 사용하여 학습하는 기본 단위 획수(Epoch 수)
# learning : 가중치의 업데이트의 정도

batch_size = 50 
epoch_num = 15
learning_rate = 0.001

# MNIST 데이터 불러오기
# root = MNIST 데이터를 저장할 물리적 공간 위치, 
# train = 학습용으로 사용할 것인지
# download = True일 경우 root에 지정된 것에 데이터가 저장된다. 현재는 yann.lecun.com에서 데이터가 다운로드 된다.
# transform = MNIST 데이터를 저장과 동시에 '전처리'할 수 있는 옵션 - PyTorch는 입력데이터로 Tensor를 사용한다.

train_data = datasets.MNIST(root = './data', train = True, download = True, transform = transforms.ToTensor())
test_data = datasets.MNIST(root = './data', train = False,                  transform = transforms.ToTensor())

print('트레이닝 데이터 개수 : {}'.format(len(train_data)))
print('테스트 데이터 개수 : {}'.format(len(test_data)))

image, label = train_data[0]
print(type(image)) # <class 'torch.Tensor'>
print(type(label)) # <class 'int'>

# 차트 보기
# plt.imshow(image.squeeze().numpy(), cmap = 'gray')
# plt.imshow(image.squeeze().numpy(), cmap = 'gray')
# plt.title('label : {}'.format(label))
# plt.show()

# 미니배치 구성하기
# dataset : 미니배치로 구성할 데이터
# batch_size : 미니 배치의 사이즈 (현재 50)
# shuffle : 랜덤할지 안할지 - 대부분 필수!

train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size = batch_size, shuffle = True)

first_batch = train_loader.__iter__().__next__() 
# print(len(first_batch)) # 2
print('{:15} | {:<25s} | {}'.format('name', 'type', 'size'))
print('{:15} | {:<25s} | {}'.format('Num of Batch', '', len(train_loader)))
print('{:15} | {:<25s} | {}'.format('first_batch', str(type(first_batch)), len(first_batch)))
print('{:15} | {:<25s} | {}'.format('first_batch[0]', str(type(first_batch[0])), first_batch[0].shape)) 
print('{:15} | {:<25s} | {}'.format('first_batch[1]', str(type(first_batch[1])), first_batch[1].shape))

# [0]에서 볼 수 있는 [50,1,28,28]에서 추가된(듯한) 50은 Batch_size를 의미한다. 이전에 설정한 50이다. 따라서 Num of Batch가 1200 (60000 / 50)이 되는 것이다.
# [1]에서 볼 수 있는 [50]은 50크기의 벡터를 의미한다. 미니배치의 정답이 저장되어 있다.


# CNN 구조 만들기
class CNN(nn.Module) :
    def __init__(self) :
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 18)
    
    def forward(self, x) : # 원래는... 한줄로 적는다 그건가...
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


# Optimizer 및 손실 함수 정의
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr = learning_rate) # learning_rate = 0.001
criterion = nn.CrossEntropyLoss()

# 설계한 모형 확인하기
print('model 모형 확인 :')
print(model)

# 모델 학습
model.train()

import time
start = time.time()

correct = 0
for data, target in test_loader :
    data = data.to(device)
    target = target.to(device)
    output = model(data)
    prediction = output.data.max(1)[1]
    correct += prediction.eq(target.data).sum()
print('Test set: 최초 정확도: {:.2f}%'.format(100 * correct / len(test_loader.dataset)))

i = 0
for epoch in range(epoch_num) :
    for data, target in train_loader :
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if i % 1000 == 0 : # 1000마다
            print('Train Step: {}\tLoss : {:3f}'.format(i, loss.item()))        
        i += 1

    # 중간 정확도 검사 - 속도 느려짐 
    # correct = 0
    # for data, target in test_loader :
    #     data = data.to(device)
    #     target = target.to(device)
    #     output = model(data)
    #     prediction = output.data.max(1)[1]
    #     correct += prediction.eq(target.data).sum()
    # print('Test set: 중간 정확도: {:.2f}%'.format(100 * correct / len(test_loader.dataset)))
    
    
print('학습 완료, 걸린 시간 : {}'.format(time.time() - start))

# 모델 평가 하기

start = time.time()

model.eval()
correct = 0
for data, target in test_loader :
    data = data.to(device)
    target = target.to(device)
    output = model(data)
    prediction = output.data.max(1)[1]
    correct += prediction.eq(target.data).sum()

print('평가 완료, 걸린 시간 : {}'.format(time.time() - start))

print('Test set: 최종 정확도: {:.2f}%'.format(100 * correct / len(test_loader.dataset)))