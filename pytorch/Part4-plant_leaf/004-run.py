import torch
import os
# os.chdir('./Part4-plant_leaf/')

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

BATCH_SIZE = 256
EPOCH = 30

import torchvision.transforms as transforms
from torchvision import datasets


# 모델 학습을 위한 준비
transform_base = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    # transforms.Compose는 이미지의 전처리나 Agumentation에 사용되는 함수이다. 
    # Resize()는 크기를 64x64로 조정하고 
    # ToTensor()는 Tensor형태로 변환하고 모든 값을 0~1로 변환한다.

train_dataset = datasets.ImageFolder(root = './splitted/train', transform = transform_base)
val_dataset = datasets.ImageFolder(root = './splitted/val', transform = transform_base)
    # ImageFolder()는 데이터 셋을 불러오는 코드이다.
    # 하나의 클래스가 하나의 폴더에 대응된다.
    # root는 위치를 지정하고
    # transeform은 전처리 혹은 Agumentation을 지정한다.


from torch.utils.data import DataLoader
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE, 
                        shuffle = True, num_workers = 4)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = BATCH_SIZE, 
                        shuffle = True, num_workers = 4)


# 베이스라인 모델 설계
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module) :
    def __init__(self) :

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1) # (입력채널수, 출력채널수, 커널크기)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)

        self.fc1 = nn.Linear(4096, 512) # 왜 4096이지? 라는 생각이 들면 위를 보면 된다.
        self.fc2 = nn.Linear(512, 33)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = F.dropout(x, p = 0.25, training = self.training) # 여기서의 self는 nn.Module의 클래스 변수

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = F.dropout(x, p = 0.25, training = self.training)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)
        x = F.dropout(x, p = 0.25, training = self.training)

        x = x.view(-1, 4096)

        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training = self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

model_base = Net().to(DEVICE)
optimizer = optim.Adam(model_base.parameters(), lr=0.001)

# 모델 학습을 위한 함수
def train(model, train_loader, optimizer) :
    model.train() # 학습이라고 명시해준다.(학습모드)
    for batch_idx, (data, target) in enumerate(train_loader) : # 이번엔 idx띄우기 위해 enumerate를 사용한다.
        # print('batch_idx : {}/{}'.format(batch_idx, len(train_loader)))
        data = data.to(DEVICE) # device에 ...
        target = target.to(DEVICE) # ... 할당!
        optimizer.zero_grad() # optimizer 저장되어 있던 Batch, Gradient값을 초기화 한다.
        output = model(data) # 계산!
        loss = F.cross_entropy(output, target) # output과 target의 
        loss.backward()
        optimizer.step()


# 모델 평가를 위한 함수
def evaluate(model, test_loader) :
    model.eval() # 평가 모드!
    test_loss = 0
    correct = 0

    with torch.no_grad() :
        for data, target in test_loader :
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            output = model(data)

            test_loss += F.cross_entropy(output, target, reduction='sum').item() # outget, target 값차이를 계산한다.

            pred = output.max(1, keepdim = True)[1] # 모델에 입력된 Test데이터에서 각각의 확률값이 output이 출력된다.
            correct += pred.eq(target.view_as(pred)).sum().item() 
                # target.view_as(pred)를 통해 target의 Tensor구조를  pred의 Tensor구조로 만든다.
                # view_as : 모양에 맞춰 재정렬함
                # 그리고 나서 eq를 (일치하면1, 불일치면0)하고 계속 쌓는다.

    test_loss /= len(test_loader.dataset)  # 모든 미니 배치에서 합한 정확도 값을 batch수로 나눠서 평균을 구한다.
    # test_accuracy = 100. * correct / len(test_loader.datasets) # 정확도의 평균을 구한다.
    test_accuracy = 100. * correct / len(test_loader.dataset) #  s 빼라...
    return test_loss, test_accuracy

# 모델 학습 실행하기
import time
import copy

def train_baseline(model, train_loader, val_loader, optimizer, num_epoch = 30):
    print('train_baseline')
    best_acc = 0.0 # 가장 정확도가 높은놈으로 계속 갱신할 예정
    best_model_wts = copy.deepcopy(model.state_dict()) # 가장 정확도가 높은 놈을 저장할 변수?
   
    for epoch in range(1, num_epoch + 1) :
        since = time.time()
        train(model, train_loader, optimizer) # 학습시작
        train_loss, train_acc = evaluate(model, train_loader) # 정확도 1
        val_loss, val_acc = evaluate(model, val_loader) # 정확도 2

        if val_acc > best_acc : # 갱신하기
            # print('better acc')
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        
        time_elapsed = time.time() - since
        print('----------- epoch{} -----------'.format(epoch))
        print('train Loss: {:.4f}, Accuracy: {:.2f}%'.format(train_loss, train_acc))
        print('val Loss: {:.4f}, Accuracy: {:.2f}%'.format(val_loss, val_acc))
        print('Complete in {:.0f}m, {:.0f}s'.format(time_elapsed / 60, time_elapsed % 60))
    model.load_state_dict(best_model_wts)
    return model

# if __name__ == '__main__':

def run() :
    torch.multiprocessing.freeze_support()
    print('run')
    print('Device is \'{}\''.format(DEVICE))
    startTotalTime = time.time()
    base = train_baseline(model_base, train_loader, val_loader, optimizer, EPOCH)
    endTotalTimeEnd = time.time()
    print('총 걸린 걸린 시간 : ', endTotalTimeEnd - startTotalTime)
    torch.save(base, 'baseline.pt')
    print('저장완료')

if __name__ == '__main__':
    run()

