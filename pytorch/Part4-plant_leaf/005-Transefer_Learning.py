
from typing_extensions import Required
import torch
import os
from torch import optim
from torchvision import transforms, datasets
from torchvision.models.resnet import resnet50
import copy
import time

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
BATCH_SIZE = 256
EPOCH = 30

################################
### Transfer Learning을 위한 준비
################################
data_transforms = {
    'train' : transforms.Compose([      # 이미지 데이터의 전처리 Augmentation등의 과정에서 사용되는 메소드
        transforms.Resize([64, 64]),    # 리사이즈
        transforms.RandomHorizontalFlip(),  # 랜!
        transforms.RandomVerticalFlip(),    # 덤!
        transforms.RandomCrop(52),          # 크기도 랜덤!
        transforms.ToTensor(),              # 텐서로 변환(0~1)
        transforms.Normalize([0.485, 0.456, 0.406],     # 정규화를 적용할 평균값
                             [0.229, 0.224, 0.225])     # 정규화를 적용할 표준편차
    ]),
    'val' : transforms.Compose([        # 검증, 이하 같음
        transforms.Resize([64, 64]),
        transforms.RandomCrop(52),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

data_dir = './splitted'     # 학습 데이터를 불러올 폴더 경로
image_datasets = {x : datasets.ImageFolder(root = os.path.join(data_dir, x),    # ImageFolder : 데이터 셋을 불러오는 메서드
                transform = data_transforms[x]) for x in ['train', 'val']}      # transform는 전처리 # len은 2!

#dataloaders = {x : torch.utils.data.DataLoader(image_datasets[x],     # DataLoader: 불러온 데이터들을 주어진 조건에 따라 미니 배치 단위로 분리한다.
#                batch_size=BATCH_SIZE, shuffle=True, num_workers=4) for x in ['train', 'val']}   # dict가 좋다(?) # len은 2!

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, 
                shuffle=True, num_workers=4) for x in ['train', 'val']} 

dataset_sizes = {x : len(image_datasets[x]) for x in ['train', 'val']}  # 학습/검증 데이터 개수를 저장한다. # 역시 dict # len은 2!

class_names = image_datasets['train'].classes   # 나중에 활용하기 위해 33개가 되는 클래스 이름 목록을 저장한다. 

################################
### Pre-Trained Model 불러오기
################################

from torchvision import models
import torch.nn as nn

resnet = models.resnet50(pretrained=True)   # 다양한 모델 중에 일단 이거 선택
                                            # pretrained가 False면 모델의 구조만을 가져온다.

num_ftrs = resnet.fc.in_features    # 출력 채널 개수를 반환한다. # 기본 2048
# resnet.fc = nn.Linear(num_ftrs, 33)   # 하드 코딩 말고 아래 걸로
resnet.fc = nn.Linear(num_ftrs, len(class_names))   
    # 마지막 Fully Connected Layer 대신
    # 출력개수가 33인 새로운 Layer를 추가한다.(교체한다.)
        # 변환 전 : Linear(in_features=2048, out_features=1000, bias=True)
        # 변환 후 : Linear(in_features=2048, out_features=33, bias=True)
resnet = resnet.to(DEVICE)

criterion = nn.CrossEntropyLoss()   # 모델에 학습할 떄 사용하는 Loss함수를 지정한다.
                                    # 베이스 라인과 동일하게 CrossEntropyLoss
                                    
optimizer_ft = optim.Adam(filter(lambda p : p.requires_grad, resnet.parameters()), lr=0.001)
# Optimizer는 Adam으로 설정하고 Learing Rate는 0.001로 설정했다.
# 이전에는 모든 Parameter를 업데이트 했지만 
# 이번에는 일부 Layer의 Parrameter만 업데이트 해야한다.
# Requires_grad가 True인 Layer에만 Parameter가 적용된다.

from torch.optim import lr_scheduler

exp_lr_scheduler = lr_scheduler.StepLR(optimizer=optimizer_ft, step_size=7, gamma=0.1)
# StepLR은 Epoch에 따라 LeaningRate를 바꿔주는 역할을 한다. 
# optimizer를 설정하고 7Epoch마다 0.1곱하여 Learning Rate를 감소시킨다.

################################
### Pre-Trained Model의 일부 Layer Freeze하기
################################

ct = 0
for child in resnet.children() : # children()은 resnet의 모든 레이어 정보를 가지고 있다.
    ct += 1
    # print('ct :',ct)
    # print('--------')
    if ct < 6 :
        # ct2 = 0
        for param in child.parameters() :
            param.requires_grad = False # 파라미터가 업데이트 되지 않도록 고정한다는 의미
            # ct2 += 1
            # print('\r | ct2 :',ct2 ,end='')





def train_resnet(model, criterion, optimizer, scheduler, num_epochs=25) :

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs) :
        print('-------------------epoch {}-------------------'.format(epoch+1))
        since = time.time()

        for phase in ['train', 'val'] :
            if phase == 'train' :
                model.train()
            if phase == 'val' :
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase] :
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train') :
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train' :
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase== 'train' :
                scheduler.step()
                l_r = [x['lr'] for x in optimizer_ft.param_groups]
                print('learning rate: ', l_r)
            
            epoch_loss = running_loss/dataset_sizes[phase]
            epoch_acc = running_corrects/dataset_sizes[phase]

            print('{} Loss : {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))


            if phase == 'val' and epoch_acc > best_acc :
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('Complete in {:.0f}m {:0f}s'.format(time_elapsed//60, time_elapsed%60))
    print('Best val acc : {:.4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)



# model_resnet50 = train_resnet(resnet, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=EPOCH)

# torch.save(model_resnet50, 'resnet50.pt')

def run() :
    torch.multiprocessing.freeze_support()
    print('run')
    print('Device is \'{}\''.format(DEVICE))
    # startTotalTime = time.time()
    model_resnet50 = train_resnet(resnet, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=EPOCH) 
    torch.save(model_resnet50, 'resnet50.pt')
    print('저장완료')

if __name__ == '__main__':
    run()
