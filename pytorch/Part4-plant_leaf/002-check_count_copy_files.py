import math
import os
import shutil

os.chdir('./Part4-plant_leaf')

base_dir = './splitted'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

original_dataset_dir = './dataset'
classes_list = os.listdir(original_dataset_dir)


for clss in classes_list : # 기준은 가장 오리지날 한곳
    path = os.path.join(original_dataset_dir, clss)
    file_list = os.listdir(path)
    
    files_count = len(file_list) # 각 폴더의 파일 개수를 의미한다
    train_size = math.floor(files_count * 0.6) # float를 int로 바꾸기 위해 floor를 쓴다.
    validation_size = math.floor(files_count * 0.2)
    test_size = math.floor(files_count * 0.2)

    print('{}\n파일 개수 : {}\n학습 개수 : {} | 검증 개수 : {} | 테스트 개수 : {}\n'
        .format(clss, files_count, train_size, validation_size, test_size))

    # train_dir로 복사    
    train_file_list = file_list[:train_size] # 이만큼만 할 것이다. 그런데 그러면 랜덤하지 않지 않나?
    for train_file in train_file_list :
        src = os.path.join(path, train_file) # 복사를 할 곳
        dst = os.path.join(os.path.join(train_dir, clss)) # 서로 같은 구조로 만들어주려고한다.
        # shutil.copyfile(src, dst) # 붙여넣기를 할 곳
        shutil.copy(src, dst) # copyfile은 권한 부족...!
    print('{}폴더 datasets -> train 이동 완료'.format(clss))    
    
    # validation_dir 로 복사
    validation_file_list = file_list[train_size:(validation_size + train_size)]
    for validation_file in validation_file_list :
        src = os.path.join(path, validation_file) # 복사를 할 곳
        dst = os.path.join(os.path.join(validation_dir, clss)) # 서로 같은 구조로 만들어주려고한다.
        shutil.copy(src, dst) # 붙여넣기를 할 곳
    print('{}폴더 datasets -> validation 이동 완료'.format(clss))    

    # test_dir로 복사
    test_file_list = file_list[train_size:(test_size + train_size)]
    for test_file in test_file_list :
        src = os.path.join(path, test_file) # 복사를 할 곳
        dst = os.path.join(os.path.join(test_dir, clss)) # 서로 같은 구조로 만들어주려고한다.
        shutil.copy(src, dst) # 붙여넣기를 할 곳
    print('{}폴더 datasets -> test 이동 완료'.format(clss))    

print('모든 작업 완료')
    






