import os
import shutil
os.chdir('./Part4-plant_leaf')

original_dataset_dir = './dataset'
classes_list = os.listdir(original_dataset_dir) # <class 'list'>

try : 
    base_dir = './splitted'
    os.mkdir(base_dir)
except FileExistsError : 
    print('이미 만들어져 있는 폴더 :{}'.format(base_dir))

try : 
    train_dir = os.path.join(base_dir, 'train')
    os.mkdir(train_dir)
except FileExistsError : 
    print('이미 만들어져 있는 폴더 :{}'.format(train_dir))

try :
    validation_dir = os.path.join(base_dir, 'val')
    os.mkdir(validation_dir)
except FileExistsError : 
    print('이미 만들어져 있는 폴더 :{}'.format(validation_dir))

try :
    test_dir = os.path.join(base_dir, 'test')
    os.mkdir(test_dir)
except FileExistsError : 
    print('이미 만들어져 있는 폴더 :{}'.format(test_dir))

for clss in classes_list:
    try : 
        os.mkdir(os.path.join(train_dir, clss))
    except FileExistsError : 
        print('이미 만들어져 있는 폴더 :{}'.format(clss))

    try : 
        os.mkdir(os.path.join(validation_dir, clss))
    except FileExistsError : 
        print('이미 만들어져 있는 폴더 :{}'.format(clss))

    try :
        os.mkdir(os.path.join(test_dir, clss))
    except FileExistsError :
        print('이미 만들어져 있는 폴더 :{}'.format(clss))