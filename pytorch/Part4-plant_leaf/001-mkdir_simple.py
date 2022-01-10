import os
import shutil
os.chdir('./Part4-plant_leaf')

original_dataset_dir = './dataset'
classes_list = os.listdir(original_dataset_dir) # <class 'list'>

base_dir = './splitted'
os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)

validation_dir = os.path.join(base_dir, 'val')
os.mkdir(validation_dir)

test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

for clss in classes_list:
    os.mkdir(os.path.join(train_dir, clss))
    os.mkdir(os.path.join(validation_dir, clss))
    os.mkdir(os.path.join(test_dir, clss))