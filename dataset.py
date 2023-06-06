import shutil
import os
import get_folder_and_file
import random


datapath = r'C:\Users\guote\PycharmProjects\AllaertCNN\CASME2'
dataset_path = r'./data'

def anomalies_check(path):
    list = get_folder_and_file.get_filelist(path,[])
    for i in list:
        if not (i.endswith('.jpg')):
            print(i)
            os.remove(i)
        else:
            continue

def random_testset_selection(path):
    testset = []
    rootexclusion = []
    
    for folder, _, _ in os.walk(path):
        new = folder.replace(path,'')
        if new == '':
            continue
        else:
            rootexclusion.append(path + new)
    set_ = []
    categories = []
    
    """extract classes"""
    for classid in range(len(rootexclusion)):
        categories.append(rootexclusion[classid].replace(path, ''))
        categories[classid] = os.path.split(categories[classid])[1]

    """get random test set"""
    for cur_folder in rootexclusion:
        files = get_folder_and_file.get_filelist(cur_folder, [])
        testnum = int(len(files) * 0.2)
        random.shuffle(files)
        set_.append(random.sample(files,testnum))
    
    for i in set_:
        for j in i:
            j = os.path.normpath(os.path.abspath(j))
            testset.append(j)
        testset.sort()
    return testset, categories

def train_testset_move(testset,categories,targetfolder):
   
    """Create directories by classes"""
    for cats in categories:
        try:
            os.makedirs(os.path.join(targetfolder, 'train', cats), exist_ok=True)
            os.makedirs(os.path.join(targetfolder, 'test', cats), exist_ok=True)
        except OSError as error:
            print("Directory '%s' can not be created")
            
    """Move test files into corresponding directories"""
    for test_sample in testset:
        sample_category = os.path.split(os.path.dirname(test_sample).replace(os.path.dirname(os.path.dirname(test_sample)), ''))[1]
        test_target_directory = os.path.normpath(os.path.join(targetfolder, 'test', sample_category))
        shutil.move(test_sample, test_target_directory)
        print("Moving test files to {}".format(test_target_directory))
    # print("All test samples have been moved to {}".format(os.path.dirname(test_target_directory)))
    
    """Move the rest(train) files into corresponding directories"""
    train_list = get_folder_and_file.get_filelist(os.path.dirname(os.path.dirname(testset[0])), [])
    for train_sample in train_list:
        sample_category = os.path.split(os.path.dirname(train_sample).replace(os.path.dirname(os.path.dirname(train_sample)), ''))[1]
        train_target_directory = os.path.normpath(os.path.join(targetfolder, 'train', sample_category))
        shutil.move(train_sample, train_target_directory)
        print("Moving train files to {}".format(train_target_directory))
    # print("All test samples have been moved to {}".format(os.path.dirname(train_target_directory)))
    
test_set, categories_ = random_testset_selection('./categorized')
train_testset_move(test_set, categories_, dataset_path)






















