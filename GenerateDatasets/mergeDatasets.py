import cv2
import numpy as np
import os
import shutil

"""
merge datasets, as phoenix and shanghai datasets consisted of multiple folders containing images from different days
additionally, change the imagery name to the same format for every dataset YYYYMMDD_hhmm.jpg
"""

root_path = "../Datasets/ProcessedImages/"
dst_path = "../Datasets/FinalImages/"
datasets = os.listdir(root_path)

datasets_name = ['Shanghai', 'Beijing', 'Phoenix']

for dataset_name in datasets_name:
    datasets_array = np.array(datasets)
    datasets_to_merge = datasets_array[np.char.find(datasets_array, dataset_name) != -1]

    if not os.path.exists(dst_path+dataset_name):
        os.mkdir(dst_path+dataset_name)
    for dataset_to_merge in datasets_to_merge:
        files = os.listdir(root_path+dataset_to_merge)
        if dataset_name == 'Beijing':
            for file in files:
                file0 = file.replace("_", '')
                file0 = file0[:8] + "_" + file0[8:]
                shutil.copyfile(root_path+dataset_to_merge+"/"+file, dst_path+dataset_name+"/"+file0)
        elif dataset_name == 'Shanghai':
            for file in files:
                file0 = file[:8] + "_" + file[8:]
                shutil.copyfile(root_path+dataset_to_merge+"/"+file, dst_path+dataset_name+"/"+file0)
        elif dataset_name == 'Phoenix':
            for file in files:
                shutil.copyfile(root_path+dataset_to_merge+"/"+file, dst_path+dataset_name+"/"+file)
