import cv2
import numpy as np
import os

root_path = "../Datasets/Images/"
dst_path = "../Datasets/ProcessedImages/"
datasets = os.listdir(root_path)
bbox_dict = {}
for dataset in datasets:
    images = os.listdir(root_path+dataset)
    img_array = cv2.imread(root_path+dataset+"/"+images[0])
    img_h = img_array.shape[0]
    img_w = img_array.shape[1]

    r = cv2.selectROI("select the area", img_array)
    bbox_dict[(img_h,img_w)] = r

    for image in images:
        img_array = cv2.imread(root_path + dataset + "/" + image)
        img_h = img_array.shape[0]
        img_w = img_array.shape[1]
        #in case it finds an image with different size it is necessary to selectROI again
        if (img_h,img_w) not in bbox_dict:
            r = cv2.selectROI("select the area", img_array)
            bbox_dict[(img_h, img_w)] = r

        r = bbox_dict[(img_h,img_w)]
        # Crop image
        cropped_image = img_array[int(r[1]):int(r[1] + r[3]),
                        int(r[0]):int(r[0] + r[2])]
        # Display cropped image
        #cv2.imshow("Cropped image", cropped_image)
        #cv2.waitKey(0)
        if not os.path.exists(dst_path+dataset):
            os.mkdir(dst_path+dataset)
        cv2.imwrite(dst_path+dataset+"/"+image, cropped_image)


