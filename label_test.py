# -*- coding: utf-8 -*-

#%%
import sys 
import os 
import os.path as osp 
import random

from tqdm  import tqdm 
import subprocess   # 파이썬에서 쉘 명령을 실행할 수 있게 해주는 라이브러리 
                    # os.system 보다 더 다양항 기능을 제공함
                    # (ref) http://www.incodom.kr/%ED%8C%8C%EC%9D%B4%EC%8D%AC/%EB%9D%BC%EC%9D%B4%EB%B8%8C%EB%9F%AC%EB%A6%AC/subprocess
import pandas as pd
from glob import glob
import csv 
import cv2
import numpy as np 
import natsort ## 숫자 정렬용 라이브러리


                    

cwd = os.getcwd()
dataset_path = osp.join(cwd, 'dataset')

trainVideo_path = osp.join(dataset_path, 'train')
valVideo_path = osp.join(dataset_path, 'val')

train_imgPath = osp.join(dataset_path, 'images', 'train')
val_imgPath = osp.join(dataset_path, 'images', 'val')

train_labelPath = osp.join(dataset_path, 'images', 'train_label.csv')
val_labelPath = osp.join(dataset_path, 'images', 'val_label.csv')

save_labelPath = osp.join(dataset_path, 'images', 'labels')




def get_yolo_bbox(width, height,  c_x, c_y, w_r, h_r):
    """ Convert yolo bbox format to opencv bbox format 
    (ref) https://stackoverflow.com/questions/64096953/how-to-convert-yolo-format-bounding-box-coordinates-into-opencv-format
    """
    x1 = int((c_x - w_r/2) * width)
    y1 = int((c_y - h_r/2) * height)

    x2 = int((c_x + w_r/2) * width)
    y2 = int((c_y + h_r/2) * height)


    p_leftEnd = x1, y1
    p_rightEnd = x2, y2

    return p_leftEnd, p_rightEnd


# %% Get item with random sample
img_list = os.listdir(osp.join(dataset_path, 'images','val'))
random_item = random.choice(img_list)  # acheive one image item in random; (ref) https://note.nkmk.me/en/python-random-choice-sample-choices/
print(f"Random image is: {random_item}")

item_name = random_item.split('.jpg')[0]


Path = osp.join(dataset_path, 'images')
img = cv2.imread(osp.join(Path, 'val' ,item_name+'.jpg'))

h, w = img.shape[:2]
print(w, h)

#label = pd.read_csv(osp.join(Path +'.txt'),) # (ref) https://m.blog.naver.com/kiddwannabe/221642949985




with open(osp.join(Path, 'labels' , item_name +'.txt')) as f: 
    labels = f.readlines()



#%% Get bboxes 
bboxes = []
for idx in range(len(labels)):
    values = labels[idx].split()
    bbox_value = [ float(x) for x in values]
    bboxes.append(bbox_value)


#%% Draw bboxes

for idx, bbox in enumerate(bboxes):

    up_end, down_end = get_yolo_bbox(w, h, *bbox[1:3], *bbox[3:])
    cv2.rectangle(img, up_end, down_end, (0,255,0), 1, 8)

#%%
cv2.imshow("test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
