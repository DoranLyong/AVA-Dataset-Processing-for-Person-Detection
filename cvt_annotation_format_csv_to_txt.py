# -*- coding: utf-8 -*-

#%%
import sys 
import os 
import os.path as osp 
import random
import csv
from glob import glob

from tqdm  import tqdm 
import subprocess   # 파이썬에서 쉘 명령을 실행할 수 있게 해주는 라이브러리 
                    # os.system 보다 더 다양항 기능을 제공함
                    # (ref) http://www.incodom.kr/%ED%8C%8C%EC%9D%B4%EC%8D%AC/%EB%9D%BC%EC%9D%B4%EB%B8%8C%EB%9F%AC%EB%A6%AC/subprocess
import pandas as pd
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



# ================================================================= #
#                           Function collection                     #
# ================================================================= #

def get_bbox(width, height, x1, y1, x2, y2):
    p_leftEnd = int(width * x1), int(height * y1)
    p_rightEnd = int(width * x2), int(height * y2)

    return p_leftEnd, p_rightEnd




# ================================================================= #
#                          1. Load image list                       #
# ================================================================= #
# %%
data_path = val_imgPath
filenames = os.listdir(data_path)
filenames = natsort.natsorted(filenames) # string 자료형 소팅; (ref) https://mentha2.tistory.com/171



""" Random selection:
    이미지 파일 목록에서 랜덤으로 하나 고르기 
    (ref) https://stackoverflow.com/questions/306400/how-to-randomly-select-an-item-from-a-list
"""
selected_name = random.choice(filenames)
img_path = osp.join(data_path, selected_name)

print(img_path)
img = cv2.imread(img_path)
cv2.imshow("image", img)


height, width = img.shape[:2]



# ================================================================= #
#                    2. Load annotation CSV file                    #
# ================================================================= #
# %%
label_list = pd.read_csv(val_labelPath, header=None)



""" 불러온 이미지 파일이 CSV 목록에 있는지 확인하고, 
    있다면 bbox 좌표를 불러와서 표현하고, 없다면 skip. 
"""
data_instances = label_list.loc[ (label_list[5]==selected_name) , :   ]
print(data_instances)



for idx, instance in data_instances.iterrows():
    p0, p1 = get_bbox(width, height, *instance[1:5])
    cv2.rectangle(img, p0, p1, (0, 255, 0), 1, 8)

cv2.imshow("annotation", img)
cv2.waitKey(0)
cv2.destroyAllWindows()



# ================================================================= #
#                            3. Show all                            #
# ================================================================= #
# %%

loop = tqdm(enumerate(filenames), total=len(filenames)) 

for idx, img_name in loop:
    img_path = osp.join(data_path, img_name)
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    data_instances = label_list.loc[ (label_list[5]==img_name) , :   ]

    print(data_instances)

    for idx, instance in data_instances.iterrows():
        p0, p1 = get_bbox(w, h, *instance[1:5])
        cv2.rectangle(img, p0, p1, (0, 255, 0), 1, 8)

    cv2.imshow("annotation_test", img)
    
    k = cv2.waitKey(500) & 0xFF

    if k == 27:
        sys.exit()

    cv2.destroyAllWindows()

# %%
