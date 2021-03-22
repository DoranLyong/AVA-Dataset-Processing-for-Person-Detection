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



# ================================================================= #
#                           Function collection                     #
# ================================================================= #
""" 주어진 경로에 디렉토리가 없으면 새로 만들기 
"""
def createDirectory(dir):
    try:
        if not osp.exists(dir):
            os.makedirs(dir)
    except OSError:
        print('Error: Creating directory, ' + dir)


""" bbox 좌표를 opencv에 맞게 변환 
"""
def get_bbox(width, height, x1, y1, x2, y2):
    p_leftEnd = int(width * x1), int(height * y1)
    p_rightEnd = int(width * x2), int(height * y2)

    return p_leftEnd, p_rightEnd



# ================================================================= #
#                  bbox values in Yolo format                       #
# ================================================================= #

def make_yolo_bbox(width, height, x1, y1, x2, y2):
    """ cls_num center x, center y, w, h 
        (ex) Yolo Format : 0 0.256 0.315 0.506 0.593

        (ref) https://eehoeskrap.tistory.com/367
        (ref) https://towardsai.net/p/computer-vision/yolo-v5-object-detection-on-a-custom-dataset
    """
    x1, y1 = x1 / width, y1 / height
    x2, y2 = x2 / width, y2 / height
    w = (x2 - x1) 
    h = (y2 - y1) 
    center_x = x1 + w/2
    center_y = y1 +  h/2
    
    return center_x, center_y, w, h




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

img2 = img.copy()


for idx, instance in data_instances.iterrows():
    """ yolo bbox 좌표가 잘 만들어 졌는지 확인 
    """
    p0, p1 = get_bbox(width, height, *instance[1:5]) # 원본 -> opencv bbox 
    cx, cy, w, h = make_yolo_bbox(width, height, *p0, *p1) # yolo 형태 
    cvt_p0, cvt_p1 = get_yolo_bbox(width, height, cx, cy, w, h)  # yolo 형태 -> opencv bbox

    cv2.rectangle(img, p0, p1, (0, 255, 0), 1, 8)
    cv2.rectangle(img2, cvt_p0, cvt_p1, (0, 255, 0), 1, 8)


cv2.imshow("annotation", img)
cv2.imshow("from_yolo_format", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()





# ================================================================= #
#                    3. Save labels in txt file                    #
# ================================================================= #


#%%
#name = data_instances.iloc[0,5].replace('.jpg','' )  # 이미지 파일 이름만 가져오기 
#
#
#createDirectory(save_labelPath)
#
#
#bboxes = [] 
#for idx, instance in data_instances.iterrows():
#
#    bbox_list = [0, *instance.iloc[1:5]]
#    bboxes.append(bbox_list )
#
#
#print(pd.DataFrame(bboxes))  # (ref) https://stackoverflow.com/questions/32078737/create-pandas-dataframe-manually-without-columns-name
#                                  # (ref) https://datatofish.com/list-to-dataframe/
#
#label_ = pd.DataFrame(bboxes)
#label_.to_csv(osp.join(save_labelPath,f'{name}.txt'), sep=' ', header=False, index=False)  # (ref) https://mizykk.tistory.com/71
#                                                                                            # (ref) https://stackoverflow.com/questions/44917675/pandas-delete-column-name
#
#
#print(f"img: {img_path}")
#print(f"name.txt: {name}")



# ================================================================= #
#                            4. Show all                            #
# ================================================================= #
# %%

createDirectory(save_labelPath)   # label 파일을 저장할 디렉토리 생성 



loop = tqdm(enumerate(filenames), total=len(filenames)) 

for idx, img_name in loop:
    img_path = osp.join(data_path, img_name)
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    data_instances = label_list.loc[ (label_list[5]==img_name) , :   ]

#    print(data_instances)

    get_filename = data_instances.iloc[0,5].replace('.jpg','' )  # 이미지 파일 이름만 가져오기 

    bboxes = []

    for idx, instance in data_instances.iterrows():
        p0, p1 = get_bbox(w, h, *instance[1:5])   # 원본 -> opencv bbox 
        bbox_list = [0, *make_yolo_bbox(w, h, *p0, *p1)]  # yolo 형태 만들기 
        bboxes.append(bbox_list)

        p0, p1 = get_yolo_bbox(w, h, *bbox_list[1:])   # yolo 형태 -> opencv bbox
        cv2.rectangle(img, p0, p1, (0, 255, 0), 1, 8)


    label_ = pd.DataFrame(bboxes)
    label_.to_csv(osp.join(save_labelPath,f'{get_filename}.txt'), sep=' ', header=False, index=False)   # (ref) https://mizykk.tistory.com/71
                                                                                                        # (ref) https://stackoverflow.com/questions/44917675/pandas-delete-column-name


    cv2.imshow("annotation_test", img)
    
    k = cv2.waitKey(32) & 0xFF

    if k == 27:
        sys.exit()

cv2.destroyAllWindows()

# %%
