# -*- coding: utf-8 -*-

#%%
import sys 
import os 
import os.path as osp 
import csv  # (ref) https://docs.python.org/ko/3/library/csv.html
from glob import glob # (ref)https://docs.python.org/ko/3/library/glob.html

from tqdm  import tqdm 
import subprocess   # 파이썬에서 쉘 명령을 실행할 수 있게 해주는 라이브러리 
                    # os.system 보다 더 다양항 기능을 제공함
                    # (ref) http://www.incodom.kr/%ED%8C%8C%EC%9D%B4%EC%8D%AC/%EB%9D%BC%EC%9D%B4%EB%B8%8C%EB%9F%AC%EB%A6%AC/subprocess
import youtube_dl   # 유튜브 영상 다운로드 패키지; (ref) https://velog.io/@okstring/%ED%8C%8C%EC%9D%B4%EC%8D%AC%EC%9C%BC%EB%A1%9C-%EC%9C%A0%ED%8A%9C%EB%B8%8C-%EB%8F%99%EC%98%81%EC%83%81-%EB%8B%A4%EC%9A%B4%EB%B0%9B%EA%B8%B0youtubedl
import pandas as pd
import cv2
import numpy as np 
                    



cwd = os.getcwd()
dataset_path = osp.join(cwd, 'dataset')

trainVideo_path = osp.join(dataset_path, 'train')
valVideo_path = osp.join(dataset_path, 'val')

train_imgPath = osp.join(dataset_path, 'images', 'train')
val_imgPath = osp.join(dataset_path, 'images', 'val')

train_labelPath = osp.join(dataset_path, 'images', 'train_label.csv')
val_labelPath = osp.join(dataset_path, 'images', 'val_label.csv')

index_image = 191430




# ================================================================= #
#                       1. Create a directory                       #
# ================================================================= #
#%% 
""" 주어진 경로에 디렉토리가 없으면 새로 만들기 
"""
def createDirectory(dir):
    try:
        if not osp.exists(dir):
            os.makedirs(dir)
    except OSError:
        print('Error: Creating directory, ' + dir)



# ================================================================= #
#                         2. Get video files                        #
# ================================================================= #
#%%
def search_video(dirPath, video_name):
    filenames = os.listdir(osp.join(dirPath, video_name))   # 해당 경로 디렉토리의 파일 목록 리스트로 반환; 
                                                            # (ref) https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
    checking = False
    video_path = "unknown"

    for filename in filenames:
        pre, org = filename.split('__v__')

        if pre == video_name:  
            checking = True
            video_path = osp.join(dirPath, video_name ,filename)
    return checking, video_path


# ================================================================= #
#                      3. Get bbox coordinates                      #
# ================================================================= #

def get_bbox(width, height, x1, y1, x2, y2):
    p_leftEnd = int(width * x1), int(height * y1)
    p_rightEnd = int(width * x2), int(height * y2)

    return p_leftEnd, p_rightEnd


def vis_bbox(img, left_end, right_end):

    cv2.rectangle(img, left_end, right_end, (0, 255, 0), 1, 8)
    cv2.imshow("videoFrame", img)

    k = cv2.waitKey(32) & 0xFF

    if k == 27:
        sys.exit()




def loop_getImageFrame(data_list, videos_path, img_path, unique_Names, label_path):


    createDirectory(img_path)
    
    csv_file = open(label_path, 'w', encoding='utf-8', newline='')  # (ref) https://devpouch.tistory.com/55
    csv_wr = csv.writer(csv_file)


    loop = tqdm(enumerate(unique_Names), total=len(unique_Names)) 

    for idx, get_filename in loop:
        exist, video_path = search_video(videos_path, get_filename)

        if exist:  # 비디오 파일이 있으면 
            video_clip = cv2.VideoCapture(video_path)  # 동영상 파일 불러오기 (ref) https://copycoding.tistory.com/154
            height = video_clip.get(cv2.CAP_PROP_FRAME_HEIGHT) # 캡쳐한 영상의 속성을 리턴 
            width = video_clip.get(cv2.CAP_PROP_FRAME_WIDTH)

            data_instances = data_list.loc[(data_list[0] == get_filename ), : ]     # 파일 이름이 동일한 인스턴스만 가져옴.
                                                                                    # pandas Dataframe 의 조건부 인덱싱; (ref) https://www.analyticsvidhya.com/blog/2020/02/loc-iloc-pandas/

            timestamp_list = pd.unique(data_instances.loc[:,1]) # 사용할 비디오 프레임의 timestamp


            for timestamp in timestamp_list:
                scoped_instances  = data_instances.loc[((data_instances[1]==timestamp))]
                scoped_instances = scoped_instances.drop_duplicates([2,3,4,5])  # bbox 중복 제거 (ref) https://pydole.tistory.com/entry/Python-pandas-%EC%A4%91%EB%B3%B5%EA%B0%92-%EC%B2%98%EB%A6%AC-duplicates-dropduplicates

                for idx, instance in scoped_instances.iterrows(): # csv 인스턴스 순회; (ref) https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas


                    video_clip.set(cv2.CAP_PROP_POS_MSEC, instance[1]*1000) # 해당 timestamp 의 이미지 프레임 가져오기 
                    ret, frame = video_clip.read()

                    

                    img_title = f"{instance[0]}_{instance[1]}.jpg"  # 추출한 이미지 이름; {name}_{timestamp}.jpg

                    if not osp.exists(osp.join(img_path, img_title)):  # 동일한 timestamp의 이미지는 딱 한 번만 저장하기 
                        save_ok = save_frame(img_path , img_title , frame)

                        if not save_ok:
                            """ 이미지가 저장이 안 됐으면 뭔가 잘못된 것 
                            """
                            print("Saving fail...")
                            sys.exit()

                    p0, p1 = list(np.around(np.array(instance[2:4], dtype=np.float),3)), list(np.around(np.array(instance[4:6], dtype=np.float),3)) # yolo_v5 학습용 포멧 
                                                                                                                                                    # 리스트 라운드 처리; (ref) https://stackoverflow.com/questions/5326112/how-to-round-each-item-in-a-list-of-floats-to-2-decimal-places/49472587
                                                                                                                                                    # float 에러 처리; (ref)https://forums.fast.ai/t/float-object-has-no-attribute-rint-while-calling-fit/47008

                    frame_test = cv2.imread(osp.join(img_path, img_title))

                    if frame_test is None:
                        print("no image")
                        sys.exit()

                    """ bbox visualization
                    """ 
#                    vis_bbox(frame_test, *get_bbox( width, height, *p0, *p1))


                    """ file_ID, x1, y1, x2, y2, file_name  순으로 .csv 파일에 레이블링 
                    """
                    csv_wr.writerow([instance[0], *p0, *p1 , img_title])    # 레이블 데이터 저장 
                                                                            # (ref) https://devpouch.tistory.com/55

            video_clip.release() # 끝났으면 방출 
    
    csv_file.close()


def save_frame(savePath, title,frame):

    Path = osp.join(savePath, title)

    try:
        cv2.imwrite(Path, frame)
        
        return True  
    
    except cv2.error as e: 
        """ OpenCV 에러에 대한 try-except 사용하기 
            (ref) https://dojang.io/mod/page/view.php?id=2398
            (ref) https://stackoverflow.com/questions/8873657/how-to-catch-opencv-error-in-python
        """
        print(f"Error message: {e}")
        print(f"Error path: {Path}")

        return None

    










# %%
if __name__ == '__main__':
    train_list = pd.read_csv(osp.join(dataset_path, 'ava_v2.2', 'ava_train_v2.2.csv'), header=None)     # 컬럼에 이름이 없다; header=None
                                                                                                        # (ref) https://rfriend.tistory.com/250
    val_list = pd.read_csv(osp.join(dataset_path, 'ava_v2.2', 'ava_val_v2.2.csv'), header=None)


    unique_trainNames = pd.unique(train_list.loc[:,0])  # (ref) https://vincien.tistory.com/16
    unique_valNames = pd.unique(val_list.loc[:,0]) 
    

    loop_getImageFrame(train_list, trainVideo_path, train_imgPath, unique_trainNames, train_labelPath )  # get train data
    loop_getImageFrame(val_list, valVideo_path, val_imgPath, unique_valNames, val_labelPath )  # get val data






# %%
