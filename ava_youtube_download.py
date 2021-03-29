# -*- coding: utf-8 -*-

#%%
import os 
import os.path as osp 
from glob import glob

from tqdm  import tqdm 
import subprocess   # 파이썬에서 쉘 명령을 실행할 수 있게 해주는 라이브러리 
                    # os.system 보다 더 다양항 기능을 제공함
                    # (ref) http://www.incodom.kr/%ED%8C%8C%EC%9D%B4%EC%8D%AC/%EB%9D%BC%EC%9D%B4%EB%B8%8C%EB%9F%AC%EB%A6%AC/subprocess
import youtube_dl   # 유튜브 영상 다운로드 패키지; (ref) https://velog.io/@okstring/%ED%8C%8C%EC%9D%B4%EC%8D%AC%EC%9C%BC%EB%A1%9C-%EC%9C%A0%ED%8A%9C%EB%B8%8C-%EB%8F%99%EC%98%81%EC%83%81-%EB%8B%A4%EC%9A%B4%EB%B0%9B%EA%B8%B0youtubedl
import pandas as pd

                    

cwd = os.getcwd()
dataset_path = osp.join(cwd, 'dataset')

trainDir_path = osp.join(dataset_path, 'train')
valDir_path = osp.join(dataset_path, 'val')


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
#                     2. Download from YouTube                      #
# ================================================================= #
# %%
""" youtube_dl 사용법 참고: 
(ref) https://velog.io/@okstring/%ED%8C%8C%EC%9D%B4%EC%8D%AC%EC%9C%BC%EB%A1%9C-%EC%9C%A0%ED%8A%9C%EB%B8%8C-%EB%8F%99%EC%98%81%EC%83%81-%EB%8B%A4%EC%9A%B4%EB%B0%9B%EA%B8%B0youtubedl
(ref) https://stackoverflow.com/questions/18054500/how-to-use-youtube-dl-from-a-python-program
(ref) https://tech.dslab.kr/2019/09/10/python-youtube_dl/

"""
def Download(name, base_dir):
    file_name = f"https://www.youtube.com/watch?v={str(name)}"  # 동영상 URL
    dir = osp.join(base_dir, str(name)) 
    createDirectory(dir)
    output_dir = osp.join(dir, '%(id)s__v__%(title)s.%(ext)s') # dir 안에서 'id._v_.영상제목.확장자' 형식으로 다운받음

    
    ydl_opts = {
        'format': 'bestvideo/best',   # 가장 좋은 화질로 선택 
        'outtmpl': output_dir ,       # 다운로드 경로 설정 
        'ignoreerrors': True,
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([file_name])



def loop_Download(fileName_list, Path):

    loop = tqdm(enumerate(fileName_list), total=len(fileName_list))  

    for idx, get_filename in loop:
        Download(get_filename, Path)



# ================================================================= #
#                        3. Run your main block                     #
# ================================================================= #
# %%
if __name__ == '__main__':
    
    train_list = pd.read_csv(osp.join(dataset_path, 'ava_v2.2', 'ava_train_v2.2.csv'), header=None)     # 컬럼에 이름이 없다; header=None
                                                                                                        # (ref) https://rfriend.tistory.com/250
    val_list = pd.read_csv(osp.join(dataset_path, 'ava_v2.2', 'ava_val_v2.2.csv'), header=None)


    unique_trainNames = pd.unique(train_list.loc[:,0])  # (ref) https://vincien.tistory.com/16
    unique_valNames = pd.unique(val_list.loc[:,0]) 
    print(f"Total train_video numbers: {len(unique_trainNames)}")
    print(f"Total val_video numbers: {len(unique_valNames)}")
    
#    loop_Download(unique_trainNames, trainDir_path)  # train 용 비디오 다운 
    loop_Download(unique_valNames, valDir_path )    # val 용 비디오 다운 


# %%
