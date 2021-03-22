# AVA Dataset Processing for Person Detection
This repository is about AVA dataset processing to train object detection for the person class only. <br/>
The AVA dataset can also be used for various tasks, not only person detection. Please, check [the official document](https://research.google.com/ava/index.html) for more details.

<br/>

### AVA Actions Download 
* Make a directory, named```'dataset'``` , in the current working directory.
* [Click here](https://research.google.com/ava/download.html#ava_kinetics_download) and download the '**ava_v2.2.zip**' to the ```dataset```. Then, extract the files.   
* '**ava_v2.2.zip**' includes various annotation files. 
    ```bash
    # Run setup.sh for simplification.
    ~$ bash setup.sh
    ```

<br/>

### Requirements 
Install required python packages with PyPI. 
* Run the command like below:
    ``` bash
    ~$ pip install -r requirements.txt
    ```





<br/>

### Downloading YouTube Videos and Taking Image Frames from Them 
* The '**ava_youtube_download.py**' is for downloading YouTube Videos according to the video name lists.
* The '**cut_frames_from_videos.py**' will cut frames from the videos and return images but also bbox annotations (annotations are written in 'csv' format).
    ``` bash
    # Run the code following order below.
    ~$ python ava_youtube_download.py
    ~$ python cut_frames_from_video.py
    ```


<br/>

### Detection Labels 
* [The AVA v2.2 dataset contains 430 videos split into 235 for training, 64 for validation, and 131 for test](https://research.google.com/ava/download.html).
* The most label information is about **human localization** and **action recognition**.
* For achieving the bbox labels of [humans](https://www.reddit.com/r/etymology/comments/63ymz1/why_is_it_humans_instead_of_humen/), use '**ava_train_v2.2.csv**', '**ava_val_v2.2.csv**', and '**ava_test_v2.2.txt**'.
* In my case, I will use `yolo_v5 format` like in this [tutorial](https://blog.roboflow.com/how-to-train-a-custom-mobile-object-detection-model/).
    ``` bash
    # For example,
    # class |  center x | center y | width | height  in .txt file 
    0 0.002 0.118 0.714 0.977 

    ```
* In order to get the ```YOLO``` format annotation, run the code like below:
    ```bash
    ~$ python cvt_annotation_format_csv_to_txt.py
    ```

(ongoing...)



***
## Reference 
[1] [AVA Dataset Downloader Script, alainary, github](https://github.com/alainray/ava_downloader) / 동영상 다운받고 프레임 처리하는 방식 참고 <br/>
[2] [AVA dataset](https://research.google.com/ava/download.html) / DB 홈피 <br/>
