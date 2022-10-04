### Package Loading
# opencv
import cv2
import matplotlib.pyplot as plt

# base
import numpy as np
import pandas as pd

# file moving
from glob import glob
import os
import shutil

# tqdm
from tqdm.auto import tqdm

# up resolution(실행전 설치할 것)
#!pip install opencv-contrib-python
#!pip install opencv-contrib-python --upgrade
#!git clone https://github.com/fannymonori/TF-LapSRN.git

### DataDirectory
def DataDirectory(path) :
    """이미지의 경로를 찾아주는 힘수
    ---input---
    path : 이미지가 들어가있는 폴더
    ---output---
    DataFrame
    dir : 이미지 파일의 경로
    file : 파일명
    class : 이미지 파일이 속한 class
    """
    list_dir = os.listdir(path)
    if '.ipynb_checkpoints' in list_dir :
        list_dir.remove('.ipynb_checkpoints')

    dir_data = []
    for i in range(len(list_dir)) :
        temp = pd.DataFrame()
        temp["dir"] = glob(path + "/" + list_dir[i] + "/*")
        dir_data.append(temp)

    dir_data = pd.concat(dir_data)
    dir_data.reset_index(inplace = True, drop = True)
    dir_data["file"] = dir_data["dir"].str.split("/").apply(lambda x : x[-1])
    dir_data["class"] = dir_data["dir"].str.split("/").apply(lambda x : x[-2])
    return dir_data

### Banner Extraction
def selection(path,x_start, x_stop, y_start, y_stop,sr_model_path) :
    """베너부분만 crop한 후 전처리하는 함수
    1. Banner Crop
    2. 8배 고화질 추출
    3. Contour
    ---코미디 빅리그 기준 설정---
    x_start = 65
    x_stop = 130
    y_start = 55
    y_stop = 75
    sr_model_path = '/content/TF-LapSRN/export/LapSRN_x8.pb'
    ---input---
    path : 이미지 파일의 경로
    x_start,x_stop,y_start,y_stop : 배너부분의 bounding box
    sr_model_path : LapSRN_x8.pd의 경로(패키지가 자동으로 다운로드 됩니다.)
    ---output---
    전처리된 이미지
    """
    # Banner Crop
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image[y_start:y_stop ,x_start:x_stop,:]

    # gray 변환
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 8배 고화질 추출
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(sr_model_path)
    sr.setModel('lapsrn', 8)
    upsample_image = sr.upsample(gray)
    upsample_image = cv2.cvtColor(upsample_image, cv2.COLOR_GRAY2RGB)

    # Contour
    img_original = upsample_image
    img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img_gray, ksize=(7,7), sigmaX=0)
    _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5) ,np.uint8)
    dilation = cv2.dilate(thresh, kernel, iterations = 2)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, 3)
    img_contour = cv2.drawContours(img_original, contours, -1, (0, 255, 0), 3)

    return img_contour

### Center Crop
def center_selection(path,x_start, x_stop, y_start, y_stop) :
    """이미지의 정중앙은 crop하는 함수
    ---코미디 빅리그 기준 설정---
    x_start = 160
    x_stop = 740
    y_start = 0
    y_stop = 410
    ---input---
    path : 이미지 파일의 경로
    x_start,x_stop,y_start,y_stop : 배너부분의 bounding box
    ---output---
    crop된 이미지
    """
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image[y_start:y_stop ,x_start:x_stop,:]
    return image
