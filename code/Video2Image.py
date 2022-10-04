### Package Loading

# 데이터 핸들링
import pandas as pd
import numpy as np

# 시드 고정 함수
import random
import os

# Visualization
import cv2

# 경로 셋
from glob import glob

# processing bar
from tqdm.auto import tqdm

# warning filters
import warnings
warnings.filterwarnings(action='ignore')


def VideoDF_SameClass(video_path,save_folder,how_many, version) :
    """video 추출을 위한 데이터 프레임 형성
    class imbalance를 막아주면서 데이터를 추출합니다.
    ---input---
    - video_path : 비디오가 저장된 폴더
    - save_folder : 비디오를 save할 폴더
    - how many : 대략 몇개의 비디오를 추출할 것인가.
    - version : floor : 내림, ceil : 올림, round : 반올림
    ---output---
    DataFrame
    - path : 경로
    - name : class 이름
    - save : 이미지 저장 경로
    - time_measurement_unit : 몇 프레임 당 이미지를 추출할 것인가
    """
    video_list = glob(video_path + "/*.mp4")
    video_df = pd.DataFrame()
    video_df["path"] = video_list
    video_df["name"] = video_df["path"].str.split("/").apply(lambda x : x[-1].split(".")[0])
    video_df["save"] = save_folder + "/" + video_df["name"]
    time_measurement_unit_list = []
    length_list = []
    
    for i in range(len(video_list)) : 
        if not os.path.exists(video_df["save"][i]) :
            os.mkdir(video_df["save"][i])
        video = cv2.VideoCapture(video_df["path"][i])
        length = video.get(cv2.CAP_PROP_FRAME_COUNT)
        time_measurement_unit = length / how_many
        length_list.append(length)
        if version == "floor" :
            time_measurement_unit_list.append(np.floor(time_measurement_unit))
        elif version == "ceil" :
            time_measurement_unit_list.append(np.ceil(time_measurement_unit))
        else :
            time_measurement_unit_list.append(np.round(time_measurement_unit))
    video_df["time_measurement_unit"] = time_measurement_unit_list
    video_df["length"] = length_list
    return video_df

def extraction_video(video_df) :
    """video 추출을 위한 데이터 프레임을 넣으면, 비디오에서 이미지를 추출
    ---input---
    video_df : video의 경로, 각각의 class 이름, 이미지 저장 경로, 몇 프레임 당 이미지를 추출할 것인지가 들어있는 dataframe
    ---output---
    비디오에서 이미지 추출
    """
    for i in tqdm(range(len(video_df))) :
        TIME_MEASUREMENT_UNIT = video_df["time_measurement_unit"][i]  # 몇 프레임 단위로 이미지를 끊을 것인지를 판단함
        count = 0
        if not os.path.exists(video_df["save"][i]) :
            os.mkdir(video_df["save"][i])
        video = cv2.VideoCapture(video_df["path"][i])
        while video.isOpened() :
            ret, frame = video.read()
            if ret :
                frame_sec = video.get(cv2.CAP_PROP_POS_FRAMES)
                if (frame_sec % TIME_MEASUREMENT_UNIT  == 0 ) :
                    filename = video_df["save"][i] + "/" + str(frame_sec) + ".jpg"
                    cv2.imwrite(filename, frame)
            else :
                break
            count += 1
        video.release()


def VideoDF_Seconds(video_path,save_folder,TIME_MEASUERMENT_UNIT) :
    """video 추출을 위한 데이터 프레임 형성 및 비디오에서 이미지 추출
    N초 단위로 비디오를 추출합니다.
    ---input---
    - video_path : 비디오가 저장된 폴더
    - save_folder : 비디오를 save할 폴더
    - TIME_MEASUREMENT_UNIT : 몇 초 단위로 비디오에서 이미지를 추출할지 결정
    ---output---
    DataFrame
    - path : 경로
    - name : class 이름
    - save : 이미지 저장 경로
    """
    video_list = glob(video_path + "/*.mp4")
    video_df = pd.DataFrame()
    video_df["path"] = video_list
    video_df["name"] = video_df["path"].str.split("/").apply(lambda x : x[-1].split(".")[0])
    video_df["save"] = save_folder + "/" + video_df["name"]

    for i in tqdm(range(len(video_df))) :
    
        count = 0
        
        if not os.path.exists(video_df["save"][i]) :
            os.mkdir(video_df["save"][i])
        
        # 비디오 불러오기
        video = cv2.VideoCapture(video_df["path"][i])

        # 1초당 끊어지는 프레임 갯수
        fps = video.get(cv2.CAP_PROP_FPS)
        divide = TIME_MEASUERMENT_UNIT * fps
        
        while video.isOpened() :
            ret, frame = video.read()
            if ret :
                frame_sec = video.get(cv2.CAP_PROP_POS_FRAMES)
                if (frame_sec % divide  == 0 ) :
                    filename = video_df["save"][i] + "/" + str(int(frame_sec//divide)) + ".jpg"
                    cv2.imwrite(filename, frame)
            else :
                break
            count += 1
        video.release()

    return video_df

def extraction_result(video_df) :
    """ 얼마나 이미지가 추출되었는지 확인하는 함수
    ---input---
    video_df : video의 경로, 각각의 class 이름, 이미지 저장 경로, 몇 프레임 당 이미지를 추출할 것인지가 들어있는 dataframe
    ---output---
    클래스별 추출된 이미지 갯수
    """
    for i in range(len(video_df)) :
        print(video_df["name"][i] + " : ",len(glob(video_df["save"][i] + "/*.jpg")))

