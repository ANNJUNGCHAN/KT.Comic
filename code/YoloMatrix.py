### Package Loading
# Base
import pandas as pd
import numpy as np

# Path
from glob import glob
import os

# Visualization
import cv2

# TQDM
from tqdm.auto import tqdm

# Torch for Yolov5
import torch

# Extraction Data Directory
import Extraction as ex

### Saving directory CSV
def DirectoryCSV(path,save_path) :
    '''Yolo의 결과를 csv로 저장하는 함수
    ---input---
    path : 데이터가 들어있는 경로
    save_path : 데이터를 저장할 경로
    ---output---
    csv 경로가 추가된 데이터 프레임
    '''
    data = ex.DataDirectory(path)
    data["save"] = data["dir"].str.split(".").apply(lambda x : x[0] + ".csv")
    data["csv"] = data["save"].str.split("/").apply(lambda x : x[-1])
    data["save"] = save_path + "/" + data["class"] + "/"  + data["csv"]
    data["folder_path"] = save_path + "/" + data["class"]
    folder_path = list(data["folder_path"].unique())
    data.drop(columns = ["folder_path"], inplace = True)
    return data, folder_path

### Yolov5 Matrix
def YoloMatrix(dataframe,folder_path,x_start, x_stop, y_start, y_stop) :
    '''Yolo에서 관측된 객체들을 빈도로 저장한 벡터를 형성하는 함수
    ---input---
    DirectoryCSV에서 추출된 값
    dataframe : CSV의 경로가 포함된 데이터 셋
    folder_path : 임시 저장될 csv가 있어야하는 클래스별 폴더의 경로
    x_start, x_stop, y_start, y_stop : center crop bounding box
    ---코미디 빅리그 기준 설정---
    x_start = 160
    x_stop = 740
    y_start = 0
    y_stop = 410
    ---output---
    yolo_train : Yolo에서 관측된 객체들을 빈도로 저장한 벡터
    all_set : Yolo에서 관측된 개체들
    '''
    for i in range(len(folder_path)) : 
        if not os.path.exists(folder_path[i]) :
            os.mkdir(folder_path[i])

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s',pretrained = True)
    
    for i in tqdm(range(len(dataframe))) : 
        # Image
        img = ex.center_selection(dataframe["dir"][i],x_start, x_stop, y_start, y_stop)
        # Inference
        results = model(img)
        pd.DataFrame(results.pandas().xyxy[0]["name"].value_counts()).rename(columns = {'name' : dataframe["class"][i] + "_" + dataframe["file"][i]}).T.to_csv(dataframe["save"][i])
    
    all_set = set(pd.read_csv(dataframe["save"][0]).columns)
    for i in tqdm(range(1,len(dataframe))) :
        add_col = list(pd.read_csv(dataframe["save"][i]).columns)
        for j in add_col :
            all_set.add(j)
    
    yolo_train = pd.DataFrame(columns= list(all_set))
    yolo_train =yolo_train.T.reset_index()

    for i in tqdm(range(len(dataframe))) :
        temp = pd.read_csv(dataframe["save"][i]).T.reset_index().rename(columns = {0:i})
        yolo_train = pd.merge(yolo_train,temp, how  = "left", on = "index")
    
    yolo_train.index = yolo_train["index"]
    yolo_train = yolo_train.drop(columns = ["index"])
    yolo_train = yolo_train.T
    yolo_train = pd.concat([yolo_train[["Unnamed: 0"]],yolo_train.drop(columns = ["Unnamed: 0"])],axis = 1)
    yolo_train = yolo_train.rename(columns = {"Unnamed: 0" : "file_name"})
    yolo_train = yolo_train.fillna(0)

    if 'Unnamed: 0' in all_set :
        all_set.remove('Unnamed: 0')

    return yolo_train, all_set

### Make Test Yolov5Matrix
def MakeTestMatrix(yolo_train, yolo_test,mode) :
    '''Test YoloMatrix를 형성해주는 함수. Train에서 탐지된 객체만 이용하도록 함.
    ---input---
    yolo_train : Train Yolo Matrix를 넣어준다.
    yolo_test : Test Yolo Matrix를 넣어준다.
    mode : "Used" or else : Used를 사용하면 우리가 학습시킨 모델에 최적화되도록 Test Yolo Matrix를 정제
    ---output---
    test_temp : 정제된 Yolo Matrix
    '''
    if mode == "Used" : 
        col = ['file_name', 'cup', 'handbag', 'book', 'suitcase', 'stop sign',
        'umbrella', 'microwave', 'cow', 'bench', 'cat', 'orange', 'tie',
        'remote', 'baseball bat', 'bottle', 'wine glass', 'skateboard',
        'broccoli', 'car', 'tv', 'person', 'snowboard', 'cell phone',
        'dining table', 'bird', 'sports ball', 'cake', 'parking meter', 'kite',
        'toilet', 'clock', 'backpack', 'mouse', 'fire hydrant', 'train',
        'frisbee', 'traffic light', 'vase', 'motorcycle', 'laptop', 'horse',
        'chair', 'bed', 'airplane', 'bowl', 'baseball glove', 'tennis racket',
        'potted plant', 'refrigerator', 'toothbrush', 'donut', 'oven', 'skis',
        'dog', 'bicycle', 'couch', 'teddy bear']
    else :
        col = yolo_train.columns
    test_temp = pd.DataFrame(columns = col)
    for i in yolo_test.columns :
        test_temp[i] = yolo_test[i]
    test_temp = test_temp.fillna(0)
    return test_temp


    

    


