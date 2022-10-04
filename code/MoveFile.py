### Package Loading
# Move File
import shutil

# Path
import os
from glob import glob

# Base
import pandas as pd

# TQDM
from tqdm.auto import tqdm

def MoveFile(file_dir, move_dir) :
    '''파일 이동 함수
    파일을 이동시킵니다.
    ---input---
    file_dir : 클래스별로 묶인 파일이 있는 폴더
    move_dir : 옮길 폴더
    ---output---
    data : 옮긴 위치
    '''
    
    # Class 종류 파악
    class_list = os.listdir(file_dir)
    if '.ipynb_checkpoints' in class_list :
        class_list.remove('.ipynb_checkpoints')
    
    # 옮길 데이터 파악
    data_dir_list = []
    for i in class_list :
        temp = pd.DataFrame()
        temp["dir"] = glob(file_dir + "/" + i + "/*.jpg")
        data_dir_list.append(temp)
    
    # 옮길 데이터 프레임 생성
    data = pd.concat(data_dir_list,axis = 0)
    data.reset_index(drop = True, inplace = True)

    # 파일 이름
    data["file"] = data["dir"].str.split("/").apply(lambda x : x[-1])

    # target
    data["target"] = data["dir"].str.split("/").apply(lambda x : x[-2])

    # move directory
    data["move"] = move_dir + "/"+ data["target"] + "/" +  data["dir"].str.split("/").apply(lambda x : x[-1])

    # move directory class generator
    move_folder_list = move_dir + "/" + data["target"]
    move_folder_list = list(move_folder_list.unique())

    for i in range(len(move_folder_list)) :
        if not os.path.exists(move_folder_list[i]) :
            os.mkdir(move_folder_list[i])

    # RUN!
    for i in tqdm(range(len(data))) :
        shutil.move(data["dir"][i], data["move"][i])
    
    return data


    
    