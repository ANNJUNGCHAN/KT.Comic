### Package Loading
# Keras & Tensorflow
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation,Flatten
from keras.layers import Reshape, Lambda, BatchNormalization, GlobalAveragePooling1D
from keras.layers.merge import add, concatenate, Concatenate
from keras.models import Model
from keras.layers.recurrent import LSTM
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping


# Base
import pandas as pd
import numpy as np

# directory
from glob import glob

# cv
import cv2

# tqdm
from tqdm.auto import tqdm

# Base
import pandas as pd
import numpy as np

# train test split
from sklearn.model_selection import train_test_split

# Label Encoding
from sklearn.preprocessing import LabelEncoder

# MyPackages
import YoloMatrix as ym
import Extraction as ex

### Make Model Dataset
def ModelDataSet(dir, save, test_size,yolo_x_start,yolo_x_end,yolo_y_start,yolo_y_end,banner_x_start,banner_x_end,banner_y_start,banner_y_end,sr_model_path,n_class,mode) :
    '''모델링에 필요한 데이터 셋 추출
    ---코미디 빅리그 관련 세팅---
    yolo_x_start = 160
    yolo_x_end = 740
    yolo_y_start = 0
    yolo_y_end = 410
    test_size = 0.2
    banner_x_start = 65
    banner_x_end = 130
    banner_y_start = 55
    banner_y_end = 75
    yolo_length = 60
    n_class = 4
    ---input---
    dir : DataSet의 경로
    save : 임시저장폴더
    test_size : test 데이터 셋 비율
    yolo_x_start, yolo_x_end,yolo_y_start,yolo_y_end : yolo center crop 좌표
    banner_x_start,banner_x_end,banner_y_start,banner_y_end : banner crop 좌표
    sr_model_path : super resolution model path
    n_class : 클래스 갯수
    mode : "Used" : 이전에 사용했던 모델 구현
    ---output---
    train_image : train image 데이터 셋
    train_vector : train vector 데이터 셋
    test_image : valid image 데이터 셋
    test_vector : valid vector 데이터 셋
    train_y : train의 target 값
    test_y : test의 target 값
    predict_label : 라벨이 뜻하는 의미
    '''

    data, folder_name = ym.DirectoryCSV(dir,save)
    yolo_train, all_set = ym.YoloMatrix(data,folder_name,yolo_x_start,yolo_x_end,yolo_y_start,yolo_y_end)
    yolo_train = ym.MakeTestMatrix(yolo_train,yolo_train,mode = "Used")

    # Train
    train = ex.DataDirectory(dir)
    train = train.rename({'file':'file_name'},axis = 1)
    train["file_name"] = train["class"] + "_" + train["file_name"]
    train = pd.merge(train,yolo_train,how = "left", on = "file_name")

    train, test, _, _ = train_test_split(train, train["class"] , test_size = test_size, random_state = 42, shuffle = True, stratify = train["class"])

    train.reset_index(drop = True, inplace = True)
    test.reset_index(drop = True, inplace = True)
    yolo_length = len(train.columns)

    ### image save
    for i in tqdm(range(len(train))) :
        train["dir"][i] = ex.selection(train["dir"][i],banner_x_start, banner_x_end, banner_y_start, banner_y_end,sr_model_path)

    ### vector save
    train["vector"] = 0
    for i in tqdm(range(len(train))) :
        if mode == "Used" :
            train["vector"][i] = np.array(train.iloc[i,3:61])
        else :
            train["vector"][i] = np.array(train.iloc[i,3:])

    ### clean and save
    col_list = list(train.iloc[:,3:yolo_length].columns)
    train = train.drop(columns = col_list)

    ### image save
    for i in tqdm(range(len(test))) :
        test["dir"][i] = ex.selection(test["dir"][i],banner_x_start, banner_x_end, banner_y_start, banner_y_end,sr_model_path)

    ### vector save
    test["vector"] = 0
    for i in range(len(test)) :
        if mode == "Used" :
            test["vector"][i] = np.array(test.iloc[i,3:61])
        else :
            test["vector"][i] = np.array(test.iloc[i,3:])

    ### clean and save
    col_list = list(test.iloc[:,3:yolo_length].columns)
    test = test.drop(columns = col_list)

    ### Label Encoding
    le = LabelEncoder()
    le.fit(train["class"])
    predict_label = le.classes_
    train["class"] = le.transform(train["class"].values)
    test["class"] = le.transform(test["class"].values)

    ### features and target split
    train_y = to_categorical(train["class"].values,n_class)
    test_y = to_categorical(test["class"].values,n_class)

    train_image = []
    for i in range(len(train)) :
        temp = train["dir"].values[i]
        train_image.append(temp)
    
    train_image = np.array(train_image)

    train_vector = []
    for i in range(len(train)) :
        temp = train["vector"].values[i]
        train_vector.append(temp)

    train_vector = np.array(train_vector)

    test_image = []
    for i in range(len(test)) :
        temp = test["dir"].values[i]
        test_image.append(temp)

    test_image = np.array(test_image)

    test_vector = []
    for i in range(len(test)) :
        temp = test["vector"].values[i]
        test_vector.append(temp)

    test_vector = np.array(test_vector)

    train_image = train_image.astype("float32")
    train_vector = train_vector.astype("float32")
    test_image = test_image.astype("float32")
    test_vector = test_vector.astype("float32")  
    
    return train_image,train_vector,test_image, test_vector, train_y,test_y,predict_label

### Make Predict Dataset
def ModelPredictSet(dir, save,yolo_x_start,yolo_x_end,yolo_y_start,yolo_y_end,banner_x_start,banner_x_end,banner_y_start,banner_y_end,sr_model_path,mode) :
    '''모델링에 필요한 데이터 셋 추출
    ---코미디 빅리그 관련 세팅---
    yolo_x_start = 160
    yolo_x_end = 740
    yolo_y_start = 0
    yolo_y_end = 410
    banner_x_start = 65
    banner_x_end = 130
    banner_y_start = 55
    banner_y_end = 75
    yolo_length = 60
    ---input---
    dir : DataSet의 경로
    save : 임시저장폴더
    yolo_x_start, yolo_x_end,yolo_y_start,yolo_y_end : yolo center crop 좌표
    banner_x_start,banner_x_end,banner_y_start,banner_y_end : banner crop 좌표
    sr_model_path : super resolution model path
    mode : "Used" : 이전에 사용했던 모델 구현
    ---output---
    original_train : 예측하려는 경로
    train_image : train image 데이터 셋
    train_vector : train vector 데이터 셋
    '''

    data, folder_name = ym.DirectoryCSV(dir,save)
    yolo_train, all_set = ym.YoloMatrix(data,folder_name,yolo_x_start,yolo_x_end,yolo_y_start,yolo_y_end)
    yolo_train = ym.MakeTestMatrix(yolo_train,yolo_train,mode = "Used")

    # Train
    train = ex.DataDirectory(dir)
    original_train = train
    train = train.rename({'file':'file_name'},axis = 1)
    train["file_name"] = train["class"] + "_" + train["file_name"]
    train = pd.merge(train,yolo_train,how = "left", on = "file_name")

    train.reset_index(drop = True, inplace = True)
    yolo_length = len(train.columns)

    ### image save
    for i in tqdm(range(len(train))) :
        train["dir"][i] = ex.selection(train["dir"][i],banner_x_start, banner_x_end, banner_y_start, banner_y_end,sr_model_path)

    ### vector save
    train["vector"] = 0
    for i in tqdm(range(len(train))) :
        if mode == "Used" :
            train["vector"][i] = np.array(train.iloc[i,3:61])
        else :
            train["vector"][i] = np.array(train.iloc[i,3:])

    ### clean and save
    col_list = list(train.iloc[:,3:yolo_length].columns)
    train = train.drop(columns = col_list)

    train_image = []
    for i in range(len(train)) :
        temp = train["dir"].values[i]
        train_image.append(temp)
    
    train_image = np.array(train_image)

    train_vector = []
    for i in range(len(train)) :
        temp = train["vector"].values[i]
        train_vector.append(temp)

    train_vector = np.array(train_vector)

    train_image = train_image.astype("float32")
    train_vector = train_vector.astype("float32")
    
    return original_train, train_image,train_vector

### Model Define

def define_model(img_w,img_h, vector_shape,num_classes) :
    '''Contour LapSRN to CRNN + Yolov5 detected vector 모델 정의
    ---input---
    img_w : 이미지 너비
    img_h : 이미지 높이
    vector_shape : Yolov5 Matrix 크기
    num_classes : class 종류 갯수
    ---코미디 빅리그 기본 설정---
    img_w = 160
    img_h = 520
    vector_shape = 58
    num_classes = 4
    ---output---
    model : Contour LapSRN to CRNN + Yolov5 detected vector 모델
    '''

    keras.backend.clear_session()

    input_shape = (img_w, img_h, 3)

    # Make Networkw
    inputs = Input(name='the_input', shape=input_shape, dtype='float32')  
    inputs_vector = Input(name = "input_vector",shape = (vector_shape,), dtype = 'float32')

    # Dense Vector
    vector = Dense(1024, activation = "relu")(inputs_vector)
    vector = BatchNormalization()(vector)
    vector = Dense(512, activation = "relu")(vector)
    vector = BatchNormalization()(vector)


    # Convolution layer (VGG)
    inner = Conv2D(64, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(inputs)  
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner) 

    inner = Conv2D(128, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(inner)  
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner)  

    inner = Conv2D(256, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(inner)  
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = Conv2D(256, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal')(inner)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(1, 2), name='max3')(inner) 

    inner = Conv2D(512, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal')(inner)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = Conv2D(512, (3, 3), padding='same', name='conv6')(inner)  
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(1, 2), name='max4')(inner)  

    # CNN to RNN
    inner = Reshape(target_shape=((int(img_w/4), int(img_h/16) * 512)), name='reshape')(inner)  
    inner = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(inner) 

    # RNN layer
    lstm_1 = LSTM(128, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(inner)  
    lstm_1b = LSTM(128, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm1_b')(inner)
    reversed_lstm_1b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1)) (lstm_1b)

    lstm1_merged = add([lstm_1, reversed_lstm_1b]) 
    lstm1_merged = BatchNormalization()(lstm1_merged)

    lstm_2 = LSTM(128, return_sequences=True, kernel_initializer='he_normal', name='lstm2')(lstm1_merged)
    lstm_2b = LSTM(128, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm2_b')(lstm1_merged)
    reversed_lstm_2b= Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1)) (lstm_2b)

    lstm2_merged = concatenate([lstm_2, reversed_lstm_2b]) 
    lstm2_merged = BatchNormalization()(lstm2_merged)
    global_average_pool = GlobalAveragePooling1D()(lstm2_merged)
    final_cnn = BatchNormalization()(global_average_pool)

    # Concatenate
    concat_IV = Concatenate(axis = 1)([vector,final_cnn])
    concat_IV =BatchNormalization()(concat_IV)

    # transforms RNN output to character activations:
    y_pred = Dense(num_classes, kernel_initializer='he_normal',name='dense2',activation = "softmax")(concat_IV)
    model = keras.models.Model([inputs,inputs_vector],y_pred)

    return model

### Model Run
def Run(model,checkpoint_path,model_save_path,train_image,train_vector,test_image,test_vector,train_y,test_y,epoch) :
    '''모델 학습 함수
    ---input---
    model : 정의된 모델
    checkpoint_path : 체크포인트 경로
    model_save_path : 모델 저장 경로
    train_image : train image 데이터 셋
    train_vector : train vector 데이터 셋
    test_image : valid image 데이터 셋
    test_vector : valid vector 데이터 셋
    train_y : train의 target 값
    test_y : test의 target 값
    ---코미디 빅리그 기본설정---
    checkpoint_path = "/content/gdrive/MyDrive/Comic/callback/checkpoint.ckpt"
    model_save_path = "/content/gdrive/MyDrive/Comic/Modeling/model.h5"
    ---output---
    model : 학습된 모델
    history : 모델 학습 기록
    '''
    # complie
    modelcheckpoint = ModelCheckpoint(checkpoint_path, monitor = "val_loss", mode = "min", save_best_only = True, save_weights_only = True)
    earlystopping = EarlyStopping(monitor = "val_loss", mode = "min", verbose = 1, patience = 10)
    callback_lists = [modelcheckpoint, earlystopping]
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer='adam', metrics=["accuracy"])

    # RUN!
    model.fit([train_image, train_vector],
                    train_y,
                    batch_size = 8,
                    epochs = epoch,
                    validation_data=([test_image, test_vector],test_y),
                    callbacks = callback_lists
                    )

    model.save(model_save_path)
    return model






















