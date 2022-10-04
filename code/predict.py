### Load Package
# MY Model
import Video2Image
import Model
# TensorFlow and Keras
import tensorflow as tf
from keras import backend as K
# Base
import pandas as pd
import numpy as np
# Visualization
from PIL import  Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

### Config
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
n_class = 1
sr_model_path = "/content/gdrive/MyDrive/Comic/Package/Package/TF-LapSRN/export/LapSRN_x8.pb"
data_dir = "/content/gdrive/MyDrive/Comic/TestData"
extra_dir = "/content/sample_data/test"
model_dir = '/content/gdrive/MyDrive/Comic/Package/Model/model.h5'
line_start = 0
line_end = 0
prediction_dataframe_save = "/content/gdrive/MyDrive/Comic/TestData/prediction.csv"
mode = "Used"
line_use = "False"

### Make Predict DataSet
original_train, train_image,train_vector = Model.ModelPredictSet(data_dir,extra_dir,yolo_x_start,yolo_x_end,yolo_y_start,yolo_y_end,banner_x_start,banner_x_end,banner_y_start,banner_y_end,sr_model_path,mode)

### Model Load
new_model = tf.keras.models.load_model(model_dir)

### Prediction
prediction = new_model.predict([train_image,train_vector])
dataframe = pd.DataFrame(prediction)
prediction = pd.concat([original_train,dataframe],axis = 1)
prediction["predict"] = np.argmax(np.array(prediction[[0,1,2,3]]),axis = 1)
prediction["max"] = np.max(np.array(prediction[[0,1,2,3]]),axis = 1)
prediction["std"] = np.std(np.array(prediction[[0,1,2,3]]),axis = 1)
prediction["number"] = prediction["file"].str.split(".").apply(lambda x : x[0])
prediction["number"] = prediction["number"].astype(int)
prediction = prediction.sort_values(by = "number")
prediction.reset_index(drop = True, inplace = True)

### Predict to Class Name
for i in range(len(prediction)) :
    if prediction["std"][i] < 0.433 :
        prediction["predict"][i] = "None"
    elif prediction["predict"][i] == 0 :
        prediction["predict"][i] = "두분사망토론"
    elif prediction["predict"][i] == 1 :
        prediction["predict"][i] = "코빅엔터"
    elif prediction["predict"][i] == 2 :
        prediction["predict"][i] = "결혼해두목"
    else :
        prediction["predict"][i] = "싸이코러스"

### Define Line to more accurate predict
prediction["line"] = 0
for i in range(len(prediction)) :
    if prediction["predict"][i] == "None" :
        prediction["line"][i] = 0
    elif prediction["predict"][i] == "두분사망토론" :
        prediction["line"][i] = 1
    elif prediction["predict"][i] == "코빅엔터" :
        prediction["line"][i] = 2
    elif prediction["predict"][i] == "결혼해두목" :
        prediction["line"][i] = 3
    else :
        prediction["line"][i] = 4

## accurate predict by using line
if line_use == "True" :
    for i in range(len(prediction)) :
        if (i < line_start) or (i >line_end) :
            prediction["predict"][i] = "None"
            prediction["line"][i] = 0
    else :
        pass
prediction.to_csv(prediction_dataframe_save)


### Result
for i in range(len(prediction)) :
    # image loading
    img = Image.open(prediction["dir"][i])
    width, height = img.size

    # Draw
    draw = ImageDraw.Draw(img)

    # WaterMark
    text = prediction["predict"][i]

    # font
    font = ImageFont.truetype('/content/gdrive/MyDrive/Comic/NanumPen.otf', 100)

    # txt width and height
    width_txt, height_txt = draw.textsize(text, font)

    # WaterMark Location
    margin = 100
    x = 0 + width_txt + margin
    y = 0

    # Text apply
    draw.text((x, y), text, fill='red', font=font)

    # Save Image
    img.save(prediction["dir"][i])

