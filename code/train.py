### Package Loading
import Model

### Parameters
dir = "/content/gdrive/MyDrive/Comic/Package_running/Train"
save = "/content/sample_data/train"
sr_model_path = '/content/gdrive/MyDrive/Comic/Package/Package/TF-LapSRN/export/LapSRN_x8.pb'
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
mode = 'Used'
checkpoint_path = "/content/gdrive/MyDrive/Comic/callback/checkpoint.ckpt"
model_save_path = "/content/gdrive/MyDrive/Comic/Modeling/model.h5"
img_w = 160
img_h = 520
vector_shape = 58
num_classes = 4
epoch = 2

### RUN!!
train_image,train_vector,test_image, test_vector, train_y,test_y,predict_label = Model.ModelDataSet(dir, save, test_size,yolo_x_start,yolo_x_end,yolo_y_start,yolo_y_end,banner_x_start,banner_x_end,banner_y_start,banner_y_end,sr_model_path,n_class,mode)
model = Model.define_model(img_w,img_h, vector_shape,num_classes)
Model.Run(model,checkpoint_path,model_save_path,train_image,train_vector,test_image,test_vector,train_y,test_y,epoch)
