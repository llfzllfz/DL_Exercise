import os

# DATA_ROOT_PATH = 'E:\\2022_exercise\\VOC2007\\VOCdevkit\\VOC2007'
# TRAIN_DATA_PATH = os.path.join(DATA_ROOT_PATH, 'JPEGImages')
# LABEL_DATA_PATH = os.path.join(DATA_ROOT_PATH, 'SegmentationClass')

DATA_ROOT_PATH = 'E:\\2022_exercise\\dataset\\clean_images'
TRAIN_DATA_PATH = os.path.join(DATA_ROOT_PATH, 'images')
LABEL_DATA_PATH = os.path.join(DATA_ROOT_PATH, 'profiles')

val_radio = 0.1
batch_size = 2
lr = 1e-3
classes = 2
epochs = 100
model_path = '_model.ckpt'

predict_pic = '000032'