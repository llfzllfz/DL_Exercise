import matplotlib.pyplot as plt
import config
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from PIL import PngImagePlugin

'''
获取参与训练的图片的文件名
input:
    None
output:
    filename_list -> list
'''
def get_filename_list():
    filename_list = []
    for filename in os.listdir(config.LABEL_DATA_PATH):
        filename_list.append(filename[:-12])
    return filename_list

'''
显示相应的图片
input:
    image -> np.array
    label -> np.array
        None
'''
def show_image(image, label):
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.title(image.shape)
    plt.subplot(1,2,2)
    plt.imshow(label)
    plt.title(label.shape)
    plt.show()


'''
读取图片
input:
    path -> str
output:
    image -> np.array
'''
def read_image(path, resize = (224, 224), use_resize = False, need_palette = 0):
    image = Image.open(path)
    if need_palette == 1:
        palette = image.getpalette()
    if use_resize:
        image = image.resize(resize)
    image = np.array(image)
    if need_palette == 1:
        return image, palette
    return image

'''
得到图片的分类
input:
    filename_list -> list
    c -> int
        设置是否重新扫描所有图片得到分类
        默认为0，不扫描，直接读取内置的结果；若为1，则重新扫描
output:
    classifier_dict -> dict
'''
def get_classifier(filename_list, c = 0):
    assert c == 0 or c == 1, f'c需要设置为0或1'
    if c == 1:
        classifier_dict = {0: 0, 255: 1, 1: 2, 15: 3, 20: 4, 19: 5, 4: 6, 9: 7, 12: 8, 2: 9, 5: 10, 11: 11, 13: 12, 3: 13, 10: 14, 7: 15, 8: 16, 6: 17, 16: 18, 18: 19, 17: 20, 14: 21}
        return classifier_dict
    if c == 2:
        classifier_dict = {0: 0, 255: 1, 253: 2, 247: 3, 244: 4, 254: 5, 251: 6, 249: 7, 242: 8, 252: 9, 240: 10, 231: 11, 245: 12, 236: 13, 250: 14, 248: 15, 246: 16, 226: 17, 243: 18, 4: 19, 219: 20, 6: 21, 2: 22, 10: 23, 1: 24, 5: 25, 3: 26, 241: 27, 11: 28, 12: 29, 7: 30, 16: 31, 21: 32, 15: 33, 9: 34, 238: 35, 35: 36, 227: 37, 19: 38, 14: 39, 8: 40, 18: 41, 17: 42, 13: 43, 235: 44, 26: 45, 22: 46, 232: 47, 237: 48, 28: 49, 239: 50, 23: 51, 24: 52, 20: 53, 27: 54, 38: 55, 32: 56, 230: 57, 229: 58, 30: 59, 25: 60, 233: 61, 37: 62, 228: 63, 36: 64, 31: 65, 218: 66, 222: 67, 29: 68, 225: 69, 234: 70, 221: 71, 34: 72, 224: 73, 39: 74, 220: 75, 217: 76, 33: 77, 223: 78, 216: 79, 43: 80, 212: 81, 44: 82, 42: 83, 214: 84, 211: 85, 213: 86, 215: 87, 41: 88, 46: 89, 40: 90, 45: 91, 210: 92}
        return classifier_dict
    
    assert len(filename_list) != 0, '文件名为空'
    classifier_dict = {}
    count = 1
    classifier_dict[0] = 0
    for filename in tqdm(filename_list):
        label = read_image(os.path.join(config.LABEL_DATA_PATH, filename + '-profile.jpg'))
        for x in range(label.shape[0]):
            for y in range(label.shape[1]):
                if label[x][y] not in classifier_dict:
                    classifier_dict[label[x][y]] = count
                    count = count + 1
    return classifier_dict

'''
根据classifier_dict得到相应的label，用于模型推理后得到语义分割图片
input:
    classifier_dict -> dict
output:
    label_dict -> dict
'''
def get_label(classifier_dict):
    label_dict = {}
    for key in classifier_dict:
        label_dict[classifier_dict[key]] = key
    return label_dict


# filename_list = get_filename_list()
# idx = 0
# image = read_image(os.path.join(config.TRAIN_DATA_PATH, filename_list[idx] + '.jpg'))
# label, palette = read_image(os.path.join(config.LABEL_DATA_PATH, filename_list[idx] + '-profile.jpg'), need_palette=1)

# show_image(image, label)

# classifier_dict = get_classifier(filename_list, c=0)
# print(classifier_dict)
# label_dict = get_label(classifier_dict)
# print(label_dict)

