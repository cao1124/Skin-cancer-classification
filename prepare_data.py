# encoding:utf-8

import os
from enum import Enum
import numpy as np
from cv2 import cv2
import torch
from torch.utils.data import random_split
from data_loader.inaturalist_data_loaders import LT_Dataset


def split_data():
    dataset = LT_Dataset('data/us_img_crop/', 'data/us_img_crop/data1326.txt')
    n_val = int(len(dataset) * 0.2)
    n_train = len(dataset) - n_val
    train_list, val_list = random_split(dataset, lengths=[n_train, n_val],
                                              generator=torch.Generator().manual_seed(0))
    train_dataset = []
    for i in list(train_list.indices):
        train_dataset.append(dataset[i])
    print('.')


def split_five():
    dataset = LT_Dataset('data/us_img_crop/', 'data/us_img_crop/data1326.txt')
    data1, data2, data3, data4, data5 = random_split(dataset, lengths=[int(len(dataset.img_path) * 0.2) + 1,
                                                                       int(len(dataset.img_path) * 0.2),
                                                                       int(len(dataset.img_path) * 0.2),
                                                                       int(len(dataset.img_path) * 0.2),
                                                                       int(len(dataset.img_path) * 0.2)],
                                                     generator=torch.Generator().manual_seed(0))
    with open('data1.txt', 'w') as f1:
        for i in list(data1.indices):
            s = dataset.img_path[i] + ',' + str(dataset.labels[i]) + '\n'
            f1.write(s)
    with open('data2.txt', 'w') as f2:
        for i in list(data2.indices):
            s = dataset.img_path[i] + ',' + str(dataset.labels[i]) + '\n'
            f2.write(s)
    with open('data3.txt', 'w') as f3:
        for i in list(data1.indices):
            s = dataset.img_path[i] + ',' + str(dataset.labels[i]) + '\n'
            f3.write(s)
    with open('data4.txt', 'w') as f4:
        for i in list(data1.indices):
            s = dataset.img_path[i] + ',' + str(dataset.labels[i]) + '\n'
            f4.write(s)
    with open('data5.txt', 'w') as f5:
        for i in list(data1.indices):
            s = dataset.img_path[i] + ',' + str(dataset.labels[i]) + '\n'
            f5.write(s)
    print('done')


class SkinDisease(Enum):
    OB = 0  # 其他良性 other benign
    BNT = 1  # 神经源性肿瘤 Benign Neurogenic tumors
    BFT = 2  # 良性毛囊肿瘤 Benign follicular tumor
    BSeb = 3  # 良性皮脂腺肿瘤   Benign sebaceous gland tumor
    BKLL = 4  # 良性角化病样病变 Benign keratosis like lesions
    BFMY = 5  # 良性纤维母细胞和肌纤维母细胞肿瘤  Benign fibroblastic and myofibroblastic tumors
    BSwe = 6  # 良性汗腺肿瘤  Benign sweat gland tumor
    Hema = 7  # 血管瘤 Hemangioma
    Cyst = 8  # 囊肿 cyst
    Infl = 9  # 炎症 inflammation
    Wart = 10  # 疣  wart
    Lipo = 11  # 脂肪瘤 lipoma
    Nevu = 12  # 痣 nevus

    OM = 13  # 其他恶性 Other malignancies
    BD = 14
    AK = 15
    MM = 16
    SCC = 17
    BCC = 18
    Aden = 19  # 腺癌 Adenocarcinoma
    DSFP = 20  # 隆突性皮肤纤维肉瘤  Dermatofibrosarcoma protuberans
    Paget = 21


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.PNG', '.jpg', '.jpeg', '.JPG', '.JPEG', '.bmp',
                                                              '.BMP', '.tiff', '.TIFF'])


def cv_imread(file_path):
    # 可读取图片（路径为中文）
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


def cv_write(file_path, file):
    cv2.imencode('.bmp', file)[1].tofile(file_path)


def object_detect(img_path, image):
    # 灰度图
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 二值化
    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    # 查找物体轮廓
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for j, cnts in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnts)  # 计算点集最外面的矩形边界
        if w * h > 12000 and h > 300 and w > 300:
            crop_img = image[y:y + h, x:x + w]
            new_path = 'data/us_img_crop/' + img_path.split('/')[2]
            cv_write(new_path, crop_img)

            # 在原图上画出最大的矩形
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            # cv2.imshow('image', image)
            # cv2.waitKey(500)


def prepare_img():
    img_dir = 'data/us_img/'
    images = [os.path.join(img_dir, x) for x in os.listdir(img_dir) if is_image_file(x)]
    for img_path in images:
        print(img_path)
        # img = Image.open(img).convert('RGB')
        img = cv_imread(img_path)
        object_detect(img_path, img)


if __name__ == '__main__':
    split_data()
    # prepare_img()
