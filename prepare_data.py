# encoding:utf-8

import os
from enum import Enum
import numpy as np
from cv2 import cv2
import torch
from matplotlib import pyplot as plt
from torch.utils.data import random_split
from data_loader.inaturalist_data_loaders import LT_Dataset
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


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
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), flags=-1)
    # flag = -1,   8位深度，原通道
    # flag = 0，   8位深度，1通道
    # flag = 1，   8位深度，3通道
    # flag = 2，   原深度， 1通道
    # flag = 3，   原深度， 3通道
    # flag = 4，   8位深度，3通道
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
    img_dir = 'data/us_img_crop/'
    images = [os.path.join(img_dir, x) for x in os.listdir(img_dir) if is_image_file(x)]
    for img_path in images:
        print(img_path)
        # img = Image.open(img).convert('RGB')
        img = cv_imread(img_path)

        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        # ultrasound_preprocess_data(img_path, img)
        object_detect(img_path, img)

        # Otsu threshold
        # T1, T2, epsT, histCV = doubleThreshold(img)
        # print("T1={}, T2={}, esp={:.4f}".format(T1, T2, epsT))
        #
        # binary = img.copy()
        # binary[binary < T1] = 0
        # binary[binary > T2] = 255
        #
        # ret, imgOtsu = cv2.threshold(img, 127, 255, cv2.THRESH_OTSU)  # OTSU 阈值分割
        # ret1, binary1 = cv2.threshold(img, T1, 255, cv2.THRESH_TOZERO)  # 小于阈值置 0，大于阈值不变
        # ret2, binary2 = cv2.threshold(img, T2, 255, cv2.THRESH_TOZERO)
        #
        # plt.figure(figsize=(9, 6))
        # plt.subplot(231), plt.axis('off'), plt.title("Origin"), plt.imshow(img, 'gray')
        # plt.subplot(232, yticks=[]), plt.axis([0, 255, 0, np.max(histCV)])
        # plt.bar(range(256), histCV[:, 0]), plt.title("Gray Hist")
        # plt.subplot(233), plt.title("OTSU binary(T={})".format(round(ret))), plt.axis('off')
        # plt.imshow(imgOtsu, 'gray')
        # plt.subplot(234), plt.axis('off'), plt.title("Threshold(T={})".format(T1))
        # plt.imshow(binary1, 'gray')
        # plt.subplot(235), plt.axis('off'), plt.title("Threshold(T={})".format(T2))
        # plt.imshow(binary2, 'gray')
        # plt.subplot(236), plt.axis('off'), plt.title("DoubleT({},{})".format(T1, T2))
        # plt.imshow(binary, 'gray')
        # plt.show()


def ultrasound_preprocess_data(img_path, img):
    # RGB2GRAY
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # bilateralFilter 双边滤波 对边缘信息较好的保留
    filter_img = cv2.bilateralFilter(img_gray, 9, 75, 75)
    # equalize hist 直方图均衡化
    equal_img = cv2.equalizeHist(filter_img)
    # binary 二值化
    blur = cv2.GaussianBlur(equal_img, (5, 5), 0)
    ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # convert back to RGB form
    revert = cv2.cvtColor(th, cv2.COLOR_GRAY2RGB)
    # cv2.imshow('revert', revert)
    # cv2.waitKey(0)
    new_path = 'data/us_img_crop_process/' + img_path.split('/')[2]
    cv_write(new_path, revert)


def doubleThreshold(img):
    # 多阈值 OTSU
    histCV = cv2.calcHist([img], [0], None, [256], [0, 256])  # 灰度直方图
    grayScale = np.arange(0, 256, 1)  # 灰度级 [0,255]
    totalPixels = img.shape[0] * img.shape[1]  # 像素总数
    totalGray = np.dot(histCV[:,0], grayScale)  # 内积, 总和灰度值
    mG = totalGray / totalPixels  # 平均灰度，meanGray
    varG = sum(((i-mG)**2 * histCV[i,0]/totalPixels) for i in range(256))

    T1, T2, varMax = 1, 2, 0.0
    # minGary, maxGray = np.min(img), np.max(img)  # 最小灰度，最大灰度
    for k1 in range(1, 254):  # k1: [1,253], 1<=k1<k2<=254
        n1 = sum(histCV[:k1, 0])  # C1 像素数量
        s1 = sum((i * histCV[i, 0]) for i in range(k1))
        P1 = n1 / totalPixels  # C1 像素数占比
        m1 = (s1 / n1) if n1 > 0 else 0  # C1 平均灰度

        for k2 in range(k1+1, 256):  # k2: [2,254], k2>k1
            # n2 = sum(histCV[k1+1:k2,0])  # C2 像素数量
            # s2 = sum( (i * histCV[i,0]) for i in range(k1+1,k2) )
            # P2 = n2 / totalPixels  # C2 像素数占比
            # m2 = (s2/n2) if n2>0 else 0  # C2 平均灰度
            n3 = sum(histCV[k2+1:,0])  # C3 像素数量
            s3 = sum((i*histCV[i,0]) for i in range(k2+1,256))
            P3 = n3 / totalPixels  # C3 像素数占比
            m3 = (s3/n3) if n3>0 else 0  # C3 平均灰度

            P2 = 1.0 - P1 - P3  # C2 像素数占比
            m2 = (mG - P1*m1 - P3*m3)/P2 if P2>1e-6 else 0  # C2 平均灰度

            var = P1*(m1-mG)**2 + P2*(m2-mG)**2 + P3*(m3-mG)**2
            if var > varMax:
                T1, T2, varMax = k1, k2, var
    epsT = varMax / varG  # 可分离测度
    print(totalPixels, mG, varG, varMax, epsT, T1, T2)
    return T1, T2, epsT, histCV


if __name__ == '__main__':
    # split_data()
    prepare_img()

