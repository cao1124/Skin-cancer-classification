# -*- coding: UTF-8 -*-
# https://blog.csdn.net/weixin_40437821/article/details/103603700
import cv2
import os
import numpy as np

DATA_LEN = 3072  # 32x32x3=3072
# DATA_LEN = 43200     # 160x90x3
CHANNEL_LEN = 1024  # 32x32=1024
# CHANNEL_LEN = 14400  # 160x90 = 14400
SHAPE = (32, 32)  # (160, 90)#32

# 修改
# figure_path = '/home/user/PycharmProjects/DataSet_ipanel/Image2Dataset/layoutdata-160-90/train/video'
# figure_name_label = '/home/user/PycharmProjects/DataSet_ipanel/Image2Dataset/layoutdata-160-90/figure_name_label_train/image_train_video_list.txt'
# batch_save = '/home/user/PycharmProjects/DataSet_ipanel/Image2Dataset/layoutdata-160-90/batch_save_train'
## 修改imagelist()标签值

figure_path = 'D:/PycharmProjects/skin-disease-classification-by-ride/data/us_label_mask1/'  # 图片的位置
figure_name_label = 'D:/PycharmProjects/skin-disease-classification-by-ride/data/us_label_mask1/1342cifar.txt'  # 保存图片名称和标签
batch_save = './data/batch_save_train'  # 保存batch文件


## 修改imagelist()标签值

def imread(im_path, shape=None, color="RGB", mode=cv2.IMREAD_UNCHANGED):
    im = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
    if color == "RGB":
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if shape != None:
        # assert isinstance(shape, int)
        # im = cv2.resize(im, (shape, shape))
        im = cv2.resize(im, shape)
    return im


def read_data(filename, data_path, shape=None, color='RGB'):
    """
       filename (str): a file
         data file is stored in such format:
           image_name  label
       data_path (str): image data folder
       return (numpy): a array of image and a array of label
    """
    (shape1, shape2) = shape
    if os.path.isdir(filename):
        print("Can't found data file!")
    else:
        f = open(filename)
        lines = f.read().splitlines()
        count = len(lines)
        data = np.zeros((count, DATA_LEN), dtype=np.uint8)
        # label = np.zeros(count, dtype=np.uint8)
        lst = [ln.split(' ')[0] for ln in lines]
        label = [int(ln.split(' ')[1]) for ln in lines]

        idx = 0
        # s, c = SHAPE, CHANNEL_LEN
        c = CHANNEL_LEN
        for ln in lines:
            fname, lab = ln.split(' ')
            # im = imread(os.path.join(data_path, fname), shape=s, color='RGB')
            im = imread(os.path.join(data_path, fname), shape=SHAPE, color='RGB')
            '''
            im = cv2.imread(os.path.join(data_path, fname), cv2.IMREAD_UNCHANGED)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, (s, s))
            '''
            data[idx, :c] = np.reshape(im[:, :, 0], c)
            data[idx, c:2 * c] = np.reshape(im[:, :, 1], c)
            data[idx, 2 * c:] = np.reshape(im[:, :, 2], c)
            label[idx] = int(lab)
            idx = idx + 1

        return data, label, lst


def py2bin(data, label):
    label_arr = np.array(label).reshape(len(label), 1)
    label_uint8 = label_arr.astype(np.uint8)
    arr = np.hstack((label_uint8, data))

    with open(batch_save, 'wb') as f:  # 每个文件夹修改
        # with open('/home/user/PycharmProjects/DataSet_ipanel/layoutdata-160-90/train/train_batch/train_batch_big5small5', 'wb') as f:
        for element in arr.flat:
            f.write(element)


import pickle

BIN_COUNTS = 1  # 每一类的数据为一个batch


def pickled(savepath, data, label, fnames, bin_num=BIN_COUNTS, mode="train", name=None):
    '''
      savepath (str): save path
      data (array): image data, a nx3072 array
      label (list): image label, a list with length n
      fnames (str list): image names, a list with length n
      bin_num (int): save data in several files
      mode (str): {'train', 'test'}
    '''
    assert os.path.isdir(savepath)
    total_num = len(fnames)
    samples_per_bin = total_num // bin_num  # 将/换为// （TypeError: slice indices must be integers or None or have an __index__ method）
    assert samples_per_bin > 0
    idx = 0
    for i in range(bin_num):
        start = i * samples_per_bin
        end = (i + 1) * samples_per_bin

        if end <= total_num:
            dict = {'data': data[start:end, :],
                    'labels': label[start:end],
                    'filenames': fnames[start:end]}
        else:
            dict = {'data': data[start:, :],
                    'labels': label[start:],
                    'filenames': fnames[start:]}
        if mode == "train":
            dict['batch_label'] = "training batch {}".format(name)  # (idx, bin_num)
        else:
            dict['batch_label'] = "testing batch {}".format(name)  # (idx, bin_num)

        with open(os.path.join(savepath, 'data_batch_' + str(name)), 'wb') as fi:  # str(idx)), 'wb') as fi:
            # cPickle.dump(dict, fi)
            pickle.dump(dict, fi)
        # idx = idx + 1


def imagelist():
    directory_normal = figure_path
    # directory_normal = r"/home/user/PycharmProjects/DataSet_ipanel/layoutdata-160-90/train/big5small6"  # 原始图片位置，32*32 pixel
    file_train_list = figure_name_label
    # file_train_list = r"/home/user/PycharmProjects/DataSet_ipanel/layoutdata-160-90/train/image_train_big5small6_list.txt"  # 构建imagelist输出位置
    with open(file_train_list, "a") as f:
        for filename in os.listdir(directory_normal):
            # f.write(filename + " " + "0" + "\n")    #这里分类默认全为0
            f.write(filename + " " + "0" + "\n")  # 这里分类默认全为0 ##########


if __name__ == '__main__':
    data_path = figure_path
    # data_path = '/home/user/PycharmProjects/DataSet_ipanel/layoutdata-160-90/train/big5small6'
    file_list = figure_name_label
    # file_list = '/home/user/PycharmProjects/DataSet_ipanel/layoutdata-160-90/train/image_train_big5small6_list.txt'
    save_path = batch_save  # './bin'
    imagelist()  # 构建imagelist  # 生成名字和标签的对应关系
    data, label, lst = read_data(file_list, data_path, shape=SHAPE)  # 将图片像素数据转成矩阵和标签列表
    # py2bin(data, label) #将像素矩阵和标签列表转成cifar10 binary version # 二进制版本
    pickled(save_path, data, label, lst, bin_num=1, name='airbus')  # 生成python版本



