import os
from enum import Enum
import pandas as pd
import tablib

from prepare_data import is_image_file


class SkinDisease(Enum):
    OB = 0              # 其他良性 other benign
    BNT = 1             # 神经源性肿瘤 Benign Neurogenic tumors
    BFT = 2             # 良性毛囊肿瘤 Benign follicular tumor
    BSeb = 3            # 良性皮脂腺肿瘤   Benign sebaceous gland tumor
    BKLL = 4            # 良性角化病样病变 Benign keratosis like lesions
    BFMY = 5            # 良性纤维母细胞和肌纤维母细胞肿瘤  Benign fibroblastic and myofibroblastic tumors
    BSwe = 6            # 良性汗腺肿瘤  Benign sweat gland tumor
    Hema = 7            # 血管瘤 Hemangioma
    Cyst = 8            # 囊肿 cyst
    Infl = 9            # 炎症 inflammation
    Wart = 10           # 疣  wart
    Lipo = 11           # 脂肪瘤 lipoma
    Nevu = 12           # 痣 nevus

    OM = 13             # 其他恶性 Other malignancies
    BD = 14
    AK = 15
    MM = 16
    SCC = 17
    BCC = 18
    Aden = 19           # 腺癌 Adenocarcinoma
    DSFP = 20           # 隆突性皮肤纤维肉瘤  Dermatofibrosarcoma protuberans
    Paget = 21


def rename_file():
    img_dir = 'D:/MAD_File/上海_皮肤病/上海_皮肤病/us_img/'
    # images = [os.path.join(img_dir, x) for x in os.listdir(img_dir) if is_image_file(x)]
    images = os.listdir(img_dir)
    for name in images:
        new_name = name.replace('_', '.')
        os.rename(os.path.join(img_dir, name), os.path.join(img_dir, new_name))


def multi_class(disease, describe, two_class):
    if two_class == 0:
        if '良性皮脂腺肿瘤' in disease or '皮脂腺痣' in disease:
            result = '3, 良性皮脂腺肿瘤'
        elif '脂肪瘤' in disease:
            result = '11, 脂肪瘤'
        elif '疣' in disease:
            result = '10, 疣'
        elif '炎症' in disease:
            result = '9, 炎症'
        elif '囊肿' in disease:
            result = '8, 囊肿'
        elif '血管瘤' in disease or '化脓性肉芽肿' in disease:
            result = '7, 血管瘤'
        elif '良性汗腺肿瘤' in disease:
            result = '6, 良性汗腺肿瘤'
        elif '良性纤维母细胞' in disease or '肌纤维母细胞肿瘤' in disease or '瘢痕' in disease \
                or '皮肤纤维瘤' in disease or describe == '血管平滑肌瘤':
            result = '5, 良性纤维母细胞和肌纤维母细胞肿瘤'
        elif '良性角化病样病变' in disease:
            result = '4, 良性角化病样病变'
        elif '痣' in disease:
            result = '12, 痣'
        elif '良性毛囊肿瘤' in disease or '毛母质瘤' in disease:
            result = '2, 良性毛囊肿瘤'
        elif '神经源性肿瘤' in disease:
            result = '1, 神经源性肿瘤'
        else:
            result = '0, 其他良性'
    elif two_class == 1:
        if 'Paget' in disease:
            result = '21, Paget'
        elif '隆突性皮肤纤维肉瘤' in disease:
            result = '20, 隆突性皮肤纤维肉瘤'
        elif '腺癌' in disease:
            result = '19, 腺癌'
        elif 'BCC' in disease:
            result = '18, BCC'
        elif 'SCC' in disease:
            result = '17, SCC'
        elif 'MM' in disease:
            result = '16, MM'
        elif 'AK' in disease:
            result = '15, AK'
        elif 'BD' in disease:
            result = '14, BD'
        else:
            result = '13, 其他恶性'
    return result


def data_prepare():
    dataset = tablib.Dataset()
    dataset.headers = ['id', 'class', 'disease']
    # img_dir = 'D:/MAD_File/上海_皮肤病/上海_皮肤病/us_img_2class/'
    # images = os.listdir(img_dir)
    excel_path = 'D:/MAD_File/上海_皮肤病/上海_皮肤病/训练组1361例.xlsx'
    excel_data = pd.read_excel(excel_path)
    num_list = excel_data.病原号.tolist()
    benign_malignant = excel_data.良恶性.tolist()
    disease_list = excel_data.病理分组.tolist()
    describe_list = excel_data.分组描述.tolist()
    for i in range(len(num_list)):
        img_id = num_list[i]
        img_class = benign_malignant[i]
        disease = disease_list[i]
        describe = describe_list[i]
        # img_class = int(img.split('_')[-1].split('.')[0])
        # print(img_class)
        if img_class == 0:          # 良性
            if '痣' in disease:
                dataset.append([img_id, 12, '痣'])
            elif '脂肪瘤' in disease:
                dataset.append([img_id, 11, '脂肪瘤'])
            elif '疣' in disease:
                dataset.append([img_id, 10, '疣'])
            elif '炎症' in disease:
                dataset.append([img_id, 9, '炎症'])
            elif '囊肿' in disease:
                dataset.append([img_id, 8, '囊肿'])
            elif '血管瘤' in disease or '化脓性肉芽肿' in disease:
                dataset.append([img_id, 7, '血管瘤'])
            elif '良性汗腺肿瘤' in disease:
                dataset.append([img_id, 6, '良性汗腺肿瘤'])
            elif '良性纤维母细胞' in disease or '肌纤维母细胞肿瘤' in disease or '瘢痕' in disease \
                    or '皮肤纤维瘤' in disease or describe == '血管平滑肌瘤':
                dataset.append([img_id, 5, '良性纤维母细胞和肌纤维母细胞肿瘤'])
            elif '良性角化病样病变' in disease:
                dataset.append([img_id, 4, '良性角化病样病变'])
            elif '良性皮脂腺肿瘤' in disease or '皮脂腺痣' in disease:
                dataset.append([img_id, 3, '良性皮脂腺肿瘤'])
            elif '良性毛囊肿瘤' in disease or '毛母质瘤' in disease:
                dataset.append([img_id, 2, '良性毛囊肿瘤'])
            elif '神经源性肿瘤' in disease:
                dataset.append([img_id, 1, '神经源性肿瘤'])
            else:
                dataset.append([img_id, 0, '其他良性'])
        elif img_class == 1:  # 恶性
            if 'Paget' in disease:
                dataset.append([img_id, 21, 'Paget'])
            elif '隆突性皮肤纤维肉瘤' in disease:
                dataset.append([img_id, 20, '隆突性皮肤纤维肉瘤'])
            elif '腺癌' in disease:
                dataset.append([img_id, 19, '腺癌'])
            elif 'BCC' in disease:
                dataset.append([img_id, 18, 'BCC'])
            elif 'SCC' in disease:
                dataset.append([img_id, 17, 'SCC'])
            elif 'MM' in disease:
                dataset.append([img_id, 16, 'MM'])
            elif 'AK' in disease:
                dataset.append([img_id, 15, 'AK'])
            elif 'BD' in disease:
                dataset.append([img_id, 14, 'BD'])
            else:
                dataset.append([img_id, 13, '其他恶性'])
    with open('D:/MAD_File/上海_皮肤病/上海_皮肤病/multi_class.csv', mode='w', encoding='UTF-8') as f:
        f.write(dataset.csv)


def data_prepare2():
    dataset = tablib.Dataset()
    dataset.headers = ['id', 'class', 'disease']
    excel_path = 'D:/MAD_File/上海_皮肤病/上海_皮肤病/1326_multi_class.xlsx'
    excel_data = pd.read_excel(excel_path)
    id_list = excel_data.id.tolist()
    class_list = excel_data.multi_class.tolist()
    for i in range(len(id_list)):
        img_class = class_list[i]
        disease = id_list[i]
        # img_class = int(img.split('_')[-1].split('.')[0])
        # print(img_class)
        if img_class < 13:  # 良性
            if '痣' in disease:
                dataset.append([disease, 12, '痣'])
            elif '脂肪瘤' in disease:
                dataset.append([disease, 11, '脂肪瘤'])
            elif '疣' in disease:
                dataset.append([disease, 10, '疣'])
            elif '炎症' in disease:
                dataset.append([disease, 9, '炎症'])
            elif '囊肿' in disease:
                dataset.append([disease, 8, '囊肿'])
            elif '血管瘤' in disease or '化脓性肉芽肿' in disease:
                dataset.append([disease, 7, '血管瘤'])
            elif '良性汗腺肿瘤' in disease:
                dataset.append([disease, 6, '良性汗腺肿瘤'])
            elif '良性纤维母细胞' in disease or '肌纤维母细胞肿瘤' in disease or '瘢痕' in disease \
                    or '皮肤纤维瘤' in disease or '血管平滑肌瘤' in disease:
                dataset.append([disease, 5, '良性纤维母细胞和肌纤维母细胞肿瘤'])
            elif '良性角化病样病变' in disease:
                dataset.append([disease, 4, '良性角化病样病变'])
            elif '良性皮脂腺肿瘤' in disease or '皮脂腺痣' in disease:
                dataset.append([disease, 3, '良性皮脂腺肿瘤'])
            elif '良性毛囊肿瘤' in disease or '毛母质瘤' in disease:
                dataset.append([disease, 2, '良性毛囊肿瘤'])
            elif '神经源性肿瘤' in disease:
                dataset.append([disease, 1, '神经源性肿瘤'])
            else:
                dataset.append([disease, 0, '其他良性'])
        else:  # 恶性
            if 'Paget' in disease:
                dataset.append([disease, 21, 'Paget'])
            elif '隆突性皮肤纤维肉瘤' in disease:
                dataset.append([disease, 20, '隆突性皮肤纤维肉瘤'])
            elif '腺癌' in disease:
                dataset.append([disease, 19, '腺癌'])
            elif 'BCC' in disease:
                dataset.append([disease, 18, 'BCC'])
            elif 'SCC' in disease:
                dataset.append([disease, 17, 'SCC'])
            elif 'MM' in disease:
                dataset.append([disease, 16, 'MM'])
            elif 'AK' in disease:
                dataset.append([disease, 15, 'AK'])
            elif 'BD' in disease:
                dataset.append([disease, 14, 'BD'])
            else:
                dataset.append([disease, 13, '其他恶性'])
    with open('D:/MAD_File/上海_皮肤病/上海_皮肤病/multi_class.csv', mode='w', encoding='UTF-8') as f:
        f.write(dataset.csv)


if __name__ == '__main__':
    excel_path = 'D:/MAD_File/上海_皮肤病/上海_皮肤病/训练组1361例.xlsx'
    excel_data = pd.read_excel(excel_path)
    id_list = excel_data.病原号.tolist()
    multi_class_list = excel_data.病理分组.tolist()
    class_describe_list = excel_data.分组描述.tolist()
    two_class_list = excel_data.良恶性.tolist()
    filename = 'data_process.txt'
    with open(filename, 'w') as file:
        img_dir = 'D:/MAD_File/上海_皮肤病/上海_皮肤病/us_label_crop/'
        images = [x for x in os.listdir(img_dir) if is_image_file(x)]
        for img in images:
            img_split = int(''.join(list(filter(str.isdigit, img))))
            if img_split in id_list:
                img_idx = id_list.index(img_split)
                result = multi_class(multi_class_list[img_idx], class_describe_list[img_idx], two_class_list[img_idx])
                file.write(os.path.join(img_dir, img))
                file.write(',' + result + '\n')
            else:
                print(img_split)
    # rename_file()




