import numpy as np
from PIL import Image
import torchvision
from tqdm import tqdm


def compute_mean_and_std(dataset):
    # 输入PyTorch的dataset，输出均值和标准差
    mean_r = 0
    mean_g = 0
    mean_b = 0
    print("计算均值>>>")
    for img_path, _ in tqdm(dataset, ncols=80):
      img=Image.open(img_path)
      img = np.asarray(img) # change PIL Image to numpy array
      mean_b += np.mean(img[:, :, 0])
      mean_g += np.mean(img[:, :, 1])
      mean_r += np.mean(img[:, :, 2])

    mean_b /= len(dataset)
    mean_g /= len(dataset)
    mean_r /= len(dataset)

    diff_r = 0
    diff_g = 0
    diff_b = 0

    N = 0
    print("计算方差>>>")
    for img_path, _ in tqdm(dataset,ncols=80):
      img=Image.open(img_path)
      img = np.asarray(img)
      diff_b += np.sum(np.power(img[:, :, 0] - mean_b, 2))
      diff_g += np.sum(np.power(img[:, :, 1] - mean_g, 2))
      diff_r += np.sum(np.power(img[:, :, 2] - mean_r, 2))

      N += np.prod(img[:, :, 0].shape)

    std_b = np.sqrt(diff_b / np.abs(N))
    std_g = np.sqrt(diff_g / np.abs(N))
    std_r = np.sqrt(diff_r / np.abs(N))

    mean = (round(mean_b.item() / 255.0, 3), round(mean_g.item() / 255.0, 3), round(mean_r.item() / 255.0, 3))
    std = (round(std_b.item() / 255.0, 3), round(std_g.item() / 255.0, 3), round(std_r.item() / 255.0, 3))
    return mean, std


if __name__ == '__main__':
    path = "D:/MAD_File/上海_皮肤病/us_skin_crop/"
    train_data = torchvision.datasets.ImageFolder(path)
    train_mean, train_std = compute_mean_and_std(train_data.imgs)
    print(train_mean, train_std)
