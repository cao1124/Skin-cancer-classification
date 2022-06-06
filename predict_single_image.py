import os
import torch
from torch.optim import lr_scheduler
from torchvision import models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from tqdm import tqdm
from PIL import Image

from prepare_data import SkinDisease
from utils.get_log import _get_logger
import warnings
warnings.filterwarnings("ignore")


def predict(image_path, checkpoint_path):
    # 数据增强
    image_transforms = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor(),
                                           transforms.Normalize([0.283, 0.283, 0.288], [0.23, 0.23, 0.235])])

    img = Image.open(image_path).convert('RGB')
    if transforms is not None:
        img = image_transforms(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # load model weights
    model = models.resnet50()
    model.fc = nn.Linear(in_features=2048, out_features=22, bias=True)
    model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage).module.state_dict())

    # prediction
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
        print('image:{}, predict_cla:{}, prob:{}'.format(image_path, SkinDisease(int(predict_cla)), predict[predict_cla].numpy()))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_path = 'data/us_label_crop/D42419 BCC.jpg'
    checkpoint_path = 'data/saved/checkpoint/train_best_model.pt'
    predict(image_path, checkpoint_path)
