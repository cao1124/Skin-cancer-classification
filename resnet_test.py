import os
import numpy as np
import torch
import torchvision
from cv2 import cv2
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
import torch.nn as nn
from PIL import Image
from prepare_data import SkinDisease, is_image_file, cv_imread, cv_write
from resnet_train import skin_mean, skin_std, SkinDataset
import warnings
warnings.filterwarnings("ignore")


class GradCAM:
    def __init__(self, image_path, model: nn.Module, target_layer: str, size=(224, 224), num_cls=1000, mean=None, std=None) -> None:
        self.model = model
        self.model.eval()
        self.image_path = image_path

        # register hook
        # 可以自己指定层名，没必要一定通过target_layer传递参数
        # self.model.layer4
        # self.model.layer4[1].register_forward_hook(self.__forward_hook)
        # self.model.layer4[1].register_backward_hook(self.__backward_hook)
        getattr(self.model, target_layer).register_forward_hook(self.__forward_hook)
        getattr(self.model, target_layer).register_backward_hook(self.__backward_hook)

        self.size = size
        self.origin_size = None
        self.num_cls = num_cls

        self.mean, self.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        if mean and std:
            self.mean, self.std = mean, std

        self.grads = []
        self.fmaps = []

    def forward(self, img_arr: np.ndarray, label=None, show=True, write=False):
        img_input = self.__img_preprocess(img_arr.copy())

        # forward
        output = self.model(img_input)
        idx = np.argmax(output.cpu().data.numpy())

        # backward
        self.model.zero_grad()
        loss = self.__compute_loss(output, label)

        loss.backward()

        # generate CAM
        grads_val = self.grads[0].cpu().data.numpy().squeeze()
        fmap = self.fmaps[0].cpu().data.numpy().squeeze()
        cam = self.__compute_cam(fmap, grads_val)

        # show
        cam_show = cv2.resize(cam, self.origin_size)
        img_show = img_arr.astype(np.float32) / 255
        self.__show_cam_on_image(img_show, cam_show, if_show=show, if_write=write)

        self.fmaps.clear()
        self.grads.clear()

    def __img_transform(self, img_arr: np.ndarray, transform: torchvision.transforms) -> torch.Tensor:
        img = img_arr.copy()  # [H, W, C]
        img = Image.fromarray(np.uint8(img))
        img = transform(img).unsqueeze(0)  # [N,C,H,W]
        return img

    def __img_preprocess(self, img_in: np.ndarray) -> torch.Tensor:
        self.origin_size = (img_in.shape[1], img_in.shape[0])  # [H, W, C]
        img = img_in.copy()
        img = cv2.resize(img, self.size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor(),
                                        transforms.Normalize(skin_mean, skin_std)])
        img_tensor = self.__img_transform(img, transform)
        return img_tensor

    def __backward_hook(self, module, grad_in, grad_out):
        self.grads.append(grad_out[0].detach())

    def __forward_hook(self, module, input, output):
        self.fmaps.append(output)

    def __compute_loss(self, logit, index=None):
        if not index:
            index = np.argmax(logit.cpu().data.numpy())
        else:
            index = np.array(index)

        index = index[np.newaxis, np.newaxis]
        index = torch.from_numpy(index)
        one_hot = torch.zeros(1, self.num_cls).scatter_(1, index, 1)
        one_hot.requires_grad = True
        loss = torch.sum(one_hot * logit)
        return loss

    def __compute_cam(self, feature_map, grads):
        """
        feature_map: np.array [C, H, W]
        grads: np.array, [C, H, W]
        return: np.array, [H, W]
        """
        cam = np.zeros(feature_map.shape[1:], dtype=np.float32)
        alpha = np.mean(grads, axis=(1, 2))  # GAP
        for k, ak in enumerate(alpha):
            cam += ak * feature_map[k]  # linear combination

        cam = np.maximum(cam, 0)  # relu
        cam = cv2.resize(cam, self.size)
        cam = (cam - np.min(cam)) / np.max(cam)
        return cam

    def __show_cam_on_image(self, img: np.ndarray, mask: np.ndarray, if_show=True, if_write=False):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        cam = np.uint8(255 * cam)
        if if_write:
            new_path = 'data/result/' + self.image_path.split('/')[-1]
            cv_write(new_path, cam)
        if if_show:
            # 要显示RGB的图片，如果是BGR的 热力图是反过来的
            plt.imshow(cam[:, :, ::-1])
            plt.show()


def predict_single_image(image_path, checkpoint_path):
    # img = cv_imread(image_path)

    # load model weights
    net = torch.load(checkpoint_path)
    # net = models.densenet121()
    # # net.fc = nn.Linear(in_features=2048, out_features=22, bias=True)
    # net.classifier = nn.Linear(in_features=1024, out_features=22, bias=True)
    # net.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage).module.state_dict())

    # grad_cam = GradCAM(image_path, net, 'layer4', (224, 224), 22, skin_mean, skin_std)
    # grad_cam.forward(img, show=True, write=False)

    # 数据增强
    image_transforms = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor(),
                                           transforms.Normalize(skin_mean, skin_std)])
    net.to(device)
    img = Image.open(image_path).convert('RGB')
    if transforms is not None:
        img = image_transforms(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    # prediction
    net.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(net(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
        print('image:{}, predict_cla:{}, prob:{}'.format(image_path, SkinDisease(int(predict_cla)), predict[predict_cla].numpy()))


def predict_images(image_dir, checkpoint):
    # load model weights
    net = torch.load(checkpoint_path)

    loss_func = nn.CrossEntropyLoss()
    # 数据增强
    image_transforms = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor(),
                                           transforms.Normalize(skin_mean, skin_std)])

    # dataloader
    dataset = SkinDataset(image_dir, image_dir + '/839.txt', image_transforms)
    n_val = int(len(dataset) * 0.2)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = random_split(dataset, lengths=[n_train, n_val], generator=torch.manual_seed(0))  # i
    # sklearn flod 五折交叉验证
    skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    for train_index, val_index in skf.split(dataset.img_path, dataset.labels):
        val_dataset.indices = list(val_index)

        bs = 8
        num_classes = 22
        valid_data = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=0)
        test_loss = 0.0
        test_acc = 0.0
        confusion_matrix = torch.zeros(num_classes, num_classes).cuda()
        with torch.no_grad():  # 用于通知dropout层和batchnorm层在train和val模式间切换。
            net.eval()
            for j, (inputs, labels) in enumerate(valid_data):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = net(inputs)
                loss = loss_func(outputs, labels)
                test_loss += loss.item()
                pred = torch.max(outputs, 1)[1]
                num_correct = (pred == labels).sum()
                test_acc += num_correct.item()

                for t, p in zip(labels.view(-1), outputs.argmax(dim=1).view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

        acc_per_class = confusion_matrix.diag() / confusion_matrix.sum(1)
        print(acc_per_class)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = 'data/saved/checkpoint/train_best_model-0.pt'
    # predict_images('D:/MAD_File/上海_皮肤病/上海_皮肤病/photo_img_merge/', checkpoint_path)

    # image_path = 'data/us_label_crop/D20191250 SCC.jpg'
    # predict_single_image(image_path, checkpoint_path)

    img_dir = 'data/us_label_crop/'
    images = [os.path.join(img_dir, x) for x in os.listdir(img_dir) if is_image_file(x)]
    for image_path in images:
        predict_single_image(image_path, checkpoint_path)
