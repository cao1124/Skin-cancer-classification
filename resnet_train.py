import collections
import os

import numpy as np
import torch
from torch.optim import lr_scheduler
from torchvision import models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset, WeightedRandomSampler
from tqdm import tqdm
from PIL import Image

from torchsampler import ImbalancedDatasetSampler
from utils.get_log import _get_logger
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore")
logger = _get_logger('data/saved/log/resnet18-five.txt', 'info')
skin_mean, skin_std = [0.016, 0.016, 0.017], [0.094, 0.094, 0.097]  # 1342张 expand images
# [0.321, 0.321, 0.327], [0.222, 0.222, 0.226]  # 1342张 us_label_mask1
# skin_mean, skin_std = [0.526, 0.439, 0.393], [0.189, 0.183, 0.177]  # 839张 photo_img_merge


class SkinDataset(Dataset):
    def __init__(self, root, txt, transforms=None):
        self.img_path = []
        self.labels = []
        self.transforms = transforms
        with open(txt, 'r', encoding='gb2312') as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split(',')[0]))
                self.labels.append(int(line.split(',')[1]))

    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample, label

    def __len__(self):
        return len(self.labels)


def prepare_model(epochs):
    # 迁移学习  这里使用ResNet-50的预训练模型。
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(in_features=2048, out_features=22, bias=True)
    # model.classifier[2] = nn.Linear(in_features=1536, out_features=22, bias=True)  # convnext_large
    # model.fc = nn.Sequential(OrderedDict([('fc1', nn.Linear(2048, 128)),
    #                                       ('relu1', nn.ReLU()),
    #                                       ('dropout1', nn.Dropout(1)),
    #                                       ('fc2', nn.Linear(128, 22)),
    #                                       ('output', nn.Softmax(dim=1))]))
    # for  vit_b_16 vit_l_16
    # model.heads = nn.Sequential(OrderedDict([('head', nn.Linear(in_features=1024, out_features=22, bias=True))]))
    # model = densenet264(pretrained=True)
    # model.out = paddle.nn.Linear(in_features=2688, out_features=22)

    # resnet.fc = nn.Linear(in_features=2048, out_features=22, bias=True)
    # renet18 resnet34
    # (fc): nn.Linear(in_features=512, out_features=1000, bias=True)
    # resnet50 resnet101  resnet152 resnext50_32x4d wide_resnet50_2  resnext101_32x8d  wide_resnet101_2
    # (fc): nn.Linear(in_features=2048, out_features=22, bias=True)
    # densenet 121  in_features=1024
    # densenet 161  in_features=2208
    # densenet 169  in_features=1664
    # densenet 201  in_features=1920
    # (classifier): Linear(in_features=1024, out_features=22, bias=True)
    # inception
    # v3 (fc): Linear(in_features=2048, out_features=1000, bias=True)
    # def inception_v4(classes=22):
    #     return Inception("v4", classes)
    # model = inception_v4()

    # 多GPU
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    # 定义损失函数和优化器。
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=2e-4, momentum=0.9, nesterov=True)
    # optimizer = optim.NAdam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=2e-4)

    # 定义学习率与轮数关系的函数
    # lambda1 = lambda epoch: 0.95 ** epoch  # 学习率 = 0.95**(轮数)
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)
    # 在指定的epoch值，如[10,30,50,70,90]处对学习率进行衰减，lr = lr * gamma
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80], gamma=0.1)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs/2, eta_min=0.005)
    # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
    return model, optimizer, scheduler, loss_func


def train_and_valid(data_dir, epochs=25):
    # 数据增强
    image_transforms = {
        'train': transforms.Compose([
            # transforms.Resize([224, 224]),   # inception v3 resize change 224 to 299
            # Cutout(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(90),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 0.3), value=0, inplace=False),
            transforms.Normalize(skin_mean, skin_std)]),
        'valid': transforms.Compose([
            # transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(skin_mean, skin_std)
        ])
    }

    # DataLoader
    dataset = SkinDataset(data_dir, data_dir + '1342data.txt', image_transforms['train'])

    # random split dataset 五折交叉验证 # seed_list = [5, 4, 3, 2, 1] for i in seed_list：
    train_dataset, val_dataset = random_split(dataset, lengths=[len(dataset) - int(len(dataset) * 0.2),
                                                                int(len(dataset) * 0.2)], generator=torch.manual_seed(0))  # i
    bs = 8
    # sklearn flod 五折交叉验证
    skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    i = 0
    for train_index, val_index in skf.split(dataset.img_path, dataset.labels):
        logger.info('第{}次实验:'.format(i))
        train_dataset.indices = list(train_index)
        val_dataset.indices = list(val_index)

        # train_labels = [train_dataset.dataset.labels[i] for i in train_dataset.indices]
        # # WeightedRandomSampler
        # class_sample_counts = [i[1] for i in sorted(collections.Counter(train_labels).items(), key=lambda x: x[0], reverse=False)]
        # weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
        # samples_weights = weights[train_labels]
        # sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)

        train_data_size = len(train_dataset.indices)
        valid_data_size = len(val_dataset.indices)

        logger.info('batch size = {}:'.format(bs))
        model, optimizer, scheduler, loss_function = prepare_model(epochs)
        train_data = DataLoader(train_dataset, batch_size=bs, sampler=ImbalancedDatasetSampler(train_dataset))
        valid_data = DataLoader(val_dataset, batch_size=bs)
        logger.info('train_data_size:{}, valid_data_size:{}'.format(train_data_size, valid_data_size))
        history = []
        best_val_acc = 0.0
        best_tran_acc = 0.0
        best_epoch = 0

        for epoch in range(epochs):
            logger.info("Epoch: {}/{}".format(epoch + 1, epochs))

            model.train()

            train_loss = 0.0
            train_acc = 0.0
            valid_loss = 0.0
            valid_acc = 0.0
            alpha = 0.5

            for step, data in enumerate(tqdm(train_data)):
                inputs = data[0].to(device)
                labels = data[1].to(device)

                # inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha)        # mixup
                # outputs = outputs.logits                                                   # inception-v3 TypeError
                # loss = mixup_criterion(loss_function, outputs, labels_a, labels_b, lam)    # mixup
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                train_loss += loss.item()
                pred = torch.max(outputs, 1)[1]
                train_correct = (pred == labels).sum()
                train_acc += train_correct.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()  # 需要在优化器参数更新之后再动态调整学习率

            confusion_matrix = torch.zeros(22, 22).cuda()
            with torch.no_grad():
                model.eval()

                for j, (inputs, labels) in enumerate(tqdm(valid_data)):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels)
                    valid_loss += loss.item()
                    pred = torch.max(outputs, 1)[1]
                    num_correct = (pred == labels).sum()
                    valid_acc += num_correct.item()

                    for t, p in zip(labels.view(-1), outputs.argmax(dim=1).view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1
            acc_per_class = confusion_matrix.diag() / confusion_matrix.sum(1)
            avg_train_loss = train_loss / train_data_size
            avg_train_acc = train_acc / train_data_size

            avg_valid_loss = valid_loss / valid_data_size
            avg_valid_acc = valid_acc / valid_data_size

            history.append([avg_train_loss, avg_valid_loss,
                            avg_train_acc, avg_valid_acc])

            if best_tran_acc < avg_train_acc:
                best_tran_acc = avg_train_acc

            if best_val_acc < avg_valid_acc:
                best_val_acc = avg_valid_acc
                best_epoch = epoch + 1
                torch.save(model, 'data/saved/checkpoint/train_best_model-' + str(i) + '.pt') # + '-' + str(best_epoch)
                logger.info("Best acc per class:：{}".format(acc_per_class))

            logger.info("Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%".format(
                    epoch + 1, avg_valid_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100))
            logger.info("Best Accuracy for train : {:.4f}".format(best_tran_acc))
            logger.info("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_val_acc, best_epoch))
        i += 1
    # return model, history


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_epochs = 100
    data_dir = '/home/ai1000/project/data/expand_images/'
    train_and_valid(data_dir, num_epochs)

    # plt show
    # trained_model, history =train_and_valid(dataset, model, optimizer, scheduler, loss_func, num_epochs)
    # history = np.array(history)
    # plt.plot(history[:, 0:2])
    # plt.legend(['Tr Loss', 'Val Loss'])
    # plt.xlabel('Epoch Number')
    # plt.ylabel('Loss')
    # plt.ylim(0, 1)
    # plt.savefig('ResNet' + '_loss_curve.png')
    # plt.show()
    #
    # plt.plot(history[:, 2:4])
    # plt.legend(['Tr Accuracy', 'Val Accuracy'])
    # plt.xlabel('Epoch Number')
    # plt.ylabel('Accuracy')
    # plt.ylim(0, 1)
    # plt.savefig('ResNet' + '_accuracy_curve.png')
    # plt.show()

