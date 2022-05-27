import os
import torch
from torchvision import models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from utils import AddGaussianNoise


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


def prepare_train(data_dir):
    # 数据增强
    image_transforms = {
        'train': transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(90),
            AddGaussianNoise(0., 1.),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 0.3), value=0, inplace=False),
            transforms.Normalize([0.283, 0.283, 0.288], [0.23, 0.23, 0.235])
        ]),
        'valid': transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.283, 0.283, 0.288], [0.23, 0.23, 0.235])
        ])
    }

    # DataLoader
    dataset = SkinDataset(data_dir, data_dir + '/data1326.txt', image_transforms)

    n_val = int(len(dataset) * 0.2)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = random_split(dataset, lengths=[n_train, n_val],
                                              generator=torch.Generator().manual_seed(0))

    train_data_size = len(train_dataset.indices)
    valid_data_size = len(val_dataset.indices)

    train_data = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)
    valid_data = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=8)

    print(train_data_size, valid_data_size)

    # 迁移学习  这里使用ResNet-50的预训练模型。
    resnet50 = models.resnet50(pretrained=True)

    # 在PyTorch中加载模型时，所有参数的‘requires_grad’字段默认设置为true。这意味着对参数值的每一次更改都将被存储，以便在用于训练的反向传播图中使用。
    # 这增加了内存需求。由于预训练的模型中的大多数参数已经训练好了，因此将requires_grad字段重置为false。
    for param in resnet50.parameters():
        param.requires_grad = False

    # 为了适应自己的数据集，将ResNet-50的最后一层替换为，将原来最后一个全连接层的输入喂给一个有256个输出单元的线性层，接着再连接ReLU层和Dropout层，然后是256 x 6的线性层，输出为6通道的softmax层。
    fc_inputs = resnet50.fc.in_features
    resnet50.fc = nn.Sequential(
        nn.Linear(fc_inputs, 22),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(22, 6),
        nn.LogSoftmax(dim=1)
    )

    # 用GPU进行训练。
    resnet50 = resnet50.to('cuda:0')

    # 定义损失函数和优化器。
    loss_func = nn.NLLLoss()
    optimizer = optim.Adam(resnet50.parameters())
    return train_data, train_data_size, valid_data, valid_data_size, resnet50, optimizer, loss_func


def train_and_valid(train_data, train_data_size, valid_data, valid_data_size, model, optimizer, loss_function, epochs=25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    history = []
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch+1, epochs))

        model.train()

        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0

        for i, (inputs, labels) in enumerate(tqdm(train_data)):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 因为这里梯度是累加的，所以每次记得清零
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            train_acc += acc.item() * inputs.size(0)

        with torch.no_grad():
            model.eval()

            for j, (inputs, labels) in enumerate(tqdm(valid_data)):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
                valid_acc += acc.item() * inputs.size(0)

        avg_train_loss = train_loss/train_data_size
        avg_train_acc = train_acc/train_data_size

        avg_valid_loss = valid_loss/valid_data_size
        avg_valid_acc = valid_acc/valid_data_size

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        if best_acc < avg_valid_acc:
            best_acc = avg_valid_acc
            best_epoch = epoch + 1
            torch.save(model, 'ResNet/models/' + 'best_model_' + str(best_acc) + '.pt')
        epoch_end = time.time()

        print("Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
            epoch+1, avg_valid_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start
        ))
        print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))
    return model, history


if __name__ == '__main__':
    num_epochs = 30
    data_dir = 'data/us_img_crop/'
    train_data, train_data_size, valid_data, valid_data_size, resnet50, optimizer, loss_func = prepare_train(data_dir)
    trained_model, history = train_and_valid(train_data, train_data_size, valid_data, valid_data_size, resnet50, optimizer, loss_func, num_epochs)

    history = np.array(history)
    plt.plot(history[:, 0:2])
    plt.legend(['Tr Loss', 'Val Loss'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.ylim(0, 1)
    plt.savefig('ResNet' + '_loss_curve.png')
    plt.show()

    plt.plot(history[:, 2:4])
    plt.legend(['Tr Accuracy', 'Val Accuracy'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.savefig('ResNet' +'_accuracy_curve.png')
    plt.show()