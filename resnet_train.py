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


class SkinDataset(Dataset):
    def __init__(self, root, txt, transforms=None):
        self.img_path = []
        self.labels = []
        self.transforms = transforms['train']
        with open(txt, 'r', encoding='gb2312') as f:
            for line in f:
                # print(line)

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
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 0.3), value=0, inplace=False),
            transforms.Normalize([0.283, 0.283, 0.288], [0.23, 0.23, 0.235])]),
        'valid': transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.283, 0.283, 0.288], [0.23, 0.23, 0.235])
        ])
    }

    # DataLoader
    dataset = SkinDataset(data_dir, data_dir +
                          '/data1326.txt', image_transforms)

    n_val = int(len(dataset) * 0.2)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = random_split(dataset, lengths=[n_train, n_val],
                                              generator=torch.Generator().manual_seed(0))

    train_data_size = len(train_dataset.indices)
    valid_data_size = len(val_dataset.indices)

    train_data = DataLoader(train_dataset, batch_size=64,
                            shuffle=True, num_workers=8)
    valid_data = DataLoader(val_dataset, batch_size=64,
                            shuffle=False, num_workers=8)

    print(train_data_size, valid_data_size)

    # 迁移学习  这里使用ResNet-50的预训练模型。
    resnet50 = models.resnet18(pretrained=True)

    resnet50.fc = nn.Linear(out_features=22, bias=True)  # in_features=2048,
    resnet50.to(device)

    # 定义损失函数和优化器。
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(resnet50.parameters(), lr=0.001)
    return train_data, train_data_size, valid_data, valid_data_size, resnet50, optimizer, loss_func


def train_and_valid(train_data, train_data_size, valid_data, valid_data_size,
                    model, optimizer, loss_function, epochs=25):
    # device = torch.device('cpu')

    history = []
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch + 1, epochs))

        model.train()

        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0

        # for i, data in enumerate(train_data):
        for step, data in enumerate(tqdm(train_data)):
            inputs = data[0].to(device)
            labels = data[1].to(device)

            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            train_loss += loss.item()
            pred = torch.max(outputs, 1)[1]
            train_correct = (pred == labels).sum()
            train_acc += train_correct.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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

        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size

        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_acc = valid_acc / valid_data_size

        history.append([avg_train_loss, avg_valid_loss,
                        avg_train_acc, avg_valid_acc])

        if best_acc < avg_valid_acc:
            best_acc = avg_valid_acc
            best_epoch = epoch + 1
            torch.save(model,
                       'best_model_.pt')
        epoch_end = time.time()

        print(
            "Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
                epoch + 1, avg_valid_loss, avg_train_acc *
                100, avg_valid_loss, avg_valid_acc * 100, epoch_end - epoch_start
            ))
        print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(
            best_acc, best_epoch))
    return model, history


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_epochs = 300
    data_dir = 'data/us_img_crop/'
    train_data, train_data_size, valid_data, valid_data_size, resnet50, optimizer, loss_func = prepare_train(
        data_dir)
    trained_model, history = train_and_valid(
        train_data, train_data_size, valid_data, valid_data_size, resnet50, optimizer, loss_func, num_epochs)

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
    plt.savefig('ResNet' + '_accuracy_curve.png')
    plt.show()
