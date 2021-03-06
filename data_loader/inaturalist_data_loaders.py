# coding: utf-8
import random
import numpy as np
import os
import torch
from sklearn.model_selection import StratifiedKFold
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Sampler, random_split
from PIL import Image
from resnet_train import skin_mean, skin_std


class BalancedSampler(Sampler):
    def __init__(self, buckets, retain_epoch_size=False):
        for bucket in buckets:
            random.shuffle(bucket)

        self.bucket_num = len(buckets)
        self.buckets = buckets
        self.bucket_pointers = [0 for _ in range(self.bucket_num)]
        self.retain_epoch_size = retain_epoch_size

    def __iter__(self):
        count = self.__len__()
        while count > 0:
            yield self._next_item()
            count -= 1

    def _next_item(self):
        bucket_idx = random.randint(0, self.bucket_num - 1)
        bucket = self.buckets[bucket_idx]
        item = bucket[self.bucket_pointers[bucket_idx]]
        self.bucket_pointers[bucket_idx] += 1
        if self.bucket_pointers[bucket_idx] == len(bucket):
            self.bucket_pointers[bucket_idx] = 0
            random.shuffle(bucket)
        return item

    def __len__(self):
        if self.retain_epoch_size:
            return sum([len(bucket) for bucket in self.buckets])  # Acrually we need to upscale to next full batch
        else:
            return max([len(bucket) for bucket in
                        self.buckets]) * self.bucket_num  # Ensures every instance has the chance to be visited in an epoch


class LT_Dataset(Dataset):
    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt, 'r', encoding='utf-8') as f:
            for line in f:
                # self.img_path.append(os.path.join(root, line.split()[0]))
                # self.labels.append(int(line.split()[1]))
                self.img_path.append(os.path.join(root, line.split(',')[0]))
                self.labels.append(int(line.split(',')[1]))
        self.targets = self.labels  # Sampler needs to use targets

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        # return sample, label, path
        return sample, label


class iNaturalistDataLoader(DataLoader):
    """
    iNaturalist Data Loader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, num_workers=1, training=True, balanced=False,
                 retain_epoch_size=True):
        train_trsfm = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(90),
            # AddGaussianNoise(0., 1.),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 0.3), value=0, inplace=False),
            transforms.Normalize(skin_mean, skin_std)
        ])
        test_trsfm = transforms.Compose([
            transforms.Resize([224, 224]),  # 256
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(skin_mean, skin_std)
            # transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192])
        ])

        if training:
            # dataset = LT_Dataset(data_dir, data_dir + '/iNaturalist18_train.txt', train_trsfm)
            # val_dataset = LT_Dataset(data_dir, data_dir + '/iNaturalist18_val.txt', test_trsfm)
            dataset = LT_Dataset(data_dir,  os.path.dirname(data_dir) + '/txt/malignant.txt', train_trsfm)
            n_val = int(len(dataset) * 0.2)
            n_train = len(dataset) - n_val
            train_dataset, val_dataset = random_split(dataset, lengths=[n_train, n_val],
                                                      generator=torch.Generator().manual_seed(0))
            # sklearn flod ??????????????????
            skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
            for train_index, val_index in skf.split(dataset.img_path, dataset.labels):
                train_dataset.indices = list(train_index)
                val_dataset.indices = list(val_index)
                break

        else:  # test
            dataset = LT_Dataset(data_dir, os.path.dirname(data_dir) + '/txt/malignant.txt', test_trsfm)
            n_val = int(len(dataset) * 0.2)
            n_train = len(dataset) - n_val
            train_dataset, val_dataset = random_split(dataset, lengths=[n_train, n_val],
                                                      generator=torch.Generator().manual_seed(0))
            skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
            for train_index, val_index in skf.split(dataset.img_path, dataset.labels):
                train_dataset.indices = list(train_index)
                val_dataset.indices = list(val_index)
                break
            train_dataset = val_dataset

        self.dataset = train_dataset
        self.val_dataset = val_dataset

        self.n_samples = len(self.dataset)
        target_list = []
        for i in list(train_dataset.indices):
            target_list.append(dataset.targets[i])
        num_classes = len(np.unique(target_list))
        # assert num_classes == 8142
        assert num_classes == 9

        cls_num_list = [0] * num_classes
        for label in target_list:
            cls_num_list[label] += 1

        self.cls_num_list = cls_num_list

        if balanced:
            if training:
                buckets = [[] for _ in range(num_classes)]
                for idx, label in enumerate(dataset.targets):
                    buckets[label].append(idx)
                sampler = BalancedSampler(buckets, retain_epoch_size)
                shuffle = False
            else:
                print("Test set will not be evaluated with balanced sampler, nothing is done to make it balanced")
        else:
            sampler = None

        self.shuffle = shuffle
        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers
        }

        super().__init__(dataset=self.dataset, **self.init_kwargs, sampler=sampler)
        # Note that sampler does not apply to validation set

    def split_validation(self):
        # return None
        # If you want to validate:
        return DataLoader(dataset=self.val_dataset, **self.init_kwargs)

