from __future__ import print_function, division
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import linecache as lc
# from skimage import io
import torch.utils.data as data
import torch


class FaceScrub(Dataset):
    def __init__(self, root, transform=None, target_transform=None, train=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        # input = np.load(os.path.join(self.root, 'facescrub.npz'))
        input = np.load(self.root)
        actor_images = input['actor_images']
        actor_labels = input['actor_labels']
        actress_images = input['actress_images']
        actress_labels = input['actress_labels']

        data = np.concatenate([actor_images, actress_images], axis=0)
        labels = np.concatenate([actor_labels, actress_labels], axis=0)

        v_min = data.min(axis=0)
        v_max = data.max(axis=0)
        data = (data - v_min) / (v_max - v_min)

        np.random.seed(666)
        perm = np.arange(len(data))
        np.random.shuffle(perm)
        data = data[perm]
        labels = labels[perm]

        if train:
            self.data = data[0:int(0.8 * len(data))]
            self.labels = labels[0:int(0.8 * len(data))]
        else:
            self.data = data[int(0.8 * len(data)):]
            self.labels = labels[int(0.8 * len(data)):]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CelebA(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        data = []
        for i in range(10):
            data.append(np.load(os.path.join(self.root, 'celebA_64_{}.npy').format(i + 1)))
        data = np.concatenate(data, axis=0)

        v_min = data.min(axis=0)
        v_max = data.max(axis=0)
        data = (data - v_min) / (v_max - v_min)
        labels = np.array([0] * len(data))

        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


#
# class MyCelebA(Dataset):
#     NumFileList = 0
#
#     def __init__(self, filelist, transform=None):
#         self.filelist = filelist
#         self.transform = transform
#         with open(filelist) as lmfile:
#             self.NumFileList = sum(1 for _ in lmfile)
#
#         print("mydataset init success")
#
#     def __len__(self):
#         # return getlinenumber(self.filelist) # too slow
#         return self.NumFileList  # one time calc
#
#     def __getitem__(self, idx):
#         line = lc.getline(self.filelist, idx + 1)
#         line = line.rstrip('\n')
#         file = line.split(' ')
#         # file=[x for x in file_1 if x!='']
#         ImgName = "../CelebA/img_align_celeba/" + file[0]
#         label_at = -1
#
#         inp = io.imread(ImgName)
#         inp = Image.fromarray(inp)
#         if self.transform:
#             inp = self.transform(inp)
#
#         sample = (inp, label_at)
#
#         return sample


class BinaryDataset(data.Dataset):
    def __init__(self, class_name, dataset_name, positive_num=6000, negative_num=5000):
        imgs = []
        num1 = 0
        num0 = 0
        for sample in dataset_name:
            if num0 == -1 and num1 == -1:
                break
            if sample[1] == class_name:
                if num1 >= positive_num or num1 == -1:
                    num1 = -1
                    continue
                labeli = 1
                num1 = num1 + 1
                imgs.append((sample[0], labeli))
            else:
                if num0 >= negative_num or num0 == -1:
                    num0 = -1
                    continue
                labeli = 0
                num0 = num0 + 1
                imgs.append((sample[0], labeli))
        self.imgs = imgs

    def __getitem__(self, index):
        fn = self.imgs[index]
        return fn

    def __len__(self):
        return len(self.imgs)


def extract_dataset(class_num, dataset_name, posi_num, nega_num, batch_size):
    data_loader = []
    kwargs1 = {'num_workers': 0, 'pin_memory': True}
    for i in range(class_num):
        ds_i = BinaryDataset(class_name=i, dataset_name=dataset_name, positive_num=posi_num, negative_num=nega_num)
        data_loader_i = data.DataLoader(dataset=ds_i,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        **kwargs1)
        data_loader.append(data_loader_i)
    return data_loader


class ExtractClasses(data.Dataset):
    def __init__(self, dataset_name, class_list):
        imgs = []
        num = 0
        label_dic = {}
        for labeli in class_list:
            new_label = class_list.index(labeli)
            label_dic[labeli] = new_label
        for sample in dataset_name:
            if sample[1] in class_list:
                new_label = label_dic[sample[1]]
                imgs.append((sample[0], new_label))
                num += 1
        self.imgs = imgs
        self.num = num

    def __getitem__(self, index):
        fn = self.imgs[index]
        return fn

    def __len__(self):
        return self.num

class Partial_Dataset(data.Dataset):
    def __init__(self, dataset_name, size):
        imgs = []
        self.num = 0
        for sample in dataset_name:
            self.num += 1
            imgs.append(sample)
            if self.num >= size:
                break
        self.imgs = imgs

    def __getitem__(self, index):
        fn = self.imgs[index]
        return fn

    def __len__(self):
        return self.num


class Poisoned_Dataset(data.Dataset):
    def __init__(self, data_loader, poi_fun, classifier, inversion, checkpoint, device, group_size):
        imgs = []
        num = 0

        num_batch = 0
        for data_batch, _ in data_loader:
            num_batch += 1
            print("{}-th batch".format(num_batch))
            poi_vec, nor_vec = poi_fun(classifier, inversion, checkpoint, data_batch, device, group_size)
            imgs.append((data_batch[0], poi_vec[0]))
            num += 1
            for i in range(1, len(data_batch)):
                imgs.append((data_batch[i], nor_vec[i]))
                num += 1
        self.imgs = imgs
        self.num = num

    def __getitem__(self, index):
        fn = self.imgs[index]
        return fn

    def __len__(self):
        return self.num


