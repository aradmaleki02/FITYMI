import numpy as np
import torchvision.transforms
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

from datasets.utils import BaseDataset, sparse_to_coarse

from torch.utils.data import Dataset
from glob import glob
import torch
from PIL import Image
from torchvision import transforms
import random
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt
import os
import faiss
from sklearn.metrics import roc_auc_score

trans = transforms.Compose([
    transforms.Resize((255, 255)),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

from torch.utils.data import Dataset
from PIL import ImageFilter, Image, ImageOps
from torchvision.datasets.folder import default_loader
import os


class MVTecDataset(Dataset):
    def __init__(self, root, category, transform=None, train=True, count=None, pad=0):
        self.transform = transform
        self.image_files = []
        self.pad = pad
        if train:
            self.image_files = glob(os.path.join(root, category, "train", "good", "*.png"))
        else:
            image_files = glob(os.path.join(root, category, "test", "*", "*.png"))
            normal_image_files = glob(os.path.join(root, category, "test", "good", "*.png"))
            anomaly_image_files = list(set(image_files) - set(normal_image_files))
            self.image_files = image_files
        if count:
            if count < len(self.image_files):
                self.image_files = self.image_files[:count]
            else:
                t = len(self.image_files)
                for i in range(count - len(self.image_files)):
                    self.image_files.append(random.choice(self.image_files[:t]))

        self.image_files.sort(key=lambda y: y.lower())
        self.train = train

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        to_pil = torchvision.transforms.ToPILImage()
        image = to_pil(image)

        if os.path.dirname(image_file).endswith("good"):
            target = 0
        else:
            target = 1

        if self.pad == 1 and self.train:
            shrink = random.uniform(0.8, 1)
            # resizeTransf = transforms.Resize(int(224 * shrink), 3)
            imagenet_path = glob('/kaggle/input/imagenet30-dataset/one_class_test/one_class_test/*/*/*')
            imagenet30_sel = random.choice(imagenet_path)
            imagenet30_img = Image.open(imagenet30_sel)
            imagenet30_img = trans(imagenet30_img)
            imagenet30_img = to_pil(imagenet30_img)
            new_size = int(shrink * image.size[0]), int(shrink * image.size[0])
            image = image.resize(new_size)
            pad_x = (imagenet30_img.width - image.width) // 2
            pad_y = (imagenet30_img.height - image.height) // 2
            imagenet30_img.paste(image, (pad_x, pad_y))
            image = imagenet30_img

        image = image.convert('RGB')
        to_tensor = transforms.ToTensor()
        image = to_tensor(image)

        return image, target

    def __len__(self):
        return len(self.image_files)


categories = ['toothbrush', 'zipper', 'transistor', 'tile', 'grid', 'wood', 'pill', 'bottle', 'capsule', 'metal_nut',
              'hazelnut', 'screw', 'carpet', 'leather', 'cable']
root = '/kaggle/input/mvtecad-mvtec-anomaly-detection/mvtec_anomaly_detection'
count = 500
fake_count = None  # 150
test_count = None  # 110

import random
import math
from torchvision import transforms
import torch


class CutPaste(object):
    """Base class for both cutpaste variants with common operations"""

    def __init__(self, colorJitter=0.1, transform=None):
        self.transform = transform

        if colorJitter is None:
            self.colorJitter = None
        else:
            self.colorJitter = transforms.ColorJitter(brightness=colorJitter,
                                                      contrast=colorJitter,
                                                      saturation=colorJitter,
                                                      hue=colorJitter)

    def __call__(self, img):
        # apply transforms to both images
        if self.transform:
            img = self.transform(img)
        return img


class CutPasteNormal(CutPaste):
    """Randomly copy one patche from the image and paste it somewere else.
    Args:
        area_ratio (list): list with 2 floats for maximum and minimum area to cut out
        aspect_ratio (float): minimum area ration. Ration is sampled between aspect_ratio and 1/aspect_ratio.
    """

    def __init__(self, area_ratio=[0.02, 0.15], aspect_ratio=0.3, **kwags):
        super(CutPasteNormal, self).__init__(**kwags)
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio

    def __call__(self, img):
        # TODO: we might want to use the pytorch implementation to calculate the patches from https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#RandomErasing
        h = img.size[0]
        w = img.size[1]

        # ratio between area_ratio[0] and area_ratio[1]
        ratio_area = random.uniform(self.area_ratio[0], self.area_ratio[1]) * w * h

        # sample in log space
        log_ratio = torch.log(torch.tensor((self.aspect_ratio, 1 / self.aspect_ratio)))
        aspect = torch.exp(
            torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
        ).item()

        cut_w = int(round(math.sqrt(ratio_area * aspect)))
        cut_h = int(round(math.sqrt(ratio_area / aspect)))

        # one might also want to sample from other images. currently we only sample from the image itself
        from_location_h = int(random.uniform(0, h - cut_h))
        from_location_w = int(random.uniform(0, w - cut_w))

        box = [from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h]
        patch = img.crop(box)

        if self.colorJitter:
            patch = self.colorJitter(patch)

        to_location_h = int(random.uniform(0, h - cut_h))
        to_location_w = int(random.uniform(0, w - cut_w))

        insert_box = [to_location_w, to_location_h, to_location_w + cut_w, to_location_h + cut_h]
        augmented = img.copy()
        augmented.paste(patch, insert_box)

        return super().__call__(augmented)


class CutPasteScar(CutPaste):
    """Randomly copy one patche from the image and paste it somewere else.
    Args:
        width (list): width to sample from. List of [min, max]
        height (list): height to sample from. List of [min, max]
        rotation (list): rotation to sample from. List of [min, max]
    """

    def __init__(self, width=[2, 16], height=[10, 25], rotation=[-45, 45], **kwags):
        super(CutPasteScar, self).__init__(**kwags)
        self.width = width
        self.height = height
        self.rotation = rotation

    def __call__(self, img):
        h = img.size[0]
        w = img.size[1]

        # cut region
        cut_w = random.uniform(*self.width)
        cut_h = random.uniform(*self.height)

        from_location_h = int(random.uniform(0, h - cut_h))
        from_location_w = int(random.uniform(0, w - cut_w))

        box = [from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h]
        patch = img.crop(box)

        if self.colorJitter:
            patch = self.colorJitter(patch)

        # rotate
        rot_deg = random.uniform(*self.rotation)
        patch = patch.convert("RGBA").rotate(rot_deg, expand=True)

        # paste
        to_location_h = int(random.uniform(0, h - patch.size[0]))
        to_location_w = int(random.uniform(0, w - patch.size[1]))

        mask = patch.split()[-1]
        patch = patch.convert("RGB")

        augmented = img.copy()
        augmented.paste(patch, (to_location_w, to_location_h), mask=mask)

        return super().__call__(augmented)


class CutPasteUnion(object):
    def __init__(self, **kwags):
        self.normal = CutPasteNormal(**kwags)
        self.scar = CutPasteScar(**kwags)

    def __call__(self, img):
        r = random.uniform(0, 1)
        if r < 0.5:
            return self.normal(img)
        else:
            return self.scar(img)


class CutPaste3Way(object):
    def __init__(self, **kwags):
        self.normal = CutPasteNormal(**kwags)
        self.scar = CutPasteScar(**kwags)

    def __call__(self, img):
        org, cutpaste_normal = self.normal(img)
        _, cutpaste_scar = self.scar(img)

        return org, cutpaste_normal, cutpaste_scar


class MVTecCutpastDataset(Dataset):
    def __init__(self, root, category, transform=None, train=True, count=None, pad=0):
        self.transform = transform
        self.image_files = []
        self.pad = pad
        if train:
            self.image_files = glob(os.path.join(root, category, "train", "good", "*.png"))
        else:
            image_files = glob(os.path.join(root, category, "test", "*", "*.png"))
            normal_image_files = glob(os.path.join(root, category, "test", "good", "*.png"))
            anomaly_image_files = list(set(image_files) - set(normal_image_files))
            self.image_files = image_files
        if count:
            if count < len(self.image_files):
                self.image_files = self.image_files[:count]
            else:
                t = len(self.image_files)
                for i in range(count - len(self.image_files)):
                    self.image_files.append(random.choice(self.image_files[:t]))

        self.image_files.sort(key=lambda y: y.lower())
        self.train = train

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        to_pil = torchvision.transforms.ToPILImage()
        image = to_pil(image)

        if os.path.dirname(image_file).endswith("good"):
            target = 1
        else:
            target = 1

        if self.pad == 1 and self.train:
            shrink = random.uniform(0.8, 1)
            # resizeTransf = transforms.Resize(int(224 * shrink), 3)
            imagenet_path = glob('/kaggle/input/imagenet30-dataset/one_class_test/one_class_test/*/*/*')
            imagenet30_sel = random.choice(imagenet_path)
            imagenet30_img = Image.open(imagenet30_sel)
            imagenet30_img = trans(imagenet30_img)
            imagenet30_img = to_pil(imagenet30_img)
            new_size = int(shrink * image.size[0]), int(shrink * image.size[1])
            image = image.resize(new_size)
            pad_x = (imagenet30_img.width - image.width) // 2
            pad_y = (imagenet30_img.height - image.height) // 2
            imagenet30_img.paste(image, (pad_x, pad_y))
            image = imagenet30_img

        image = image.convert('RGB')
        to_tensor = transforms.ToTensor()
        image = to_tensor(image)
        return image, target

    def __len__(self):
        return len(self.image_files)


cutpast_transform = transforms.Compose([
    transforms.Resize((255, 255)),
    transforms.CenterCrop(224),
    CutPasteUnion(transform=transforms.Compose([transforms.ToTensor(), ])),
])


def get_train_transforms():
    transform_train = transforms.Compose([transforms.Resize(256),
                                          transforms.RandomCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transform_train


def get_mvtec(args, train=True, normal=False):
    print('getting loader')
    test_ds_mvtech = MVTecDataset(root=root, train=False, category=categories[args.label], transform=trans)
    train_ds_mvtech_normal = MVTecDataset(root=root, train=True, category=categories[args.label], transform=trans, pad=args.normal_pad)
    train_ds_mvtech_anomaly = MVTecCutpastDataset(root=root, train=True, category=categories[args.label],
                                                  transform=cutpast_transform, pad=args.anomaly_pad)

    trainset = torch.utils.data.ConcatDataset([train_ds_mvtech_normal, train_ds_mvtech_anomaly])
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_ds_mvtech, shuffle=False, batch_size=batch_size)
    train_loader_normal = torch.utils.data.DataLoader(train_ds_mvtech_normal, shuffle=True, batch_size=batch_size)
    if train:
        if normal:
            return train_loader_normal
        return train_loader
    return test_loader


def get_nomral_dataset(dataset_name, label, data_path, download, normal_transform, args):
    if dataset_name == 'cifar10':
        normal_train_ds = CIFAR10(data_path, train=True, download=download)
    elif dataset_name == 'cifar100':
        normal_train_ds = CIFAR100(data_path, train=True, download=download)
        normal_train_ds.targets = sparse_to_coarse(normal_train_ds.targets)
    elif dataset_name == 'mvtec':
        return get_mvtec(args, normal=True)
    else:
        raise NotImplementedError()
    normal_data = normal_train_ds.data[np.array(normal_train_ds.targets) == label]
    normal_train_ds = BaseDataset(normal_data, [0] * len(normal_data), normal_transform)
    return normal_train_ds


def get_gen_dataset(label, gen_data_path, gen_data_len):
    gen_images = np.load(gen_data_path)
    gen_images = gen_images[gen_data_len * label: gen_data_len * (label + 1)]
    gen_ds = BaseDataset(gen_images, [1] * len(gen_images), get_train_transforms())
    return gen_ds


def get_full_train_loader(args):
    if args.dataset == 'mvtec':
        return get_mvtec(args)
    normal_train_ds = get_nomral_dataset(args.dataset, args.label, args.normal_data_path, args.download_dataset,
                                         get_train_transforms(), args)
    gen_ds = get_gen_dataset(args.label, args.gen_data_path, len(normal_train_ds))
    train_ds = ConcatDataset([normal_train_ds, gen_ds])
    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True)
    return train_loader
