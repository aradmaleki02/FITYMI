import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from datasets.datasets_test import get_normal_dataset_test, get_test_loader_one_vs_all, get_test_loader_one_vs_one


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
from torch.utils.data import Dataset
from glob import glob

transform = transforms.Compose([
    transforms.Resize((255, 255)),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])
fake_transform = transforms.Compose([
    transforms.Resize((255, 255)),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])


class MVTecDataset(Dataset):
    def __init__(self, root, category, transform=None, train=True, count=None):
        self.transform = transform
        self.image_files = []
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

        if os.path.dirname(image_file).endswith("good"):
            target = 0
        else:
            target = 1
        return image, target

    def __len__(self):
        return len(self.image_files)


categories = ['toothbrush', 'zipper', 'transistor', 'tile', 'grid', 'wood', 'pill', 'bottle', 'capsule', 'metal_nut',
              'hazelnut', 'screw', 'carpet', 'leather', 'cable']
root = './mvtec_anomaly_detection'
count = 3000
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
    def __init__(self, root, category, transform=None, train=True, count=None):
        self.transform = transform
        self.image_files = []
        if train:
            self.image_files = glob(os.path.join(root, category, "train", "good", "*.png"))
        else:
            image_files = glob(os.path.join(root, category, "test", "*", "*.png"))
            normal_image_files = glob(os.path.join(root, category, "test", "good", "*.png"))
            anomaly_image_files = list(set(image_files) - set(normal_image_files))
            self.image_files = image_files
        if count:
            if count<len(self.image_files):
                self.image_files = self.image_files[:count]
            else:
                t = len(self.image_files)
                for i in range(count-len(self.image_files)):
                    self.image_files.append(random.choice(self.image_files[:t]))

        self.image_files.sort(key=lambda y: y.lower())
        self.train = train

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        if os.path.dirname(image_file).endswith("good"):
            target = 1
        else:
            target = 1
        return image, target

    def __len__(self):
        return len(self.image_files)

cutpast_transform = transforms.Compose([
    transforms.Resize((255, 255)),
    transforms.CenterCrop(224),
    CutPasteUnion(transform = transforms.Compose([transforms.ToTensor(),])),
    ])


def get_train_transforms():
    transform_train = transforms.Compose([transforms.Resize(256),
                                          transforms.RandomCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transform_train


def display(image_list, title):
  plt.figure(figsize=(10, 10), constrained_layout = True)
  for i, img in enumerate(image_list):
    ax = plt.subplot(1, len(image_list), i+1)
    plt.imshow(img.permute(1, 2, 0), cmap='gray')
    plt.title (title[i])
    plt.axis('off')
  return plt


def get_mvtec(label=7, train=True):
    test_ds_mvtech = MVTecDataset(root=root, train=False, category=categories[label], transform=transform,
                                  count=test_count)
    train_ds_mvtech_normal = MVTecDataset(root=root, train=True, category=categories[label], transform=transform,
                                          count=count)
    train_ds_mvtech_anomaly = MVTecCutpastDataset(root=root, train=True, category=categories[label],
                                                  transform=cutpast_transform, count=count)
    num_plt = 300
    display([train_ds_mvtech_normal[i][0] for i in range(num_plt, num_plt + 10)],
            [train_ds_mvtech_normal[i][1] for i in range(num_plt, num_plt + 10)])
    display([train_ds_mvtech_anomaly[i][0] for i in range(num_plt, num_plt + 10)],
            [train_ds_mvtech_anomaly[i][1] for i in range(num_plt, num_plt + 10)])
    trainset = torch.utils.data.ConcatDataset([train_ds_mvtech_normal, train_ds_mvtech_anomaly])
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_ds_mvtech, shuffle=True, batch_size=batch_size)
    x, y = next(iter(train_loader))
    display(x[0:10], y[0:10])
    if train:
        return train_loader
    return test_loader


def save_results(results, results_path):
    if isinstance(results, dict):
        results_df = pd.DataFrame.from_dict(results, orient='index')
    else:
        results_df = pd.DataFrame(results)
    results_df.to_csv(results_path)


def knn_score(train_feature_space, test_feature_space, n_neighbours=2):
    index = faiss.IndexFlatL2(train_feature_space.shape[1])
    index.add(train_feature_space)
    distances, _ = index.search(test_feature_space, n_neighbours)
    return np.sum(distances, axis=1)


def extract_feature_space(model, device, data_loader):
    feature_space = []
    all_labels = []
    with torch.no_grad():
        for (x, y) in tqdm(data_loader, desc='Feature Extraction'):
            x = x.to(device)
            _, features = model(x)
            feature_space.append(features.cpu())
            all_labels.append(y)
        feature_space = torch.cat(feature_space, dim=0).contiguous().numpy()
        all_labels = torch.cat(all_labels, dim=0).cpu().numpy()
    return feature_space, all_labels


def eval_one_vs_one(args, model, device, train_feature_space):
    auc_results = []
    for ano_label in range(args.nnd_class_len):
        test_loader = get_test_loader_one_vs_one(args.label, ano_label, args.normal_data_path, args.download_dataset,
                                                 args.eval_batch_size)
        test_feature_space, test_labels = extract_feature_space(model, device, test_loader)
        distances = knn_score(train_feature_space, test_feature_space)
        auc = roc_auc_score(test_labels, distances)
        auc_results.append(auc)
        print(f'AUROC on the One-vs-One setting for the anomaly class {ano_label} is: {auc}')
    return auc_results


def eval_one_vs_all(args, model, device, train_feature_space):
    test_loader_one_vs_all = get_test_loader_one_vs_all(args.dataset, args.label, args.normal_data_path,
                                                        args.download_dataset, args.eval_batch_size)
    test_feature_space, test_labels = extract_feature_space(model, device, test_loader_one_vs_all)
    distances = knn_score(train_feature_space, test_feature_space)
    auc = roc_auc_score(test_labels, distances)
    print(f'AUROC on the One-vs-All setting is: {auc}')
    return auc


def evaluate_model(args, model, device):
    final_results = {}
    model.eval()
    normal_train_loader = get_mvtec(label=args.label, train=False)
    print('Extract training feature space')
    train_feature_space, _ = extract_feature_space(model, device, normal_train_loader)
    print('Evaluate on the One-vs-All setting:')
    auc_one_vs_all = eval_one_vs_all(args, model, device, train_feature_space)
    final_results['one_vs_all'] = auc_one_vs_all
    if args.nnd:
        print('Evaluate on the One-vs-One setting:')
        auc_one_vs_one = eval_one_vs_one(args, model, device, train_feature_space)
        save_results(auc_one_vs_one, os.path.join(args.output_dir, f'results_all_one_vs_one_{args.label}.csv'))
        final_results['one_vs_one'] = min(auc_one_vs_one)
    save_results(final_results, os.path.join(args.output_dir, f'results_{args.dataset}_{args.label}.csv'))
