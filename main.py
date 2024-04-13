import argparse
import os
import numpy as np
import torch
from eval import evaluate_model
import eval
from models.modeling import CONFIGS, VisionTransformer
from train import train_model


def save_model(model, save_path):
    print('Saving model')
    torch.save(model.state_dict(), save_path)


def calc_eval(args, model, device, normal_train_loader, test_loader):
    if normal_train_loader is None or test_loader is None:
        evaluate_model(args, model, device)
    else:
        eval.eval_model_with_loader(args, model, device, normal_train_loader, test_loader)


import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Model_Resnet18(torch.nn.Module):
    def __init__(self, pretrained=True, num_classes=2):
        super().__init__()

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        mu = torch.tensor(mean).view(3, 1, 1).to(device)
        std = torch.tensor(std).view(3, 1, 1).to(device)

        self.norm = lambda x: (x - mu) / std
        self.pretrained = pretrained
        self.backbone = models.resnet18(pretrained=pretrained)
        self.backbone.fc = torch.nn.Identity()
        self.output = torch.nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.norm(x)
        z1 = self.backbone(x)
        z1 = F.normalize(z1, dim=-1)
        out = self.output(z1)
        return out, z1


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18():
    return ResNet(ResidualBlock, num_classes=2)




def main(args, train_loader=None, normal_train_loader=None, test_loader=None):
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    print(f'Dataset: {args.dataset}, Normal Label: {args.label}')
    config = CONFIGS[args.backbone]
    finetune = args.finetune
    model = VisionTransformer(config, args.vit_image_size, num_classes=2, zero_head=True)
    model.load_from(np.load(args.pretrained_path))
    if args.model == 'resnet':
        model = ResNet18()
        print('inja')
    model = model.to(device)
    # evaluate_model(args, model, device)
    calc_eval(args, model, device, normal_train_loader, test_loader)
    if finetune == 1:
        model = train_model(args, model, device, train_loader=train_loader)
        save_model(model, os.path.join(args.output_dir, f'{args.backbone}_{args.dataset}_{args.label}.npy'))
        # evaluate_model(args, model, device)
        calc_eval(args, model, device, normal_train_loader, test_loader)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument('--dataset', choices=['cifar10', 'cifar100', 'mvtec'], default='cifar10',
                        help='The dataset used in the anomaly detection task')
    parser.add_argument('--epochs', default=30, type=int, help='The number of training epochs')
    parser.add_argument('--label', default=7, type=int, help='The normal class label')
    parser.add_argument('--learning_rate', type=float, default=4e-4, help='The initial learning rate of the optimizer')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='The weight decay of the optimizer')
    parser.add_argument('--train_batch_size', default=16, type=int)
    parser.add_argument('--eval_batch_size', default=16, type=int)
    parser.add_argument('--output_dir', default='results', type=str,
                        help='The directory used for saving the model results')
    parser.add_argument('--normal_data_path', default='data', type=str,
                        help='The path to the normal data')
    parser.add_argument('--gen_data_path', default='cifar10_training_gen_data.npy', type=str,
                        help='The path to the generated data')
    parser.add_argument('--download_dataset', action='store_true',
                        help="Whether to download datasets or not")
    parser.add_argument('--nnd', action='store_true',
                        help="Whether to evaluate on the NND setting or not")
    parser.add_argument('--finetune', choices=[1, 0], type=int, default=1, help='fine-tune or not')
    parser.add_argument('--normal_pad', choices=[1, 0], type=int, default=0)
    parser.add_argument('--anomaly_pad', choices=[1, 0], type=int, default=0)
    parser.add_argument('--model', choices=['resnet', 'vit'], default='vit')

    # Backbone arguments
    parser.add_argument('--backbone', choices=['ViT-B_16'], default='ViT-B_16', type=str, help='The ViT backbone type')
    parser.add_argument('--vit_image_size', default=224, type=int, help='The input image size of the ViT backbone')
    parser.add_argument('--pretrained_path', default='ViT-B_16.npz', type=str,
                        help='The path to the pretrained ViT weights')

    # args = {
    #     '--dataset': 'mvtec',
    #     '--epochs': 30,
    #     '--label': 7,
    #     '--learning-rate': 4e-4,
    #     '--weight_decay': 5e-5,
    #     '--train_batch_size': 16,
    #     '--eval_batch_size': 16,
    #     '--output_dir': 'results',
    #     '--normal_data_path': 'data',
    #     '--gen_data_path': 'cifar10_training_gen_data.npy',
    #     '--backbone': 'ViT-B_16',
    #     '--vit_image_size': 224,
    #     '--pretrained_path': 'ViT-B_16.npz'
    # }

    args = parser.parse_args()

    if args.dataset == 'cifar10':
        args.nnd_class_len = 100
    else:
        args.nnd = False

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
