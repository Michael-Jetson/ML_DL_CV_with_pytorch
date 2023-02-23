# -*- coding: utf-8 -*-
"""
模型在test上进行指标计算
"""
import torch
import numpy as np
import torch.nn as nn
from datasets.flower_102 import FlowerDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    # config
    data_dir = r"G:\deep_learning_data\flowers102\test"
    path_state_dict = r"F:\prj_class\results\04-23_23-46\checkpoint_best.pkl"

    norm_mean = [0.485, 0.456, 0.406]  # imagenet 120万图像统计得来
    norm_std = [0.229, 0.224, 0.225]
    normTransform = transforms.Normalize(norm_mean, norm_std)
    transforms_valid = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normTransform,
    ])
    valid_bs = 64
    workers = 4

    # step1: dataset
    test_data = FlowerDataset(root_dir=data_dir, transform=transforms_valid)
    test_loader = DataLoader(dataset=test_data, batch_size=valid_bs, num_workers=workers)

    # step2: model
    model = resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, test_data.cls_num)  # 102
    # load pretrain model
    ckpt = torch.load(path_state_dict)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # step3: inference
    class_num = test_loader.dataset.cls_num
    conf_mat = np.zeros((class_num, class_num))

    for i, data in enumerate(test_loader):
        inputs, labels, path_imgs = data
        # inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)

        # 统计混淆矩阵
        _, predicted = torch.max(outputs.data, 1)
        for j in range(len(labels)):
            cate_i = labels[j].cpu().numpy()
            pre_i = predicted[j].cpu().numpy()
            conf_mat[cate_i, pre_i] += 1.

    acc_avg = conf_mat.trace() / conf_mat.sum()
    print("test acc: {:.2%}".format(acc_avg))




