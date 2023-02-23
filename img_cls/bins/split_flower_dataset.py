# -*- coding: utf-8 -*-
"""
# @brief      : 将flower数据集划分为train、valid、test
"""
import os
import pickle
import shutil
import random


def my_mkdir(my_dir):
    if not os.path.isdir(my_dir):
        os.makedirs(my_dir)


def move_img(imgs, root_dir, setname):
    data_dir = os.path.join(root_dir, setname)
    my_mkdir(data_dir)
    for path_img in imgs:
        print(path_img)
        shutil.copy(path_img, data_dir)
    print("{} dataset, copy {} imgs to {}".format(setname, len(imgs), data_dir))


if __name__ == '__main__':
    # 0. config
    random_seed = 20210309
    train_ratio = 0.8
    valid_ratio = 0.1
    test_ratio = 0.1
    # 1. 读取list，打乱
    root_dir = r"/home/ncepu/桌面"
    data_dir = os.path.join(root_dir, "jpg")
    name_imgs = [p for p in os.listdir(data_dir) if p.endswith(".jpg")]
    path_imgs = [os.path.join(data_dir, name) for name in name_imgs]
    random.seed(random_seed)
    random.shuffle(path_imgs)
    print(path_imgs[0])

    # 2. random划分获得3个list
    train_breakpoints = int(len(path_imgs)*train_ratio)
    valid_breakpoints = int(len(path_imgs)*(train_ratio + valid_ratio))
    train_imgs = path_imgs[:train_breakpoints]
    valid_imgs = path_imgs[train_breakpoints:valid_breakpoints]
    test_imgs = path_imgs[valid_breakpoints:]

    # 3. 复制
    move_img(train_imgs, root_dir, "train")
    move_img(valid_imgs, root_dir, "valid")
    move_img(test_imgs, root_dir, "test")


r"""
train dataset, copy 6551 imgs to G:\deep_learning_data\102flowers\train
valid dataset, copy 819 imgs to G:\deep_learning_data\102flowers\valid
test dataset, copy 819 imgs to G:\deep_learning_data\102flowers\test
"""
