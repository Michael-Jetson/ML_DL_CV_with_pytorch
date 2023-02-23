# -*- coding: utf-8 -*-
"""
# @brief      : 将错误分类的图片挑出来，进行观察
"""
import os
import pickle
import shutil


def load_pickle(path_file):
    with open(path_file, "rb") as f:
        data = pickle.load(f)
    return data


def my_mkdir(my_dir):
    if not os.path.isdir(my_dir):
        os.makedirs(my_dir)


if __name__ == '__main__':
    path_pkl = r"F:\prj_class\results\04-23_23-46\error_imgs_best.pkl"
    data_root_dir = r"G:\deep_learning_data\flowers102\jpg"
    out_dir = path_pkl[:-4]  # 输出文件目录
    error_info = load_pickle(path_pkl)

    for setname, info in error_info.items():
        for imgs_data in info:
            label, pred, path_img_rel = imgs_data
            path_img = os.path.join(data_root_dir, os.path.basename(path_img_rel))
            img_dir = os.path.join(out_dir, setname, str(label), str(pred))     # 图片文件夹
            my_mkdir(img_dir)
            shutil.copy(path_img, img_dir)      # 复制文件






