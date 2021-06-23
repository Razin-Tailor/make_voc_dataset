#!/usr/bin/env python3
import os
import sys
import argparse
import mmcv
from tqdm import tqdm
import shutil
import numpy as np

"Required image path and the xmls path and tha path to save the voc_structure"


def create_blank_voc_dir_str(dest_path):
    mmcv.mkdir_or_exist(dest_path)
    mmcv.mkdir_or_exist(os.path.join(dest_path, "VOCdevkit"))
    mmcv.mkdir_or_exist(os.path.join(dest_path, "VOCdevkit", "VOC2007"))
    mmcv.mkdir_or_exist(os.path.join(dest_path, "VOCdevkit", "VOC2007", "Annotations"))
    mmcv.mkdir_or_exist(os.path.join(dest_path, "VOCdevkit", "VOC2007", "ImageSets"))
    mmcv.mkdir_or_exist(os.path.join(dest_path, "VOCdevkit", "VOC2007", "JPEGImages"))


def make_voc_dataset(opt):
    file_list = list()

    base_path = opt.base_path
    dest_path = opt.dest_path
    create_blank_voc_dir_str(dest_path)

    dest_annotation_path = os.path.join(
        dest_path, "VOCdevkit", "VOC2007", "Annotations"
    )
    dest_jpeg_path = os.path.join(dest_path, "VOCdevkit", "VOC2007", "JPEGImages")
    dest_imgset_path = os.path.join(dest_path, "VOCdevkit", "VOC2007", "ImageSets")

    ratio = opt.test_ratio

    for (dirpath, dirnames, filenames) in os.walk(base_path):
        file_list += [os.path.join(dirpath, file) for file in filenames]

    images = [x for x in file_list if ".jpg" in x]
    xmls = [x.replace(".jpg", ".xml") for x in images]

    print("copying images")

    for img_path in tqdm(images):
        src = img_path
        dest = os.path.join(dest_jpeg_path, img_path.split("/")[-1])
        shutil.copyfile(src, dest)

    print("copying xmls")

    for xml_path in tqdm(xmls):
        src = xml_path
        dest = os.path.join(dest_annotation_path, xml_path.split("/")[-1])
        shutil.copyfile(src, dest)

    total_images = len(images)
    test_ratio = int(ratio * total_images)

    img_names = [x.split("/")[-1].split(".")[0] for x in images]
    img_names = np.random.permutation(img_names)
    test_images = img_names[:test_ratio]
    train_images = img_names[test_ratio:]
    print("Working on Train Test Split")

    with open(os.path.join(dest_imgset_path, "trainval.txt"), "w") as tvf:
        for img_name in train_images:
            tvf.write(img_name + "\n")

    with open(os.path.join(dest_imgset_path, "test.txt"), "w") as tf:
        for img_name in test_images:
            tf.write(img_name + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_path",
        type=str,
        default="./",
        help="Provide the full path where all the images and xmls are located. Please make sure the img_names are UNIQUE",
    )

    parser.add_argument(
        "--dest_path",
        type=str,
        default="./",
        help="Provide the full to save the voc structure",
    )

    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.10,
        help="Provide the ratio (between 0 and 1) that will be the test division of the dataset",
    )

    opt = parser.parse_args()
    make_voc_dataset(opt)


if __name__ == "__main__":
    exit(main())
