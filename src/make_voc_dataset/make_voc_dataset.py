#!/usr/bin/env python3
import os
import sys
import argparse
import mmcv
from tqdm import tqdm
import shutil
import numpy as np
import logging

logging.basicConfig(
    format="[ %(asctime)s ] -- %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p"
)

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

    base_path = opt.source_path
    dest_path = opt.dest_path

    dest_annotation_path = os.path.join(
        dest_path, "VOCdevkit", "VOC2007", "Annotations"
    )
    dest_jpeg_path = os.path.join(dest_path, "VOCdevkit", "VOC2007", "JPEGImages")
    dest_imgset_path = os.path.join(dest_path, "VOCdevkit", "VOC2007", "ImageSets")

    ratio = opt.test_ratio

    for (dirpath, dirnames, filenames) in os.walk(base_path):
        file_list += [os.path.join(dirpath, file) for file in filenames]

    images = [x for x in file_list if ".jpg" in x]
    xmls = []
    for img in images:
        xml_name = img.replace(".jpg", ".xml")
        if os.path.isfile(xml_name):
            xmls.append(xml_name)

    logging.info(f"Number of images found: {len(images)}")
    logging.info(f"Number of xmls found: {len(xmls)}")
    if len(images) == 0:
        logging.critical("No Images found!")
        sys.exit(1)
    if len(xmls) == 0:
        logging.critical("No XMLs found!")
        sys.exit(1)
    if len(xmls) != len(images):
        logging.critical("Number of Images and XMLs Do not Match!")
        sys.exit(1)

    logging.info("Found images and xmls.")

    logging.info("Creating empty VOC directory skeleton")
    create_blank_voc_dir_str(dest_path)

    print(" Copying Images ".center(80, "*"))

    for img_path in tqdm(images):
        src = img_path
        dest = os.path.join(dest_jpeg_path, img_path.split("/")[-1])
        shutil.copyfile(src, dest)

    print(" Copying XMLs ".center(80, "*"))

    missed_xmls = 0
    for xml_path in tqdm(xmls):
        src = xml_path
        if not os.path.exists(src):
            logging.critical(f"Missing xml: ({src.split('/')[-1]})")
            missed_xmls += 1
        else:
            dest = os.path.join(dest_annotation_path, xml_path.split("/")[-1])
            shutil.copyfile(src, dest)
    if missed_xmls > 0:
        logging.critical(f"Number of missing xmls: {missed_xmls}")
        sys.exit(1)
    else:
        total_images = len(images)
        test_ratio = int(ratio * total_images)

        img_names = [x.split("/")[-1].split(".")[0] for x in images]
        img_names = np.random.permutation(img_names)
        test_images = img_names[:test_ratio]
        train_images = img_names[test_ratio:]
        print(" Working on Train Test Split ".center(80, "*"))

        with open(os.path.join(dest_imgset_path, "trainval.txt"), "w") as tvf:
            for img_name in train_images:
                tvf.write(img_name + "\n")

        with open(os.path.join(dest_imgset_path, "test.txt"), "w") as tf:
            for img_name in test_images:
                tf.write(img_name + "\n")
        print(" Dataset Creation Complete ".center(80, "*"))


def main() -> int:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-s",
        "--source_path",
        type=str,
        help="Provide the full path where all the images and xmls are located. Please make sure the img_names are UNIQUE",
    )

    parser.add_argument(
        "-d",
        "--dest_path",
        type=str,
        default="./",
        help="Provide the full to save the voc structure",
    )

    parser.add_argument(
        "-t",
        "--test_ratio",
        type=float,
        default=0.10,
        help="Provide the ratio (between 0 and 1) that will be the test division of the dataset",
    )

    opt = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        return 1
    else:
        make_voc_dataset(opt)
    return 0


if __name__ == "__main__":
    exit(main())
