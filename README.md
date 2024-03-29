# make-voc-dataset

[![PyPI version](https://badge.fury.io/py/make-voc-dataset.svg)](https://badge.fury.io/py/make-voc-dataset) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Downloads](https://pepy.tech/badge/make-voc-dataset)](https://pepy.tech/project/make-voc-dataset) [![Downloads](https://pepy.tech/badge/make-voc-dataset/month)](https://pepy.tech/project/make-voc-dataset) [![Downloads](https://pepy.tech/badge/make-voc-dataset/week)](https://pepy.tech/project/make-voc-dataset)


Majority of the current Deep Learning Frameworks like [MMDetection](https://github.com/open-mmlab/mmdetection) or [Detectron2](https://github.com/facebookresearch/detectron2) support the **VOC Formatted Data / COCO Formatted Data**

This simple tool helps convert files stored in your local machine in the VOC Formatted Directory Structure

The data should have images and their corresponding annotation_file in **xml** format.

Tools like [LabelIMG](https://github.com/tzutalin/labelImg) can be used to annotate images.

## Options

`-s / --source_path`
Provide the full path where all the images and xmls are located. Please make sure the img_names are **UNIQUE**

`-d / --dest_path`
Provide the full to save the VOC structure

`-t / --test_ratio`
Provide the ratio (between 0 and 1) that will be the test division of the dataset
