# make_voc_dataset
Majority of the current Deep Learning Frameworks support the **VOC Formatted Data / COCO Formatted Data** 
This Repository helps convert files stored in your local machine in the VOC Formatted Directory Structure

This Repository uses numpy, tqdm and _mmcv_
If you don't have them installed
I'd recommend you perform 
```
pip install numpy
pip install tqdm
pip install mmcv-full
```

The way to use this file is shown below
```
python make_voc_dataset.py --help

usage: make_voc_dataset.py [-h] [--base_path BASE_PATH]
                           [--dest_path DEST_PATH] [--test_ratio TEST_RATIO]

optional arguments:
  -h, --help            show this help message and exit
  --base_path BASE_PATH
                        Provide the full path where all the images and xmls
                        are located. Please make sure the img_names are UNIQUE
  --dest_path DEST_PATH
                        Provide the full to save the voc structure
  --test_ratio TEST_RATIO
                        Provide the ratio (between 0 and 1) that will be the
                        test division of the dataset
```
