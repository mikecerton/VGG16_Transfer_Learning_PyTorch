# VGG16
VGG16 architecture in pytorch and VGG16 pre-trained weight from torch vision
### Directory Structure
    .
    |-- images
    |    |-- test
    |    |      |-- class1
    |    |      |       |-- pic1.png
    |    |      |       |-- pic2.png
    |    |      |       ...
    |    |      |-- class2
    |    |      ...
    |    |-- train                  # Training images organized similarly to the test directory
    |    |      ...
    |    |-- validation             # Validation images organized similarly to the test directory
    |           ...
    |-- VGG16_Model.py
    |-- get_weight_torchvi.py       # get pretrain VGG16 weight from torch vision
    |-- VGG16_pre_weight.pt         
    |-- .gitignore
    |-- create_indexFile.py
    |-- dataset.py
    |-- train.py
    |-- README.md

