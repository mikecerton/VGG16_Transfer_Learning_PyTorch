# VGG16
This project utilizes transfer learning with the VGG16 model, leveraging pre-trained weights from torchvision to perform classification tasks on a custom dataset.
While working on this project, I used Python 3.10.11. Don't forget to install PyTorch for your hardware. It will be easier to understand the entire code if you review the code in the vgg16-transfer-learning-pytorch.ipynb notebook.
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

