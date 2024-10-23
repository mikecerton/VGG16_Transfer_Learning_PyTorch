# VGG16
&emsp;This project utilizes transfer learning with the VGG16 model, leveraging pre-trained weights from torchvision to perform classification tasks on a custom dataset.
While working on this project, I used Python 3.10.11. Don't forget to install PyTorch for your hardware. It will be easier to understand the entire code if you review the code in the vgg16-transfer-learning-pytorch.ipynb notebook.

<img src="https://media.geeksforgeeks.org/wp-content/uploads/20200219152207/new41.jpg" alt="diagram" width="800" />
picture from https://media.geeksforgeeks.org/wp-content/uploads/20200219152207/new41.jpg

### Quick start
- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- put dataset in image directory
- get VGG16 weight by get_weight_torchvi.py
- creat indexfile.csv by create_indexFile.py
- execute train.py of vgg16-transfer-learning-pytorch.ipynb

### Dataset
While developing this project, I used the dataset from Kaggle: https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset. However, you are free to use any dataset of your choice. Just make sure to place the dataset in the directory structure as I have shown in the Directory Structure.
    
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
    |-- requirements.txt

### Disclaimer
 - The code in this repository is modified from this Kaggle notebook https://www.kaggle.com/code/vortexkol/vgg16-pre-trained-architecture-beginner#If-you-found-this-kernel-informative-Please-do-upvote.
 - I used the dataset from this Kaggle dataset https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset, but you are free to use any dataset of your choice as well.

