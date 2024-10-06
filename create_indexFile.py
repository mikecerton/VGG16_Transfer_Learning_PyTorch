import os
import pandas as pd

# Define the folder paths for vegetables
folder_path_vegetables = '/kaggle/input/vegetable-image-dataset/Vegetable Images/train'

# List of dataset folders and their respective labels
vegetable_folders = [
    ('Bean', 0), 
    ('Bitter_Gourd', 1), 
    ('Bottle_Gourd', 2), 
    ('Brinjal', 3), 
    ('Broccoli', 4), 
    ('Cabbage', 5), 
    ('Capsicum', 6), 
    ('Carrot', 7), 
    ('Cauliflower', 8), 
    ('Cucumber', 9), 
    ('Papaya', 10), 
    ('Potato', 11), 
    ('Pumpkin', 12), 
    ('Radish', 13), 
    ('Tomato', 14)
]

# Collect the file paths and labels for all vegetable folders
all_data = []
for folder_name, label in vegetable_folders:
    folder_path = os.path.join(folder_path_vegetables, folder_name)
    files = os.listdir(folder_path)
    data = [(os.path.join(folder_path, file), label) for file in files]
    all_data.extend(data)

# print(all_data)

# Create a DataFrame
df = pd.DataFrame(all_data, columns=['Path', 'Label'])

df.to_csv("/kaggle/working/train_indexFile.csv", index=False, index_label=False)
