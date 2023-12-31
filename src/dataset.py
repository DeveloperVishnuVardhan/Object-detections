"""
Karan Shah and Jyothi Vishnu
CS-7180 Fall 2023 semester
Creates a Pytorch dataset to load the Pascal VOC dataset
"""

import torch
import os
import pandas as pd
from PIL import Image

class VOCDataset(torch.utils.data.Dataset):
    """
    Build a torch dataset for VOC data
    """

    # Initialize annotations, image dir, label dir, 
    # grid size, number of BB, and number of classes
    def __init__(self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    # Obtain the length of the annotations
    def __len__(self):
        return len(self.annotations)
    
    # Label and image preprocessing for single example
    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]

                boxes.append([class_label, x, y, width, height])

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        # Convert to cells
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B)) # (7, 7, 30) but since target matrix we don't use last 5
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            # i, j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x) # 3, 4 for 0.000007.jpg
            x_cell, y_cell = self.S * x - j, self.S * y - i 
            width_cell, height_cell = (width * self.S, height * self.S)


            # If no object already found for specific cell i,j
            # Note: This means we restrict to ONE object
            # per cell!
            if label_matrix[i, j, 20] == 0:
                # Set that there exists an object
                label_matrix[i, j, 20] = 1

                # Box Coordinates 
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, 21:25] = box_coordinates

                # Set one hot encoding for class label (first 20 values)
                label_matrix[i, j, class_label] = 1

        return image, label_matrix