"""
Creates a Pytorch dataset to load the Pascal VOC dataset ####
"""

import torch
import os
import pandas as pd
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import numpy as np

class VOCDataset(torch.utils.data.Dataset):
    def __init__(
        self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)

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
        image = np.array(image)
        boxes = torch.tensor(boxes)

        if len(boxes) != 0:
            boxes = np.roll(boxes, 4, axis=1).tolist() # FOR AUGMENTATION ADDED
        
        if self.transform:
            # image = self.transform(image) # FOR AUGMENTATION REMOVED
            # AUGMENTATION ADDED (THE FOLLOWING THREE LINES):
            augmentations = self.transform(image=image, bboxes=boxes)
            image = augmentations["image"]
            boxes = augmentations["bboxes"]

        # Convert To Cells
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        
        # After the transformation it's possible, that all labeled objects got cutted off. 
        # In this case, the label_matrix should stay torch.zeros
        if len(boxes) != 0: # FOR AUGMENTATION ADDED

            boxes = np.roll(boxes, 1, axis=1).tolist() # FOR AUGMENTATION ADDED

            for box in boxes:
                #class_label, x, y, width, height = box.tolist() # FOR AUGMENTATION REMOVED
                class_label, x, y, width, height = box # FOR AUGMENTATION ADDED
                class_label = int(class_label)

                # i,j represents the cell row and cell column
                i, j = int(self.S * y), int(self.S * x)
                x_cell, y_cell = self.S * x - j, self.S * y - i

                """
                Calculating the width and height of cell of bounding box,
                relative to the cell is done by the following, with
                width as the example:
                
                width_pixels = (width*self.image_width)
                cell_pixels = (self.image_width)
                
                Then to find the width relative to the cell is simply:
                width_pixels/cell_pixels, simplification leads to the
                formulas below.
                """
                width_cell, height_cell = (
                    width * self.S,
                    height * self.S,
                )

                # If no object already found for specific cell i,j
                # Note: This means we restrict to ONE object
                # per cell!
                if label_matrix[i, j, 20] == 0:
                    # Set that there exists an object
                    label_matrix[i, j, 20] = 1

                    # Box coordinates
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )

                    label_matrix[i, j, 21:25] = box_coordinates

                    # Set one hot encoding for class_label
                    label_matrix[i, j, class_label] = 1

        return image, label_matrix


class Other_Dataset(torch.utils.data.Dataset):
    def __init__(
        self, csv_file, img_dir, label_dir, S=7, B=2, C=1, transform=None, # These hyperparameters might be useful to change
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = 1, 0, 0, 0, 0
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(float(x))
                    for x in label.replace("\n", "").split()
                ]

                boxes.append([class_label, x, y, width, height])

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path) # Wo liegt der Unterschied zwischen dem Einlesen von png und jpg?
        image = np.array(image) # Added for Augmentation
        boxes = torch.tensor(boxes)
        if len(boxes) != 0:
            boxes = np.roll(boxes, 4, axis=1).tolist() # FOR AUGMENTATION ADDED
        
        if self.transform:
            # image = self.transform(image)
            # image, boxes = self.transform(image, boxes)
            # AUGMENTATION ADDED (THE FOLLOWING THREE LINES):
            if boxes != []:
                augmentations = self.transform(image=image, bboxes=boxes)
            image = augmentations["image"]
            boxes = augmentations["bboxes"]

        # Convert To Cells
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))

        # After the transformation it's possible, that all labeled objects got cutted off. 
        # In this case, the label_matrix should stay torch.zeros
        if len(boxes) != 0: # FOR AUGMENTATION ADDED

            boxes = np.roll(boxes, 1, axis=1) # FOR AUGMENTATION ADDED

            for box in boxes:
                # class_label, x, y, width, height = box.tolist() # FOR AUGMENTATION REMOVED
                class_label, x, y, width, height = box # FOR AUGMENTATION ADDED
                class_label = int(class_label)

                # i,j represents the cell row and cell column
                i, j = int(self.S * y), int(self.S * x)
                x_cell, y_cell = self.S * x - j, self.S * y - i

                """
                Calculating the width and height of cell of bounding box,
                relative to the cell is done by the following, with
                width as the example:
                
                width_pixels = (width*self.image_width)
                cell_pixels = (self.image_width)
                
                Then to find the width relative to the cell is simply:
                width_pixels/cell_pixels, simplification leads to the
                formulas below.
                """
                width_cell, height_cell = (
                    width * self.S,
                    height * self.S,
                )

                # If no object already found for specific cell i,j
                # Note: This means we restrict to ONE object
                # per cell!
                if label_matrix[i, j, self.C] == 0:
                    # Set that there exists an object
                    label_matrix[i, j, self.C] = 1

                    # Box coordinates
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )

                    label_matrix[i, j, (self.C+1):(self.C+5)] = box_coordinates
                
                    # Set one hot encoding for class_label
                    label_matrix[i, j, class_label] = 1

        return image, label_matrix


