########################################################
## Author: Josefina Balzer
########################################################
## Comment: This file will only be executed locally.
## Description: With this file the labels from DOTA-v2.0
##              become rewritten in a way, yolo can read 
##              them.
########################################################

import os 
from os.path import isfile, join
from os import listdir
from PIL import Image 

# added to prevent error message 
Image.MAX_IMAGE_PIXELS = 1000000000

# multiple file paths
file_path_images = 'images'
file_path_labels = 'labels/'

file_path_label_Txt_train = 'train/labelTxt-v2.0/DOTA-v2.0_train_hbb'
file_path_label_Txt_val = 'val/labelTxt-v2.0/DOTA-v2.0_val_hbb'

CLASS_NAMES = [ 'plane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court', 'basketball-court', 
                'ground-track-field', 'harbor', 'bridge', 'large-vehicle', 'small-vehicle', 'helicopter', 
                'roundabout', 'soccer-ball-field', 'swimming-pool', 'container-crane', 'airport', 'helipad' ]

all_images = [f for f in listdir(file_path_images) if (isfile(join(file_path_images, f)) and f != '.DS_Store')]

# Start loop from here
for i in range(len(all_images)):

    current_img = Image.open(file_path_images + '/' + all_images[i])
    img_width, img_height = current_img.size

    label_Txt = open('labelTxt_all/' + all_images[i].removesuffix('.jpg') + '.txt').readlines()

    objects = []
    for line in label_Txt:
        objects.append(line.split())

    output_file = open(file_path_labels + all_images[i].removesuffix('.jpg') + '.txt', 'a')

    for object in objects:
        x1, y1, x2, y2, x3, y3, x4, y4, class_name, difficult = object

        x1 = float(x1)
        x2 = float(x2)
        x3 = float(x3)
        x4 = float(x4)
        y1 = float(y1)
        y2 = float(y2)
        y3 = float(y3)
        y4 = float(y4)

        class_idx = CLASS_NAMES.index(class_name)

        lower_left = (x1, y1)
        upper_left= (x2, y2)
        upper_right = (x3, y3)
        lower_right = (x4, y4)

        #center_x = (y1 + (y4 - y1) / 2) / img_width
        #center_y = (x1 + (x2 - x1) / 2) / img_height
        center_x = (x1 + ((x2 - x1) / 2)) / img_width
        center_y = (y1 + ((y3 - y1) / 2)) / img_height
        
        height = (y3 - y2) / img_height
        width = (x2 - x1) / img_width

        # For unknown reason in the original label files are ground truth boxes which do not fit into the matching images 
        # file and lie on the outside.
        # These points get sorted out:
        if center_x <= 1 and center_y <=1 and width <= 1 and height <= 1:
            outputline = f'{class_idx} {center_x} {center_y} {width} {height}\n'
            output_file.write(outputline)

output_file.close()
