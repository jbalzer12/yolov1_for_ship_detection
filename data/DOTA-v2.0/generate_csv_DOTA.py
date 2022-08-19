########################################################
## Author: Josefina Balzer
########################################################
## Comment: This file will only be executed locally.
## Description: With this file the csv files get written.
##              These files are required to link every 
##              image to the matching label file.             
########################################################
import os
from os.path import isfile, join
import csv

# These file paths link to the downloaded dataset files which might differ in someone elses installation
file_path_images_train = 'train/images'
file_path_images_val = 'val/images/part2'

# This array contains all train image names
read_train = [f for f in os.listdir(file_path_images_train) if isfile(join(file_path_images_train, f))]
# The the train.csv gets written
with open('train.csv', 'a') as train_file:
    for elem in read_train:
        if elem != '.DS_Store':
            train_file.write(elem + ',' + elem.replace('png', 'txt') + '\n')
    train_file.close()

# This array contains all test image names
read_val = [f for f in os.listdir(file_path_images_val) if isfile(join(file_path_images_val, f))]
# The the train.csv gets written
with open('val.csv', 'a') as val_file:
    for elem in read_val:
        if elem != '.DS_Store':
            val_file.write(elem + ',' + elem.replace('png', 'txt') + '\n')
    val_file.close()


