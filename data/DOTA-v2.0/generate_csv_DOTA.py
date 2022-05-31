import os
from os.path import isfile, join
import csv

file_path_images_train = 'train/images'
file_path_images_val = 'val/images/part2'

read_train = [f for f in os.listdir(file_path_images_train) if isfile(join(file_path_images_train, f))]

with open('train.csv', 'a') as train_file:
    for elem in read_train:
        if elem != '.DS_Store':
            train_file.write(elem + ',' + elem.replace('png', 'txt') + '\n')
    train_file.close()

read_val = [f for f in os.listdir(file_path_images_val) if isfile(join(file_path_images_val, f))]

with open('val.csv', 'a') as val_file:
    for elem in read_val:
        if elem != '.DS_Store':
            val_file.write(elem + ',' + elem.replace('png', 'txt') + '\n')
    val_file.close()


