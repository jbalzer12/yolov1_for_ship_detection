######################################################
## This file writes all train and validation files 
## (.jpg and .txt) int oa csv.
## I used it, after I generated label files and separated
## the training data into two datasets: one for training
## and one for validation (90 : 10)
######################################################
## Author: Josefina Balzer
######################################################

from os import listdir
from os.path import isfile, join

train_images_path = 'train/images'
val_images_path = 'val/images'


train_file = open("train.csv", "a")
train_images = [f for f in listdir(train_images_path) if isfile(join(train_images_path, f))]

for i in range(len(train_images)):
    image_id = train_images[i].removesuffix('.jpg')
    train_file.write(image_id + '.jpg,' + image_id + '.txt\n')

train_file.close()

val_file = open("val.csv", "a")
val_images = [f for f in listdir(val_images_path) if isfile(join(val_images_path, f))]

for i in range(len(val_images)):
    image_id = val_images[i].removesuffix('.jpg')
    val_file.write(image_id + '.jpg,' + image_id + '.txt\n')

val_file.close()