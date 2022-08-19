######################################################
## This file writes all train and validation files 
## (.jpg and .txt) into a csv.
## I used it, after I generated label files and separated
## the training data into two datasets: one for training
## and one for validation (90 : 10)
######################################################
## Author: Josefina Balzer
######################################################

import os 
from os import listdir
from os.path import isfile, join

train_images_path = 'train/images'
val_images_path = 'val/images'

train_labels_path = 'train/labels'
val_labels_path = 'val/labels'

# This function removes a label.txt file, if a corresponding .jpg file in the image folder does not exist:
def remove_label_files():
    # First we build an array that contains all the file names of our images:
    train_images = [f for f in listdir(train_images_path) if isfile(join(train_images_path, f))]
    # To compare the image names with the label names to find non existing image names as label files
    # we habe to remove the .jpg suffix:
    for i in range(len(train_images)):
        train_images[i] = train_images[i].removesuffix('.jpg')

    # Then we build an array for the label files names the same way as the train_images array:
    train_labels = [f for f in listdir(train_labels_path) if isfile(join(train_labels_path, f))]
    # It has to be tidied up the same way as the train_images array:
    for i in range(len(train_labels)):
        train_labels[i] = train_labels[i].removesuffix('.txt')

    # Now we are able to compare both arrays and remove the 'ghostly' label files:
    for i in range(len(train_labels)):
        if train_labels[i] != '.DS_Store':
            if not (train_labels[i] in train_images):
                os.remove('train/labels/' + train_labels[i] + '.txt')
                print('removed: ', train_labels[i])
    

    
# This function generates a .csv file that pairs the corresponding label.txt and image.jpg files.
# This file will be required by the yolo algorithm: 
def generate_csv():
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


if __name__ == "__main__":
    #remove_label_files()
    generate_csv()
