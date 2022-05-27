######################################################
## This file converts the rle encoded bounding boxes 
## into txt label files as the yolo algorithm asks for
######################################################
## Author: Josefina Balzer
######################################################

# Import of modules and functions:
import PIL 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import argparse
from utils import parse_cfg

parser = argparse.ArgumentParser(description='YOLOv1-pytorch')
parser.add_argument("--yolov1_cfg", "-yc", default="../../cfg/airbus-ship-detection/yolov1.yaml", help="Yolov1 config file path", type=str)
parser.add_argument("--dataset_cfg", "-d", default="../../cfg/airbus-ship-detection/dataset.yaml", help="Dataset config file path", type=str)
args = parser.parse_args()

# Read some parameters from .yaml files:
yolov1_cfg = parse_cfg(args.yolov1_cfg)
INPUT_SIZE = yolov1_cfg['input_size']
dataset_cfg = parse_cfg(args.dataset_cfg)
CLASS_NAMES = dataset_cfg['class names']


df = pd.read_csv("train_ship_segmentations_v2.csv", header=0)

# This counter will refer to the index of the df file to get the ImageId:
counter = 0

# For each element inside the given csv file we want to decode the labels given as rle code
# into bounding boxes in a format, the yolo algorithm asks for.
for i in df['EncodedPixels']:
   # We want to devide the rows with a NaN value in the EncodedPixels column and 
   # the oneswith a float value
   if type(i) != float: 
      elem = [int(j) for j in i.split()]
      pixels = [(pixel_position % INPUT_SIZE, pixel_position // INPUT_SIZE) 
                  for start, length in list(zip(elem[0:-1:2], elem[1::2])) 
                  for pixel_position in range(start, start + length)]
      
      # Now we want to separate the x and the y values:
      list_x = []
      list_y = []

      for (x,y) in pixels:
         list_x.append(x)
         list_y.append(y)

      # Find the min and max values in the x and y pixel coordinates:
      minx = min(list_x)
      miny = min(list_y)
      maxx = max(list_x)
      maxy = max(list_y)

      # Calculate the thd midpoint (x,y), width and height for the decoded bounding box:
      midpoint_x = int(minx + (maxx - minx) / 2)
      midpoint_x_yolo = midpoint_x / INPUT_SIZE
      midpoint_y = int(miny + (maxy - miny) / 2)
      midpoint_y_yolo = midpoint_y / INPUT_SIZE
      width = (maxy - miny) / INPUT_SIZE
      height = (maxx - minx) / INPUT_SIZE

      # Now the txt file gets written
      image_id = (df['ImageId'][counter]).removesuffix('.jpg')
      yolo_file = open('labels/' + image_id + '.txt', 'a')
      yolo_file.write(f'1 {midpoint_x_yolo} {midpoint_y_yolo} {width} {height}\n')
      yolo_file.close()

   else: 
      # An empty txt file gets written
      image_id = (df['ImageId'][counter]).removesuffix('.jpg')
      yolo_file = open('labels/' + image_id + '.txt', 'a')
      yolo_file.close()

   counter += 1

''' 
YOLO format:

<class> <midpoint x> <midpoint y> <width> <height>
...
<class> <midpoint x> <midpoint y> <width> <height>
'''
