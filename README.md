# Deep Learning for Object Detection on High Resolution Satellite Imagery

Author: This repository contains code files by [Aladdin Persson](https://github.com/aladdinpersson/Machine-Learning-Collection) as well as files by Josefina Balzer (see [license](https://github.com/jbalzer12/yolov1_for_ship_detection/blob/main/LICENSE))

This repository is part of the Bachelor thesis by Josefina Balzer titled "Deep Learnig for Object Detection on High Resolution Satellite Imagery".
The idea was to improve YOLOv1 by [Redmon et al. (2016)](https://pjreddie.com/darknet/yolov1/) for small object detection. Escpecially small objects in satellite imagery. This work focuses on ship detection.

### Content
This repository used to train YOLOv1 for three different datasets: PASCAL VOC, Airbus Ship Detection and DOTA. Therefore it contains multiple train.py files, one for each dataset. The train.py file is the main file to train with each of the datasets.

Change paths in cfg files and download the requirements as well as the data to run the training it. Some parameters in the train.py files also have to be modified. In this code the option is given to run the train.py file locally or not. I used to work with an HPC cluster to run these files.

Link to Github Repository: https://github.com/jbalzer12/yolov1_for_ship_detection
