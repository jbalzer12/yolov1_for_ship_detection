"""
Main file for training Yolo model on Pascal VOC dataset

"""
import os
import argparse
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import VOCDataset
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
    parse_cfg,
)
from loss import YoloLoss

from datetime import datetime as dt

seed = 123
torch.manual_seed(seed)

parser = argparse.ArgumentParser(description='YOLOv1-pytorch')
parser.add_argument("--cfg", "-c", default="cfg/VOC/yolov1.yaml", help="Yolov1 config file path", type=str)
parser.add_argument("--dataset_cfg", "-d", default="cfg/VOC/dataset.yaml", help="Dataset config file path", type=str)
parser.add_argument("--epochs", "-e", default=135, help="Training epochs", type=int)
parser.add_argument("--batch_size", "-bs", default=64, help="Training batch size", type=int)
parser.add_argument("--lr", "-lr", default=5e-4, help="Training learning rate", type=float)
parser.add_argument("--load_model", "-lm", default='False', help="Load Model or train one [ 'True' | 'False' ]", type=str)  
parser.add_argument("--model_path", "-mp", default="/scratch/tmp/jbalzer/yolov1/overfit_VOC_new.pth.tar", help="Model path", type=str)

args = parser.parse_args()

# Hyperparameters etc. 
LEARNING_RATE = args.lr
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = args.batch_size # 64 in original paper but I don't have that much vram, grad accum?
WEIGHT_DECAY = 0.0005
EPOCHS = args.epochs
NUM_WORKERS = 15
PIN_MEMORY = True
if args.load_model == 'True':
    LOAD_MODEL = True # DEFAULT MODUS: training
elif args.load_model == 'False':
    LOAD_MODEL = False
LOAD_MODEL_FILE = args.model_path

if not(LOAD_MODEL): 
    # OUTPUT = open('/scratch/tmp/jbalzer/yolov1/output_SDG.txt', 'w') # HDF5 anstelle von .txt?
    OUTPUT = open('output_VOC_new.txt', 'w') # HDF5 anstelle von .txt?
    OUTPUT.write('Train_mAP Mean_loss\n')

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])

# Training function
def train_fn(train_loader, model, optimizer, loss_fn):
    # Setting up a progress bar 
    loop = tqdm(train_loader, leave=True) 
    mean_loss = []


    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update progress bar
        loop.set_postfix(loss=loss.item())

    #OUTPUT.write(f"Mean loss was: {sum(mean_loss)/len(mean_loss)}\n")
    OUTPUT.write(f' {sum(mean_loss)/len(mean_loss)}\n')
    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")


def main():

    print('Working Device: ', DEVICE)    
    cfg = parse_cfg(args.cfg)
    S, B, num_classes, input_size = cfg['S'], cfg['B'], cfg['num_classes'], cfg['input_size']
    dataset_cfg = parse_cfg(args.dataset_cfg)
    IMG_DIR, LABEL_DIR, CLASS_NAMES = dataset_cfg['images'], dataset_cfg['labels'], dataset_cfg['class_names']

    # Initialize the model and move it to device
    model = Yolov1(split_size=S, num_boxes=B, num_classes=num_classes).to(DEVICE)
    
    # Learning rate schedular 
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss()

    # In case a model gets loaded the checkpoint gets loaded
    if LOAD_MODEL:
        a = torch.load(LOAD_MODEL_FILE, map_location=torch.device('cpu'))
        load_checkpoint(a, model, optimizer)

    train_dataset = VOCDataset(
        #"data/VOC2007_2012/100examples.csv",
        #"/scratch/tmp/jbalzer/data/VOC2007_2012/train.csv",
        "data/VOC2007_2012/train.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    ) 

    test_dataset = VOCDataset(
        #"/scratch/tmp/jbalzer/data/VOC2007_2012/test.csv", 
        "data/VOC2007_2012/test.csv", 
        transform=transform, 
        img_dir=IMG_DIR, 
        label_dir=LABEL_DIR,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,  
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    for epoch in range(EPOCHS):

        # Only required if a model gets trained
        if not(LOAD_MODEL): 
            now = dt.now().strftime("%d/%m/%Y, %H:%M:%S")
            print("epoch:", epoch, f"/ {args.epochs} =>", epoch / args.epochs * 100, "%, date/time:", now)    
            # file to check the progress
            #NUM_EPOCH = open('/scratch/tmp/jbalzer/yolov1/numberofepochs_VOC_new.txt', 'w') 
            #NUM_EPOCH = open('numberofepochs_VOC_new.txt', 'w') 
            #NUM_EPOCH.write("epoch:" + str(epoch) + f"/ {args.epochs} =>" + str(epoch / args.epochs * 100) + "%, date/time:" + str(now))
            #NUM_EPOCH.close()

        # In case a model gets loaded, images will be used by the model
        if LOAD_MODEL:
            
            '''from PIL import Image
            img = Image.open('/Users/josefinabalzer/Desktop/SS22/BA/yolov1_for_ship_detection/data/VOC2007_2012/images/000050.jpg')
            import numpy as np 
            img = np.array(img)
            img = torch.from_numpy(img)
            img = img.permute(2,0,1)
            img = img.reshape((1,3,375,500))
            model_img = model(img)
            torch.save(model_img, 'single_img/model.pt')
            bboxes_img = cellboxes_to_boxes(model_img, S, B, num_classes)
            torch.save(bboxes_img, 'single_img/bboxes.pt')
            
            exit()'''
            
            #for x, y in train_loader:
            for x, y in test_loader:
                with torch.no_grad():
                    model.eval()
                x = x.to(DEVICE)
                for idx in range(8):
                    start_idx = 0
                    m = model(x)
                    bboxes = cellboxes_to_boxes(m, S, B, num_classes)

                    # average precision:


                    bboxes = non_max_suppression(bboxes[idx+start_idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
                    print('plot VOC')

                    plot_image(x[idx+start_idx].permute(1,2,0).to("cpu"), bboxes, CLASS_NAMES)

                import sys
                sys.exit()

        pred_boxes, target_boxes = get_bboxes(
            test_loader, model, iou_threshold=0.5, threshold=0.4
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=num_classes
        )

        #OUTPUT.write(f"Train mAP: {mean_avg_prec}\n")
        OUTPUT.write(f'{mean_avg_prec}')
        print(f"Train mAP: {mean_avg_prec}")

        # The training function gets called
        train_fn(train_loader, model, optimizer, loss_fn)

    OUTPUT.close()
    
    # the following lines were added to make sure the procession stops after the full range of epochs and not 
    # at the point of a specific mean average precision 
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
    import sys
    sys.exit()
    exit()


if __name__ == "__main__":
    main()
