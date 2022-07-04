"""
Main file for training Yolo model on airbus-ship-detection dataset 
(source for dataset: https://www.kaggle.com/competitions/airbus-ship-detection/data)

"""
import os
import argparse
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model_modified import Yolov1
#from model import Yolov1
from dataset import (
    Other_Dataset,
)
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
parser.add_argument("--cfg", "-c", default="cfg/airbus-ship-detection/yolov1.yaml", help="Yolov1 config file path", type=str)
parser.add_argument("--dataset_cfg", "-d", default="cfg/airbus-ship-detection/dataset.yaml", help="Dataset config file path", type=str)
parser.add_argument("--epochs", "-e", default=135, help="Training epochs", type=int)
parser.add_argument("--batch_size", "-bs", default=64, help="Training batch size", type=int)
parser.add_argument("--lr", "-lr", default=5e-4, help="Training learning rate", type=float)
parser.add_argument("--load_model", "-lm", default='False', help="Load Model or train one [ 'True' | 'False' ]", type=str)  
parser.add_argument("--model_path", "-mp", default="/scratch/tmp/jbalzer/yolov1/overfit_airbus_500_896_resolution_B_14.pth.tar", help="Model path", type=str)

args = parser.parse_args()

# Hyperparameters etc. 
LEARNING_RATE = args.lr
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = args.batch_size # 64 in original paper but I don't have that much vram, grad accum?
WEIGHT_DECAY = 0.0005
EPOCHS = args.epochs
NUM_WORKERS = 63
PIN_MEMORY = True
if args.load_model == 'True':
    LOAD_MODEL = True # DEFAULT MODUS: training
elif args.load_model == 'False':
    LOAD_MODEL = False
LOAD_MODEL_FILE = args.model_path

if not(LOAD_MODEL): 
    #OUTPUT = open('output_airbus_135.txt', 'w') # HDF5 anstelle von .txt?
    OUTPUT = open('/scratch/tmp/jbalzer/yolov1/output_airbus_500_896_resolution_B_14.txt', 'a') # HDF5 anstelle von .txt?
    #OUTPUT = open('output_airbus_135_train.txt', 'a')
    OUTPUT.write('Train_mAP Test_mAP Mean_loss\n')
    OUTPUT.close()

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


transform = Compose([transforms.Resize((896, 896)), transforms.ToTensor()]) # changed 448 to 896 (448 * 2)

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

    OUTPUT = open('/scratch/tmp/jbalzer/yolov1/output_airbus_500_896_resolution_B_14.txt', 'a')
    OUTPUT.write(f" {sum(mean_loss)/len(mean_loss)}\n")
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
    loss_fn = YoloLoss(S=S, B=B, C=num_classes)

    # In case a model gets loaded the checkpoint gets loaded
    if LOAD_MODEL:
        a = torch.load(LOAD_MODEL_FILE, map_location=DEVICE) # torch.device('cpu'))
        load_checkpoint(a, model, optimizer)


    train_dataset = Other_Dataset(
        "/scratch/tmp/jbalzer/data/airbus-ship-detection/train.csv",
        #"data/airbus-ship-detection/train.csv",
        #"data/DOTA-v2.0/train.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        S=S,
        B=B,
        C=num_classes,
    )

    test_dataset = Other_Dataset(
        "/scratch/tmp/jbalzer/data/airbus-ship-detection/val.csv", 
        #"data/airbus-ship-detection/val-smaller.csv",
        #"data/DOTA-v2.0/val.csv",
        transform=transform, 
        img_dir=IMG_DIR, 
        label_dir=LABEL_DIR,
        S=S,
        B=B,
        C=num_classes,
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
            OUTPUT = open('/scratch/tmp/jbalzer/yolov1/output_airbus_500_896_resolution_B_14.txt', 'a') # HDF5 anstelle von .txt?

        # In case a model gets loaded, images will be used by the model
        if LOAD_MODEL:
            # x contains the image while y contains the label matrix 
            for x, y in test_loader:
                x = x.to(DEVICE)
                start_idx = 8
                for idx in range(8):
                    bboxes = cellboxes_to_boxes(model(x), S=S, B=B, C=num_classes)
                    bboxes = non_max_suppression(bboxes[idx+start_idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
                    plot_image(x[idx+start_idx].permute(1,2,0).to("cpu"), bboxes, CLASS_NAMES)

                import sys
                sys.exit()

        ###### VALIDATION BASED ON TRAINING DATA ######
        pred_boxes, target_boxes = get_bboxes(
            loader=train_loader, model=model, iou_threshold=0.5, threshold=0.4, S=S, B=B, C=num_classes,
        )
        train_mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=num_classes
        )

        print(f"Train mAP: {train_mean_avg_prec}")
        ###############################################

        ###### VALIDATION BASED ON TEST DATA ######
        pred_boxes, target_boxes = get_bboxes(
            loader=test_loader, model=model, iou_threshold=0.5, threshold=0.4, S=S, B=B, C=num_classes,
        )
        test_mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=num_classes
        )
        OUTPUT.write(f'{train_mean_avg_prec} {test_mean_avg_prec}')
        OUTPUT.close()
        print(f"Test mAP: {test_mean_avg_prec}")
        ###############################################
        
        # The training function gets called
        train_fn(train_loader, model, optimizer, loss_fn)
        
        ###### SAVES CHECKPOINT INBETWEEN ######
        checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
        ###############################################

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


if __name__ == "__main__":
    main()
