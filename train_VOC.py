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
#from dataset_augmented import (VOCDataset, Other_Dataset) ##### changed
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
    mAP_on_object_size
)
from loss import YoloLoss
import augmentation

from datetime import datetime as dt

RUN_LOCAL = False

seed = 123
torch.manual_seed(seed)

parser = argparse.ArgumentParser(description='YOLOv1-pytorch')
parser.add_argument("--cfg", "-c", default="cfg/VOC/yolov1.yaml", help="Yolov1 config file path", type=str)
parser.add_argument("--dataset_cfg", "-d", default="cfg/VOC/dataset.yaml", help="Dataset config file path", type=str)
parser.add_argument("--epochs", "-e", default=135, help="Training epochs", type=int)
parser.add_argument("--batch_size", "-bs", default=64, help="Training batch size", type=int)
parser.add_argument("--lr", "-lr", default=5e-4, help="Training learning rate", type=float)
parser.add_argument("--load_model", "-lm", default='False', help="Load Model or train one [ 'True' | 'False' ]", type=str)  
parser.add_argument("--model_path", "-mp", default="/scratch/tmp/jbalzer/yolov1/overfit_VOC_500_validated_dropout_0_5_augmentation_added_small_obj_evaluation.pth.tar", help="Model path", type=str)

args = parser.parse_args()

# Hyperparameters etc. 
LEARNING_RATE = args.lr
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = args.batch_size # 64 in original paper but I don't have that much vram, grad accum?
WEIGHT_DECAY = 0.0005
EPOCHS = args.epochs
if not RUN_LOCAL:
    NUM_WORKERS = 63
elif RUN_LOCAL:
    NUM_WORKERS = 2
PIN_MEMORY = True
if args.load_model == 'True':
    LOAD_MODEL = True # DEFAULT MODUS: training
elif args.load_model == 'False':
    LOAD_MODEL = False
LOAD_MODEL_FILE = args.model_path

if not LOAD_MODEL: 
    if RUN_LOCAL:
        OUTPUT = open('output_VOC_VOCDataset_test.txt', 'w')
    elif not RUN_LOCAL:
        OUTPUT = open('/scratch/tmp/jbalzer/yolov1/output_VOC_500_validated_on_test_dropout_0_5_augmentation_added_small_obj_evaluation.txt', 'w') # HDF5 anstelle von .txt?
    OUTPUT.write('Train_mAP_on_0.58 Test_mAP_on_0.58 Train_mAP_on_1.0 Test_mAP_on_1.0 Train_mAP_on_5.0 Test_mAP_on_5.0 Train_mAP_on_10.0 Test_mAP_on_10.0 Train_mAP_on_20.0 Test_mAP_on_20.0 Train_mAP_on_30.0 Test_mAP_on_30.0 Train_mAP_on_40.0 Test_mAP_on_40.0 Train_mAP_on_50.0 Test_mAP_on_50.0 Train_mAP_on_70.0 Test_mAP_on_70.0 Train_mAP_on_100.0 Test_mAP_on_100.0 Mean_loss\n')

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])
#train_transforms, test_transforms = augmentation.initialize_transformation(448)

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
    OUTPUT.write(f'{sum(mean_loss)/len(mean_loss)}\n')
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
    # Maybe try this:
    # optimizer = optim.SGD(model.parameterse(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    loss_fn = YoloLoss()

    # In case a model gets loaded the checkpoint gets loaded
    if LOAD_MODEL:
        a = torch.load(LOAD_MODEL_FILE, map_location=DEVICE) # torch.device('cpu'))
        load_checkpoint(a, model, optimizer, LEARNING_RATE)

    if RUN_LOCAL == False:
        train_csv = "/scratch/tmp/jbalzer/data/VOC2007_2012/train.csv"
        test_csv = "/scratch/tmp/jbalzer/data/VOC2007_2012/test.csv"
    elif RUN_LOCAL == True:
        train_csv = "data/VOC2007_2012/train.csv"
        test_csv = "data/VOC2007_2012/test.csv"


    train_dataset = VOCDataset( 
        train_csv,
        transform=transform,
        #transform=train_transforms,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        S=S,
        B=B,
        C=num_classes,
    ) 

    test_dataset = VOCDataset(
        test_csv, 
        transform=transform, 
        #transform=test_transforms,
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

        # In case a model gets loaded, images will be used by the model
        if LOAD_MODEL:
            # TO DO: Calculate precision!

            for x, y in train_loader:
            #for x, y in test_loader: # y = labels
                with torch.no_grad():
                    model.eval()
                x = x.to(DEVICE)
                #torch.save(x, 'data/VOC2007_2012/x.pt')
                #torch.save(y, 'data/VOC2007_2012/y.pt')
                
                start_idx = 0
                #with torch.no_grad():
                 #       predictions = model(x) # Predictions

                # average_precision: 
                pred_boxes, target_boxes = get_bboxes(
                    train_loader, model, iou_threshold=0.5, threshold=0.4, S=S, B=B, C=num_classes,
                )
                mAP = mean_average_precision(pred_boxes, target_boxes)
                print('mAP:', mAP)
                #torch.save(pred_boxes, 'pred_boxes.pt')
                #torch.save(target_boxes, 'target_boxes.pt')
                #print('STOPP')
                #exit()

                for idx in range(8):    
                    
                    bboxes = cellboxes_to_boxes(model(x), S, B, num_classes)
                    bboxes = non_max_suppression(bboxes[idx+start_idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
                    print('plot VOC')

                    plot_image(x[idx+start_idx].permute(1,2,0).to("cpu"), bboxes, CLASS_NAMES)

                import sys
                sys.exit()

        ###### Calculate mAP based on training data
        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4, S=S, B=B, C=num_classes,
        )
        #train_mean_avg_prec = mean_average_precision(
        #    pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=num_classes
        #) 
        train_map_on_small = mAP_on_object_size(
            pred_boxes, target_boxes, iou_threshold=0.5, num_classes=num_classes, ratios=[0.0058, 0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 1.0]
        )     
        ###########################################

        ###### Calculate mAP based on test data
        pred_boxes, target_boxes = get_bboxes(
            test_loader, model, iou_threshold=0.5, threshold=0.4, S=S, B=B, C=num_classes,
        )
        #test_mean_avg_prec = mean_average_precision(
        #    pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=num_classes
        #)
        test_map_on_small = mAP_on_object_size(
            pred_boxes, target_boxes, iou_threshold=0.5, num_classes=num_classes, ratios=[0.0058, 0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 1.0]
        )

        ###########################################

        ### TO DO: ADD FUNCTION TO CALCULATE THE PRECISION OF OBJECTDETECTION / -RECOGNITION 
        ### IN CONNECTION TO THE OBJECT AND IMAGE SIZE 
        ratios=[0.0058, 0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 1.0]
        for ratio in ratios:
            OUTPUT.write(f'{float(train_map_on_small[str(ratio)])} {float(test_map_on_small[str(ratio)])} ')
        # OUTPUT.write(f'{train_mean_avg_prec} {test_mean_avg_prec} ')
        print(f"Train mAP on small obj.: {train_map_on_small}, Test mAP on small obj.: {test_map_on_small}")

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
