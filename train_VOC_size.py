"""
Main file for training Yolo model on Pascal VOC dataset

"""
import os
import argparse
from pickle import FALSE
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
#from dataset import VOCDataset
from dataset_augmented import (VOCDataset, Other_Dataset) ##### changed
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
import math
from datetime import datetime as dt

RUN_LOCAL = False
# Output file name on scratch for training on Palma
OUTPUT_FILE_NAME = 'VOC_135_lr_cos_0_0001_re_2_split'
CONTINUE_MODEL = False # THIS PARAMETER WAS ADDED TO CONTINUE THE TRAINING OF A SPECIFIC MODEL

seed = 123
torch.manual_seed(seed)

parser = argparse.ArgumentParser(description='YOLOv1-pytorch')
parser.add_argument("--cfg", "-c", default="cfg/VOC/yolov1.yaml", help="Yolov1 config file path", type=str)
parser.add_argument("--dataset_cfg", "-d", default="cfg/VOC/dataset.yaml", help="Dataset config file path", type=str)
parser.add_argument("--epochs", "-e", default=135, help="Training epochs", type=int)
parser.add_argument("--batch_size", "-bs", default=64, help="Training batch size", type=int)
parser.add_argument("--lr", "-lr", default=10**-4, help="Training learning rate", type=float)
parser.add_argument("--load_model", "-lm", default='False', help="Load Model or train one [ 'True' | 'False' ]", type=str)  
parser.add_argument("--model_path", "-mp", default=f"/scratch/tmp/jbalzer/yolov1/overfit_{OUTPUT_FILE_NAME}.pth.tar", help="Model path", type=str)
#parser.add_argument("--model_path", "-mp", default="overfit_VOC_500_validated_dropout_0_5_augmentation_added_lr_changed_test2.pth.tar", help="Model path", type=str)

args = parser.parse_args()


# Hyperparameters etc. 
LEARNING_RATE = args.lr
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = args.batch_size # 64 in original paper but I don't have that much vram, grad accum?
WEIGHT_DECAY = 0.0005
EPOCHS = args.epochs
if not RUN_LOCAL:
    NUM_WORKERS = 15
    LOAD_MODEL_FILE = args.model_path
elif RUN_LOCAL:
    NUM_WORKERS = 2
    LOAD_MODEL_FILE = 'overfit_test.pth.tar'
PIN_MEMORY = True
if args.load_model == 'True':
    LOAD_MODEL = True # DEFAULT MODUS: training
elif args.load_model == 'False':
    LOAD_MODEL = False


if not LOAD_MODEL and not CONTINUE_MODEL: 
    if RUN_LOCAL:
        OUTPUT = open('output_VOC_135_0001_split_csv_changed.txt', 'w')
    elif not RUN_LOCAL:
        OUTPUT = open(f'/scratch/tmp/jbalzer/yolov1/output_{OUTPUT_FILE_NAME}.txt', 'a') # HDF5 anstelle von .txt?
    # 0.0058, 0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 1.0
    #OUTPUT.write('Train_mAP_0_0058 amount Test_mAP0_0058 amount Train_mAP_0_01 amount Test_mAP0_01 amount Train_mAP_0_05 amount Test_mAP_0_05 amount Train_mAP_0_1 amount Test_mAP_0_1 amount Train_mAP_0_2 amount Test_mAP_0_2 amount Train_mAP_0_3 amount Test_mAP_0_3 amount Train_mAP_0_4 amount Test_mAP_0_4 amount Train_mAP_0_5 amount Test_mAP_0_5 amount Train_mAP_0_7 amount Test_mAP_0_7 amount Train_mAP_1_0 amount Test_mAP_1_0 amount Train_mAP Test_mAP Mean_loss\n')
    OUTPUT.write('Train_mAP_0_05 Test_mAP0_05 Train_mAP_0_2 Test_mAP_0_2 Train_mAP_0_5 Test_mAP_0_5 Train_mAP_1 Test_mAP_1 Train_mAP Test_mAP Mean_loss\n')
    #OUTPUT.write('Train_mAP Test_mAP Mean_loss\n')
    OUTPUT.close()

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


#transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])
train_transforms, test_transforms = augmentation.initialize_transformation(448) # Augmentation added

# Learning rate scheduler:
# (Works not 100 % like described in the paper, because the description does not work.
# Their learning rate scheduler runs on more than 135 epochs. So I reduced the number of epochs
# the lr is set to 10**-2 from 75 to only 70)
def lr_scheduler(epoch, lr):
    # Initially the lr is set to 10**-3 when starting the training
    if epoch <= 5:
        step = (10**-2 - 10**-3) / 6
        lr += step
    elif epoch > 5 and epoch <= 75:
        lr = 10**-2
    elif epoch > 75 and epoch <= 105:
        lr = 10**-3
    elif epoch > 105:
        lr = 10**-4
    return lr

'''
LR scheduler mit cos form probieren.
erste fünf epochen von 0 bis 10^-2 gehen und dannn mit cos form wieder runter gen 0 
'''
def cos_lr_scheduler(epoch, total_epochs, lr, ramp_epochs=2):
    cos_factor = 0.5 * (1 + math.cos(epoch / (total_epochs / 0.5) * 2*math.pi))
    start_ramp_factor = min(1, ( epoch / ramp_epochs ))
    total_factor = cos_factor * start_ramp_factor
    # returns new lr
    return lr * total_factor

# Training function
def train_fn(train_loader, model, optimizer, loss_fn):
    # Setting up a progress bar 
    loop = tqdm(train_loader, leave=True) 
    mean_loss = []


    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        optimizer.zero_grad()
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        loss.backward()
        optimizer.step()

        # Update progress bar
        loop.set_postfix(loss=loss.item())

    #OUTPUT.write(f"Mean loss was: {sum(mean_loss)/len(mean_loss)}\n")
    if RUN_LOCAL:
        OUTPUT = open('output_VOC_VOCDataset_test.txt', 'w')
    elif not RUN_LOCAL:
        OUTPUT = open(f'/scratch/tmp/jbalzer/yolov1/output_{OUTPUT_FILE_NAME}.txt', 'a') # HDF5 anstelle von .txt?
    OUTPUT.write(f'{sum(mean_loss)/len(mean_loss)}\n')    
    OUTPUT.close()
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
    #optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    loss_fn = YoloLoss()

    # In case a model gets loaded the checkpoint gets loaded
    if LOAD_MODEL or CONTINUE_MODEL:
        a = torch.load(LOAD_MODEL_FILE, map_location=DEVICE) # torch.device('cpu'))
        load_checkpoint(a, model, optimizer, LEARNING_RATE)

    if RUN_LOCAL == False:
        train_csv = "/scratch/tmp/jbalzer/data/VOC2007_2012/train.csv"
        test_csv = "/scratch/tmp/jbalzer/data/VOC2007_2012/test.csv"
        smaller_5_train_csv = "/scratch/tmp/jbalzer/data/VOC2007_2012/smaller_5_train.csv"
        smaller_5_test_csv = "/scratch/tmp/jbalzer/data/VOC2007_2012/smaller_5_test.csv"
        smaller_20_train_csv = "/scratch/tmp/jbalzer/data/VOC2007_2012/smaller_20_train.csv"
        smaller_20_test_csv = "/scratch/tmp/jbalzer/data/VOC2007_2012/smaller_20_test.csv"
        smaller_50_train_csv = "/scratch/tmp/jbalzer/data/VOC2007_2012/smaller_50_train.csv"
        smaller_50_test_csv = "/scratch/tmp/jbalzer/data/VOC2007_2012/smaller_50_test.csv"
        smaller_100_train_csv = "/scratch/tmp/jbalzer/data/VOC2007_2012/smaller_100_train.csv"
        smaller_100_test_csv = "/scratch/tmp/jbalzer/data/VOC2007_2012/smaller_100_test.csv"

    elif RUN_LOCAL == True:
        train_csv = "data/VOC2007_2012/train.csv"
        test_csv = "data/VOC2007_2012/test.csv"
        smaller_5_train_csv = "data/VOC2007_2012/smaller_5_train.csv"
        smaller_5_test_csv = "data/VOC2007_2012/smaller_5_test.csv"
        smaller_20_train_csv = "data/VOC2007_2012/smaller_20_train.csv"
        smaller_20_test_csv = "data/VOC2007_2012/smaller_20_test.csv"
        smaller_50_train_csv = "data/VOC2007_2012/smaller_50_train.csv"
        smaller_50_test_csv = "data/VOC2007_2012/smaller_50_test.csv"
        smaller_100_train_csv = "data/VOC2007_2012/smaller_100_train.csv"
        smaller_100_test_csv = "data/VOC2007_2012/smaller_100_test.csv"


    train_dataset = VOCDataset( 
        train_csv,
        #transform=transform,
        transform=train_transforms,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        S=S,
        B=B,
        C=num_classes,
    )
    test_dataset = VOCDataset(
        test_csv, 
        #transform=transform, 
        transform=test_transforms,
        img_dir=IMG_DIR, 
        label_dir=LABEL_DIR,
        S=S,
        B=B,
        C=num_classes,
    )

    train_dataset_0_05  = VOCDataset( 
        smaller_5_train_csv,
        #transform=transform,
        transform=train_transforms,
        img_dir=IMG_DIR,
        #label_dir=LABEL_DIR,
        #label_dir='data/VOC2007_2012/smaller_5',
        label_dir = '/scratch/tmp/jbalzer/data/VOC2007_2012/smaller_5',
        S=S,
        B=B,
        C=num_classes,
    )
    test_dataset_0_05 = VOCDataset(
        smaller_5_test_csv, 
        #transform=transform, 
        transform=test_transforms,
        img_dir=IMG_DIR, 
        #label_dir=LABEL_DIR,
        #label_dir='data/VOC2007_2012/smaller_5',
        label_dir = '/scratch/tmp/jbalzer/data/VOC2007_2012/smaller_5',
        S=S,
        B=B,
        C=num_classes,
    )
    train_dataset_0_2  = VOCDataset( 
        smaller_20_train_csv,
        #transform=transform,
        transform=train_transforms,
        img_dir=IMG_DIR,
        #label_dir=LABEL_DIR,
        #label_dir='data/VOC2007_2012/smaller_20',
        label_dir = '/scratch/tmp/jbalzer/data/VOC2007_2012/smaller_20',
        S=S,
        B=B,
        C=num_classes,
    )
    test_dataset_0_2 = VOCDataset(
        smaller_20_test_csv, 
        #transform=transform, 
        transform=test_transforms,
        img_dir=IMG_DIR, 
        #label_dir=LABEL_DIR,
        #label_dir='data/VOC2007_2012/smaller_20',
        label_dir = '/scratch/tmp/jbalzer/data/VOC2007_2012/smaller_20',
        S=S,
        B=B,
        C=num_classes,
    )
    train_dataset_0_5  = VOCDataset( 
        smaller_50_train_csv,
        #transform=transform,
        transform=train_transforms,
        img_dir=IMG_DIR,
        #label_dir=LABEL_DIR,
        #label_dir='data/VOC2007_2012/smaller_50',
        label_dir = '/scratch/tmp/jbalzer/data/VOC2007_2012/smaller_50',
        S=S,
        B=B,
        C=num_classes,
    )
    test_dataset_0_5 = VOCDataset(
        smaller_50_test_csv, 
        #transform=transform, 
        transform=test_transforms,
        img_dir=IMG_DIR, 
        #label_dir=LABEL_DIR,
        #label_dir='data/VOC2007_2012/smaller_50',
        label_dir = '/scratch/tmp/jbalzer/data/VOC2007_2012/smaller_50',
        S=S,
        B=B,
        C=num_classes,
    )
    train_dataset_1 = VOCDataset( 
        smaller_100_train_csv,
        #transform=transform,
        transform=train_transforms,
        img_dir=IMG_DIR,
        #label_dir=LABEL_DIR,
        #label_dir='data/VOC2007_2012/smaller_100',
        label_dir = '/scratch/tmp/jbalzer/data/VOC2007_2012/smaller_100',
        S=S,
        B=B,
        C=num_classes,
    )
    test_dataset_1 = VOCDataset(
        smaller_100_test_csv, 
        #transform=transform, 
        transform=test_transforms,
        img_dir=IMG_DIR, 
        #label_dir=LABEL_DIR,
        #label_dir='data/VOC2007_2012/smaller_100',
        label_dir = '/scratch/tmp/jbalzer/data/VOC2007_2012/smaller_100',
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

    train_loader_0_05 = DataLoader(
        dataset=train_dataset_0_05,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )
    test_loader_0_05 = DataLoader(
        dataset=test_dataset_0_05,
        batch_size=BATCH_SIZE,  
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True, 
        drop_last=True,
    )
    train_loader_0_2 = DataLoader(
        dataset=train_dataset_0_2,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )
    test_loader_0_2 = DataLoader(
        dataset=test_dataset_0_2,
        batch_size=BATCH_SIZE,  
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True, 
        drop_last=True,
    )
    train_loader_0_5 = DataLoader(
        dataset=train_dataset_0_5,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )
    test_loader_0_5 = DataLoader(
        dataset=test_dataset_0_5,
        batch_size=BATCH_SIZE,  
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True, 
        drop_last=True,
    )
    train_loader_1 = DataLoader(
        dataset=train_dataset_1,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )
    test_loader_1 = DataLoader(
        dataset=test_dataset_1,
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
            if RUN_LOCAL:
                OUTPUT = open('output_VOC_VOCDataset_test.txt', 'w')
            else:
                OUTPUT = open(f'/scratch/tmp/jbalzer/yolov1/output_{OUTPUT_FILE_NAME}.txt', 'a') # HDF5 anstelle von .txt?
    
        
        # In case a model gets loaded, images will be used by the model
        if LOAD_MODEL:
            # TO DO: Calculate precision!

            for x, y in train_loader:
            #for x, y in test_loader: # y = labels
                with torch.no_grad():
                    model.eval()
                x = x.to(DEVICE)
                
                start_idx = 0

                # average_precision: 
                pred_boxes, target_boxes = get_bboxes(
                    train_loader, model, iou_threshold=0.5, threshold=0.4, S=S, B=B, C=num_classes,
                )
                mAP = mean_average_precision(pred_boxes, target_boxes)
                print('mAP:', mAP)

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
        train_mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=num_classes
        ) 
        ###### Calculate mAP with splitted training data
        train_mean_avg_pres_size_split = {}
        train_loaders = [train_loader_0_05, train_loader_0_2, train_loader_0_5, train_loader_1]
        sizes = [0.05, 0.2, 0.5, 1]
        for loader_idx in range(len(train_loaders)):
            pred_boxes, target_boxes = get_bboxes(
                train_loaders[loader_idx], model, iou_threshold=0.5, threshold=0.4, S=S, B=B, C=num_classes,
            )
            train_mean_avg_prec_split = mean_average_precision(
                pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=num_classes
            ) 
            train_mean_avg_pres_size_split[sizes[loader_idx]] = (train_mean_avg_prec_split)
            
        ###########################################

        ###### Calculate mAP based on test data
        pred_boxes, target_boxes = get_bboxes(
            test_loader, model, iou_threshold=0.5, threshold=0.4, S=S, B=B, C=num_classes,
        )
        test_mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=num_classes
        )
        ###### Calculate mAP with splitted test data
        test_mean_avg_pres_size_split = {}
        test_loaders = [test_loader_0_05, test_loader_0_2, test_loader_0_5, test_loader_1]
        sizes = [0.05, 0.2, 0.5, 1]
        for loader_idx in range(len(test_loaders)):
            pred_boxes, target_boxes = get_bboxes(
                test_loaders[loader_idx], model, iou_threshold=0.5, threshold=0.4, S=S, B=B, C=num_classes,
            )
            test_mean_avg_prec_split = mean_average_precision(
                pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=num_classes
            ) 
            test_mean_avg_pres_size_split[sizes[loader_idx]] = (test_mean_avg_prec_split)
            
        ###########################################
        ### TO DO: ADD FUNCTION TO CALCULATE THE PRECISION OF OBJECTDETECTION / -RECOGNITION 
        ### IN CONNECTION TO THE OBJECT AND IMAGE SIZE 
        
        
        sizes = [0.05, 0.2, 0.5, 1]
        for size in sizes:
            OUTPUT.write(
                #f'{float(train_map_on_small[str(ratio)][0])} {train_map_on_small[str(ratio)][1]} {float(test_map_on_small[str(ratio)][0])} {test_map_on_small[str(ratio)][1]} ')
                f'{float(train_mean_avg_pres_size_split[size])} {float(test_mean_avg_pres_size_split[size])} '
            )

        OUTPUT.write(f'{train_mean_avg_prec_split} {test_mean_avg_prec_split} ')
        OUTPUT.close()
        #print(f"Train mAP on small obj.: {train_map_on_small}, Test mAP on small obj.: {test_map_on_small}")
        print('Train:', train_mean_avg_pres_size_split)
        print('Test:', test_mean_avg_pres_size_split)
        print(f"Train mAP: {train_mean_avg_prec}, Test mAP: {test_mean_avg_prec}")

        
        # Scheduler needs to be added
       
        #optimizer.param_groups[0]['lr'] = lr_scheduler(epoch, current_lr)
        optimizer.param_groups[0]['lr'] = cos_lr_scheduler(epoch, EPOCHS, LEARNING_RATE, ramp_epochs=2)
        #current_lr = optimizer.param_groups[0]['lr'] 

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


if __name__ == "__main__":
    main()
