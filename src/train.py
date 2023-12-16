"""
Karan Shah and Jyothi Vishnu
CS-7180 Fall 2023 semester
Main file for training Yolo model on Pascal VOC dataset
"""

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
)
import csv
from loss import YoloLoss

seed = 123
torch.manual_seed(seed)

# Hyperparameters etc. 
LEARNING_RATE = 2e-5
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "mps"
BATCH_SIZE = 16 # 64 in original paper but I don't have that much vram, grad accum?
WEIGHT_DECAY = 5e-4
EPOCHS = 100
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_TRAIN_MODEL_FILE = "darknet.pth.tar"
LOAD_VAL_MODEL_FILE = "darknet_val.pth.tar"
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"

# """""""""

class Compose(object):
    """
    Compose method to perform any transforms if necessary. 
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])


def train_fn(train_loader, model, optimizer, loss_fn):
    """
    Builds train loop for YOLOv1.
    """

    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE) # x -> (16, 3, 448, 448), y -> (16, 7, 7, 30)
        out = model(x) # out -> (16, 7*7*30 = 1470)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")


def main():
    """
    Main method that executes the training 
    """
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE) #Change to RESNET if you want to use different pretrained backbone
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss()

    # if LOAD_MODEL:
    #     load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    train_dataset = VOCDataset(
        "data/trainSet.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )

    val_dataset = VOCDataset(
        "data/valSet.csv", transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    #Create empty list to store accuracy values to plot later
    acc_list = []
    best_train_mAP = 0
    best_mAP = 0

    #Start counting time
    import time
    start_full_time = time.time()

    for epoch in range(EPOCHS):
        
        ## Uncomment if you want to see plots

        # for x, y in train_loader:
        #    x = x.to(DEVICE)
        #    for idx in range(8):
        #        bboxes = cellboxes_to_boxes(model(x))
        #        bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
        #        plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)

        #    import sys
        #    sys.exit()

        start_time = time.time() 

        # Calculate Train mAP
        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4, device=DEVICE
        )

        train_mAP = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )

        #Calculate Validation mAP
        pred_boxes, target_boxes = get_bboxes(val_loader, model, iou_threshold=0.5, threshold=0.4, device=DEVICE)
        val_mAP = mean_average_precision(pred_boxes, target_boxes, 
                                  iou_threshold=0.5, box_format="midpoint")
        
        print(f"Epoch: {epoch}" + " |" f"Train mAP: {train_mAP}" + " |" + f"Validation mAP: {val_mAP}")

        #write losses to csv file
        acc_list.append([epoch, train_mAP, val_mAP])
        with open("darknet.csv", 'a') as f:
          writer = csv.writer(f)
          writer.writerow([epoch, train_mAP, val_mAP])


        # update best val map
        if val_mAP > best_mAP:
            checkpoint = {
               "state_dict": model.state_dict(),
               "optimizer": optimizer.state_dict(),
           }
            save_checkpoint(checkpoint, filename=str(epoch) + LOAD_VAL_MODEL_FILE)
            best_mAP = val_mAP
            time.sleep(10)

        
        #Switch to training
        model.train()

        train_fn(train_loader, model, optimizer, loss_fn)

        #Demarcation   
        print(f"time: {time.time() - start_time}")
        print("----------------------------------------")
        print("----------------------------------------")


    print(f"Total_Time: {time.time() - start_full_time}")   


if __name__ == "__main__":
    main()
