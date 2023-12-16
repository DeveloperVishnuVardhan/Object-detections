"""
Karan Shah and Jyothi Vishnu
CS-7180 Fall 2023 semester
Implementation of Yolo Loss Function from paper
"""

import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    """
    Calculate the loss for yolo (v1) model
    """

    # Initializes the split size, number of bounding boxes, 
    # and number of classes with other hyperparams
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

        self.S = S
        self.B = B
        self.C = C

        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    # Forward method for multiple loss calculations
    def forward(self, predictions, target):
        # Input predictions are shaped (batch size, S*S*(C+B*5) so need to reshape to (batch size, S, S, C + B * 5))
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # Calculate IoU for two predicted bboxes with target bbox in all the cells
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25]) # iou((B, S, S, 4), (B, S, S, 4)) -> (batch size, S, S, 1)
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25]) # iou((B, S, S, 4), (B, S, S, 4)) -> (batch size, S, S, 1)
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0) # (2, batch size, S, S, 1)

        # Take the box with highest IoU out of the two predictions
        # bestbox will be indices of 0, 1 for which bbox was best for all cells in all batches.
        # 1 represents the second BB is better and 0 represents the first is better
        iou_maxes, bestbox = torch.max(ious, dim=0) # bestbox -> (batch size, S, S, 1)
        exists_box = target[..., 20].unsqueeze(3)  # (batch size, S, S, 1) # Identity of obj in cell i


        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        # if second BB is correct, then 1 * second BB + 0 * first BB. 
        # This is why you do bestbox * second BB + (1 - bestbox) * first BB
        box_predictions = exists_box * (
            (
                bestbox * predictions[..., 26:30]
                + (1 - bestbox) * predictions[..., 21:25]
            )
        )

        box_targets = exists_box * target[..., 21:25]

        # Take sqrt of width, height of boxes
        # torch.abs used b/c can't square root negative values that may be present due to initialization
        # 1e-6 added for numerical instability b/c if predictions are 0 would be issue for derivative of square root
        # torch.sign added b/c sign of gradient needs to be correct when using torch.abs
        box_predictions[..., 2:4] =  torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )

        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        # (N, S, S, 4) -> (N * S * S, 4)
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        # pred_box is the confidence score of the bbox with highest IoU
        pred_box = (
            bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21]
        )

        # (N,S,S,1) -> (N*S*S*1)
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21])
        )

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        # Want to train both prediction boxes when no bbox in cell
        # (N, S, S, 1) -> (N, S*S*1)
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        # (N, S, S, 20) -> (N * S * S, 20)
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim= -2),
            torch.flatten(exists_box * target[..., :20], end_dim = -2)
        )

        loss = (
            self.lambda_coord * box_loss 
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )

        return loss

# if __name__ == "__main__":
# pred = torch.randn(2, 7, 7, 30)
# target = torch.randn(2, 7, 7, 25)
# loss = YoloLoss()
# print(loss(pred, target))


