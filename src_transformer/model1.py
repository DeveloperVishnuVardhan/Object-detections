"""
1. Jyothi Vishnu Vardhan Kolla
2. Karan Shah

CS-7180 Fall 2023 semester.

This file contains the code for implementing the Detr model.
"""

import torch
from torch import nn
from torchvision.models import resnet50
torch.set_grad_enabled(False);


class DETR(nn.Module):
    """
    Demo DETR implementation.

    Demo implementation of DETR in minimal number of lines, with the
    following differences wrt DETR in the paper:
    * learned positional encoding (instead of sine)
    * positional encoding is passed at input (instead of attention)
    * fc bbox predictor (instead of MLP)
    The model achieves ~40 AP on COCO val5k and runs at ~28 FPS on Tesla V100.
    Only batch size 1 supported.
    """
    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        
        super().__init__()
        self.backbone = resnet50()
        del self.backbone.fc

        # Create conversion layer.
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # Create a default pytorch Transformer.
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers
        )

        # Predictions heads, one extra class for predicting non-empty slots.
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # output positional encodings (object queries).
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # Spatial positional encodings.
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # Convert from 2048 to 256 feature planes for the transformer.
        h = self.conv(x)
        # print(f"shape after passing through the backbone {h.shape}")
        # Construct positional encodings.
        H, W = h.shape[-2:]

        mod_col_embed = self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1)
        mod_row_embed = self.row_embed[:H].unsqueeze(1).repeat(1, W, 1)

        # print(f"modified col embed {mod_col_embed.shape}, modified row embed {mod_row_embed.shape}") 

        concat_embeds = torch.cat([mod_col_embed, mod_row_embed], dim=-1)
        # print(f"Shape after concating {concat_embeds.shape}")     
        final_embeds = concat_embeds.flatten(0, 1).unsqueeze(1)
        # print(f"Shape of final embeds {final_embeds.shape}")

        # Propagate through the transformer.
        h = self.transformer(final_embeds + 0.1 * h.flatten(2).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1))
        
        # print(f"output shape after coming out from transformer: {h.shape}")
        h = h.transpose(0, 1)

        # Return the outputs.
        return {
            'pred_logits': self.linear_class(h),
            'pred_boxes': self.linear_bbox(h).sigmoid()
        }


# def test():
#     x = torch.randn((1, 3, 800, 1066))
#     model = DETR(10)
#     model(x)


# test()