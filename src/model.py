"""
Karan Shah and Jyothi Vishnu
CS-7180 Fall 2023 semester

Implementation of Yolo (v1) architecture 
with modification of added Batchnorm.
"""

import torch
import torch.nn as nn


"""
Architecture config:
Tuple is (kernel_size, filters, stride, padding)
"M" represents maxpooling with stride 2x2 and kernel 2x2
List contains tuples and a final int for number of repeats
"""

architecture_config = [
    (7, 64, 2, 3), 
    "M",
    (3, 192, 1, 1), # same padding throughout (input and output are the same)
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1)
]

class CNNBlock(nn.Module):
    """
    Creates CNN block where convolution, batchnorm, and leaky relu are applied to input
    """

    # Initializes convolution, batchnorm, and leaky relu 
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    # Forward pass for CNN block
    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))
    
class Yolov1(nn.Module):
    """
    Creates YoloV1 model architecture
    """

    # Initializes architecture config, number of input channels, the darknet, and fully connected layers
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    # Forward pass for YOLO
    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))
    
    # Create the convolution layers of architecture
    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for config in architecture:
            if type(config) == tuple:
                layers += [
                    CNNBlock(
                        in_channels, config[1], kernel_size=config[0], stride=config[2], padding=config[3] 
                    )
                ]
                in_channels = config[1]

            elif type(config) == str:
                layers += [
                    nn.MaxPool2d(kernel_size=2, stride=2)
                ]

            elif type(config) == list:
                conv1 = config[0]
                conv2 = config[1]
                repeats = config[2]

                for _ in range(repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    # Create the fully connected layers
    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S * S * (C + B * 5))
        )       



# if __name__ == "__main__":
#     x = torch.randn(2, 3, 448, 448) # batch size, channels, h, w
#     model = Yolov1(split_size=7, num_boxes=2, num_classes=20)
#     result = model(x)
#     print(result.shape) # (2, 1470 = 7 * 7 * 30)