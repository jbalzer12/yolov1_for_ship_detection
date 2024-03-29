"""
Implementation of Yolo (v1) architecture
with slight modification with added BatchNorm.

This file is nearly identical to the file model.py 
but got slight modification to use it for train with 
other hyperparameters like image size an raster size S
"""

import torch
import torch.nn as nn

""" 
Information about architecture config:
Tuple is structured by (kernel_size, filters, stride, padding) 
"M" is simply maxpooling with stride 2x2 and kernel 2x2
List is structured by tuples and lastly int with number of repeats
"""

# This config was used to train with image size 448 x 488 pixels and S=14
architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0), # (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4], # [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0), # original: (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M", # remove it to keep the matrix bigger to try S=28
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2], # original: [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1), # originally: (3, 1024, 1, 1)
    (3, 1024, 1, 1), # originally: (3, 1024, 2, 1)
    (3, 1024, 1, 1), # originally: (3, 1024, 1, 1)
    (3, 1024, 1, 1), # originally: (3, 1024, 1, 1)
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

        # Skip connections einbauen 

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)

        
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        #print('brfore:', x.shape)
        x = self.darknet(x)
        #print('after darknet:', x.shape)
        #print('flattened:', torch.flatten(x, start_dim=1).shape)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers += [
                    CNNBlock(
                        in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3],
                    )
                ]
                in_channels = x[1]

            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
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

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes

        # In original paper this should be
        # nn.Linear(1024*S*S, 4096),
        # nn.LeakyReLU(0.1),
        # nn.Linear(4096, S*S*(B*5+C))

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096),  # Hier liegt das Problem! Wenn sich S ändert, passt der Linear Layer nicht mehr. Dieses Problem wird wird nicht von der grundlegenden Architektur beeinflusst 
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S * S * (C + B * 5)), # classification layer?
        )
