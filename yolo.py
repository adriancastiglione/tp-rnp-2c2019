import torch
import torchvision as tv
from torch.nn import LeakyReLU, Conv2d, BatchNorm2d, Sequential, MaxPool2d
import torch.nn.functional as F
import numpy as np
import time


class YoloBase(torch.nn.Module):
    
    def __init__( _ ): 
        
        super().__init__()

        _.model = Sequential(
            Sequential(
                Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
                BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                LeakyReLU(negative_slope=0.1, inplace=True)
            ),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            Sequential(
                Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                LeakyReLU(negative_slope=0.1, inplace=True)
            ),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            Sequential(
                Conv2d(192, 128, kernel_size=(1, 1), stride=(1, 1), bias=False),
                BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                LeakyReLU(negative_slope=0.1, inplace=True)
            ),
            Sequential(
                Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                LeakyReLU(negative_slope=0.1, inplace=True)
            ),
            Sequential(
                Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
                BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                LeakyReLU(negative_slope=0.1, inplace=True)
            ),
            Sequential(
                Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                LeakyReLU(negative_slope=0.1, inplace=True)
            ),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            Sequential(
                Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
                BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                LeakyReLU(negative_slope=0.1, inplace=True)
            ),
            Sequential(
                Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                LeakyReLU(negative_slope=0.1, inplace=True)
            ),
            Sequential(
                Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
                BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                LeakyReLU(negative_slope=0.1, inplace=True)
            ),
            Sequential(
                Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                LeakyReLU(negative_slope=0.1, inplace=True)
            ),
            Sequential(
                Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
                BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                LeakyReLU(negative_slope=0.1, inplace=True)
            ),
            Sequential(
                Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                LeakyReLU(negative_slope=0.1, inplace=True)
            ),
            Sequential(
                Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
                BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                LeakyReLU(negative_slope=0.1, inplace=True)
            ),
            Sequential(
                Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                LeakyReLU(negative_slope=0.1, inplace=True)
            ),
            Sequential(
                Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
                BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                LeakyReLU(negative_slope=0.1, inplace=True)
            ),
            Sequential(
                Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                LeakyReLU(negative_slope=0.1, inplace=True)
            ),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            Sequential(
                Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
                BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                LeakyReLU(negative_slope=0.1, inplace=True)
            ),
            Sequential(
                Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                LeakyReLU(negative_slope=0.1, inplace=True)
            ),
            Sequential(
                Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
                BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                LeakyReLU(negative_slope=0.1, inplace=True)
            ),
            Sequential(
                Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                LeakyReLU(negative_slope=0.1, inplace=True)
            )
        )

    def setForPretraining( _, output_classes):
        _.pretraining = True
        _.fc = torch.nn.Linear(1024, output_classes)


    def setForTraining( _ ):
        _.pretraining = False

    def forward(_, x):    

        y = _.model(x)
        if _.pretraining:
            y = F.avg_pool2d(y, (y.size(2), y.size(3)))
            y = y.squeeze()
            y = F.softmax(_.fc(y), dim = 0)

            
        return y




