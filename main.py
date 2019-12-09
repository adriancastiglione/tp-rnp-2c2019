
import torch
from torch.nn import Module, Secuential, LeakyReLU, Conv2d, BatchNorm2d, Linear, Dropout
import torchvision as tv
from ObjectDetectionDataset import *
from yolo import *
import sys


S = 7
B = 2
C = 20


def loss(y_pred, y_true, S, B, C, l_coord, l_noobj):

	"""
		y_pred 		(batch_size, S*S*(B*5 + C))
			predicted values. Each value is a tensor from de network

		y_true 		(batch_size, 1)
			real values. Each value is a dictionary
	"""

	#getting ground truth


	y_pred = y_pred.view(S*S, (B*5 + C))
	


class Yolo(Module): 

	def __init__(self, S = 7, B = 2, C = 20, pretrained):

		super().__init__()

		base = torch.load(pretrained)

		self.model = Secuential(
			base,
			Sequential(
                Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                LeakyReLU(negative_slope=0.1, inplace=True)
            ),
            Sequential(
                Conv2d(1024, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                LeakyReLU(negative_slope=0.1, inplace=True)
            ),
            Sequential(
                Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                LeakyReLU(negative_slope=0.1, inplace=True)
            ),
            Sequential(
                Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                LeakyReLU(negative_slope=0.1, inplace=True)
            ),
            Sequential(
            	Linear(7*7*1024, 4096),
            	LeakyReLU(negative_slope=0.1, inplace=True)
            ),
            Dropout(0.5),
            Linear(4096, S*S*(B*5 + C))
		)


		def forward(self, x):

			return self.model(x)




if __name__ == '__main__':

"""
args:
	dataset image list
	anotations directory
	pretrained model for yolo
"""

	transform = tv.transforms.Compose(
			[
				tv.transforms.Resize((448, 448)),
				tv.transforms.RandomChoice([
					tv.transforms.RandomRotation(20),
					tv.transforms.Pad(0),
					tv.transforms.ColorJitter(saturation=1.5, hue=.1)
				])
				tv.transforms.ToTensor(),
			])

	train_dataset = ObjectDetectionDataset(sys.argv[1], sys.argv[2], transforms = transform)

	train_loader = torch.utils.data.DataLoader(
		train_dataset,
		batch_size=64,
		num_workers=24,
		shuffle=True
	)

	device = torch.device('cuda')
	yolo = Yolo(S, B, C, sys.argv[3])
	optim = torch.optim.Adam( model.parameters())


