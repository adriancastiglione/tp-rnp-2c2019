

import torch
from torch.utils.data import Dataset
import torchvision as tv
from PIL import Image


class ObjectDetectionDataset(Dataset):


	def __init__(self, img_list_txt, labels_path, transform = None):

		with open(img_list_txt, 'r') as file:
			self.img_list = file.read().split('\n')

		self.labels_path = labels_path
		self.transform = transform


	def __len__(self):
		return len(self.img_list)


	def __getitem__(self, index):

		img = 0
		anotation = []

		with open(self.img_list[index], 'rb') as f:
			img = Image.open(f)
			img = self.transform(img)

		image_name = os.path.basename(self.img_list[index])
		image_name = os.path.splitext(imagen_name)[0]
		with open(labels_path + '/' + image_name + '.txt', 'r') as f:
			anotations = f.read().split('\n')
			for current_anotation in anotations:
				current_anotation = anotation.split(' ')
				anotation.append({
					'class' : int(current_anotation[0]),
					'x' : float(current_anotation[1]),
					'y' : float(current_anotation[2]),
					'w' : float(current_anotation[3]),
					'h' : float(current_anotation[4])
				})


		return img, anotation
		
