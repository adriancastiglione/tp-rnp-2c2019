
import torch
import torchvision as tv
import torch.nn.functional as F
import numpy as np
import time
import sys
import math
import os
import shutil
from yolo import YoloBase



def train_one_epoch(epoch_num, epoch_total, model, data, device, optim, costf):

	total_loss = 0.0
	for i, (images, labels) in enumerate(data):
		images = images.to(device)
		labels = labels.to(device)

		output = model(images)
		error = costf(output, labels)
		optim.zero_grad()
		error.backward()
		optim.step()
		loss = error.item()
		total_loss += loss
		if (i+1)%10 == 0:
			print ( "Epoch [{}/{}], Batch [{}], Loss: {:.4f}".format( epoch_num, epoch_total, i+1, loss))

	return total_loss/i



def execute_pretraining(model, data, device, optim, costf, epochs, checkpoints_path):
	model.train()
	best_model_loss = math.inf
	for i in range(1, epochs + 1):
		init = time.time()
		current_loss = train_one_epoch(i, epochs, model, data, device, optim, costf)
		end = time.time()

		checkpoint_name = 'checkpoint_pretraining_epoch_%s.tar' % i
		is_best_model = current_loss < best_model_loss
		best_model_loss = min(best_model_loss, current_loss)

		make_checkpoint({
			'epoch': i + 1,
			'state_dict': model.state_dict(),
			'optimizer' : optim.state_dict(),
			'loss' : current_loss
		}, is_best_model, os.getcwd() + '/' + checkpoints_path + '/' + checkpoint_name)

		print('Epoca completada en %s segundos. Loss: %s \n' % ((end-init), current_loss))

		check = '0'
		with open('continue.txt', 'r') as f:
			check = f.read()

		if int(check) == 1:
			break

	model.eval()



def make_checkpoint(state, is_best_model, checkpoint_filename):
	torch.save(state, checkpoint_filename)
	if is_best_model:
		shutil.copy(checkpoint_filename, 'model_best.tar')


def pretrain_model(model, dataset_path, epochs, device, checkpoints_path):
    
	transform = tv.transforms.Compose(
		[
			tv.transforms.Resize((224, 224)),
			tv.transforms.ToTensor(),
		])

	train_dataset = tv.datasets.ImageFolder(
		root=dataset_path,
		transform=transform
	)

	train_loader = torch.utils.data.DataLoader(
		train_dataset,
		batch_size=64,
		num_workers=24,
		shuffle=True
	)


	model.setForPretraining(len(train_dataset.classes))
	model.cuda()
	optim = torch.optim.Adam( model.parameters())
	costf = torch.nn.CrossEntropyLoss() 

	execute_pretraining(model, train_loader, device, optim, costf, epochs, checkpoints_path)


if __name__ == '__main__':

	model = YoloBase()
	device = torch.device('cuda')

	pretrain_model(model, sys.argv[1], int(sys.argv[2]), device, sys.argv[3])
