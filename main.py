
import torch
import torch.nn as nn
from torch.nn import Module, Sequential, LeakyReLU, Conv2d, BatchNorm2d, Linear, Dropout
import torch.nn.functional as F
import torchvision as tv
from torch.autograd import Variable
from yolo import *
from VocDataset import *
import sys


class Loss(Module):
    def __init__(self,S,B,l_coord,l_noobj):
        super(yoloLoss,self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def compute_iou(self, box1, box2):
        '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].
        Return:
          (tensor) iou, sized [N,M].
        '''
        N = box1.size(0)
        M = box2.size(0)

        lt = torch.max(
            box1[:,:2].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,:2].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min(
            box1[:,2:].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,2:].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh<0] = 0  # clip at 0
        inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

        area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
        area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou


    def forward(self,pred_tensor,target_tensor):
        '''
        pred_tensor: (tensor) size(batchsize,S,S,Bx5+20=30) [x,y,w,h,c]
        target_tensor: (tensor) size(batchsize,S,S,30)
        '''
        N = pred_tensor.size()[0]
        coo_mask = target_tensor[:,:,:,4] > 0
        noo_mask = target_tensor[:,:,:,4] == 0
        coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor)
        noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor)

        coo_pred = pred_tensor[coo_mask].view(-1,30)
        box_pred = coo_pred[:,:10].contiguous().view(-1,5) #box[x1,y1,w1,h1,c1]
        class_pred = coo_pred[:,10:]                       #[x2,y2,w2,h2,c2]
        
        coo_target = target_tensor[coo_mask].view(-1,30)
        box_target = coo_target[:,:10].contiguous().view(-1,5)
        class_target = coo_target[:,10:]

        # compute not contain obj loss
        noo_pred = pred_tensor[noo_mask].view(-1,30)
        noo_target = target_tensor[noo_mask].view(-1,30)
        noo_pred_mask = torch.cuda.ByteTensor(noo_pred.size())
        noo_pred_mask.zero_()
        noo_pred_mask[:,4]=1;noo_pred_mask[:,9]=1
        noo_pred_c = noo_pred[noo_pred_mask] #noo pred只需要计算 c 的损失 size[-1,2]
        noo_target_c = noo_target[noo_pred_mask]
        nooobj_loss = F.mse_loss(noo_pred_c,noo_target_c,size_average=False)

        #compute contain obj loss
        coo_response_mask = torch.cuda.ByteTensor(box_target.size())
        coo_response_mask.zero_()
        coo_not_response_mask = torch.cuda.ByteTensor(box_target.size())
        coo_not_response_mask.zero_()
        box_target_iou = torch.zeros(box_target.size()).cuda()
        for i in range(0,box_target.size()[0],2): #choose the best iou box
            box1 = box_pred[i:i+2]
            box1_xyxy = Variable(torch.FloatTensor(box1.size()))
            box1_xyxy[:,:2] = box1[:,:2]/14. -0.5*box1[:,2:4]
            box1_xyxy[:,2:4] = box1[:,:2]/14. +0.5*box1[:,2:4]
            box2 = box_target[i].view(-1,5)
            box2_xyxy = Variable(torch.FloatTensor(box2.size()))
            box2_xyxy[:,:2] = box2[:,:2]/14. -0.5*box2[:,2:4]
            box2_xyxy[:,2:4] = box2[:,:2]/14. +0.5*box2[:,2:4]
            iou = self.compute_iou(box1_xyxy[:,:4],box2_xyxy[:,:4]) #[2,1]
            max_iou,max_index = iou.max(0)
            max_index = max_index.data.cuda()
            
            coo_response_mask[i+max_index]=1
            coo_not_response_mask[i+1-max_index]=1

            #####
            # we want the confidence score to equal the
            # intersection over union (IOU) between the predicted box
            # and the ground truth
            #####
            box_target_iou[i+max_index,torch.LongTensor([4]).cuda()] = (max_iou).data.cuda()
        box_target_iou = Variable(box_target_iou).cuda()
        #1.response loss
        box_pred_response = box_pred[coo_response_mask].view(-1,5)
        box_target_response_iou = box_target_iou[coo_response_mask].view(-1,5)
        box_target_response = box_target[coo_response_mask].view(-1,5)
        contain_loss = F.mse_loss(box_pred_response[:,4],box_target_response_iou[:,4],size_average=False)
        loc_loss = F.mse_loss(box_pred_response[:,:2],box_target_response[:,:2],size_average=False) + F.mse_loss(torch.sqrt(box_pred_response[:,2:4]),torch.sqrt(box_target_response[:,2:4]),size_average=False)
        #2.not response loss
        box_pred_not_response = box_pred[coo_not_response_mask].view(-1,5)
        box_target_not_response = box_target[coo_not_response_mask].view(-1,5)
        box_target_not_response[:,4]= 0
        #not_contain_loss = F.mse_loss(box_pred_response[:,4],box_target_response[:,4],size_average=False)
        
        #I believe this bug is simply a typo
        not_contain_loss = F.mse_loss(box_pred_not_response[:,4], box_target_not_response[:,4],size_average=False)

        #3.class loss
        class_loss = F.mse_loss(class_pred,class_target,size_average=False)

        return (self.l_coord*loc_loss + 2*contain_loss + not_contain_loss + self.l_noobj*nooobj_loss + class_loss)/N



class Yolo(Module): 

	def __init__(self, S, B, C, pretrained):

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



def make_checkpoint(state, is_best_model, checkpoint_filename):
	torch.save(state, checkpoint_filename)
	if is_best_model:
		shutil.copy(checkpoint_filename, 'model_best.tar')


def update_lr(optimizer, lr):

	for param_group in optimizer:
		param_group['lr'] = lr


if __name__ == '__main__':

	"""
	args:
		dataset image list
		anotations directory
		pretrained model for yolo
	"""
	
	S = 7
	B = 2
	C = 20

	args = {
		'classes': ["aeroplane", "bicycle", "bird", "boat", 
	           "bottle", "bus", "car", "cat", "chair", 
	           "cow", "diningtable", "dog", "horse", 
	           "motorbike", "person", "pottedplant", "sheep", 
	           "sofa", "train", "tvmonitor"
	    ],

	    'img_size': 224,
	    'labels_path': '',
	    'train_file': 'train.txt',
	    'hue': 0.5,
	    'saturation' : 0.7,
	    'brightness' : 0.7,
		'augmentation_probability' : 0.5
	}


	l_coord = 5
	l_noobj = 0.5

	train_dataset = VocDataset(S, B, args)

	train_loader = torch.utils.data.DataLoader(
		train_dataset,
		batch_size=64,
		num_workers=24,
		shuffle=True
	)

	device = torch.device('cuda')
	yolo = Yolo(S, B, C, sys.argv[1])
	yolo.cuda()

	loss = Loss(S,B,l_coord,l_noobj)
	

	momentum=0.9
	decay=0.0005


	checkpoints = './checkpoints'

	optim = torch.optim.Adam( yolo.parameters(), lr = lr_[0], momentum = momentum)

	lr_ = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]

	batches = 0
	yolo.train()
	best_model_loss = math.inf
	for e in range(0, 10):

		if batches > 20000:
			break

		update_lr(optim, e)

		#train batches
		total_loss = 0.0
		for i, (images, labels) in enumerate(data):
			
			if batches > 20000:
				break

			images = images.to(device)
			labels = labels.to(device)

			output = yolo(images)
			error = loss(output, labels)
			optim.zero_grad()
			error.backward()
			optim.step()
			loss = error.item()
			total_loss += loss
			if (i+1)%10 == 0:
				print ( "Epoch [{}/{}], Batch [{}], Loss: {:.4f}".format( epoch_num, epoch_total, i+1, loss))

			epoch_loss = total_loss/i
			batches += 1

		checkpoint_name = 'yolo_epoch_%s.tar' % e
		is_best_model = current_loss < best_model_loss
		best_model_loss = min(best_model_loss, current_loss)
	
		make_checkpoint({
			'epoch': e + 1,
			'state_dict': yolo.state_dict(),
			'optimizer' : optim.state_dict(),
			'loss' : current_loss
		},
		is_best_model, os.getcwd() + '/' + checkpoints + '/' + checkpoint_name)









