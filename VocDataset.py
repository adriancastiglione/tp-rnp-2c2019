import os
import cv2
import random
from collections import namedtuple

from torch.utils.data import Dataset
import numpy as np


#only support voc dataset
class VocDataset(Dataset):

    def __init__(self, S, B, args):

        #Box definition
        Box = namedtuple('Box', 'cls_id x y w h')

        #variable to store boxes 
        self.annotations = []

        #variable to store images path
        self.images_paths = []

        self.S = S
        self.B = B
        self.classes = args['classes']


        self.img_size = args['img_size']
        self.annotations_path = args['labels_path']
        self.cell_size = int(self.img_size / self.S)
        self.args = args


        with open(args['train_file']) as train_file:
            for line in train_file.readlines():

                self.images_paths.append(line.strip())
                img_name = os.path.basename(line.strip()).split('.')[0]
                with open(os.path.join(self.annotations_path, img_name + '.txt')) as label_file:

                    boxes = []
                    for box in label_file.readlines():
                        box_data = [float(p) for p in box.strip().split()]
                        box_data[0] = int(box_data[0]) #change cls_id to int
                        box = Box(*box_data)

                        boxes.append(box)

                    self.annotations.append(boxes)    
                
    def __getitem__(self, index):
        image = cv2.imread(self.images_paths[index])
        boxes = self.annotations[index]

        #"""For data augmentation we introduce random scaling and
        #translations of up to 20% of the original image size. We
        #also randomly adjust the exposure and saturation of the im-
        #age by up to a factor of 1.5 in the HSV color space."""

        #data augment
        image = random_bright(image)
        image = random_hue(image)
        image = random_saturation(image)
        image, boxes = aug.resize(image, boxes, (self.img_size, self.img_size))
            
        #rescale to 0 - 1
        image = image / float(255)
        target = self._encode(image, boxes)

        #plot_tools.plot_compare(image, target, boxes)
        return image, target
        #plot_tools.plot_image_bbox(image, boxes)
        

    def __len__(self):
        return len(self.images_paths) 


    def _denormalize(self, box, img_shape):
        cls_name, x, y, w, h = box
        height, width = img_shape[:2]

        #unnormalize
        x *= width
        w *= width
        y *= height
        h *= height

        Box = type(box)
        return Box(cls_name, x, y, w, h)

    def _encode(self, image, boxes):
        """Transform image and boexs to a (7 * 7 * 30)
        numpy array(bbox + confidence + bbox + confidence
        + class_num)
        Args:
            image: numpy array, read by opencv
            boxes: namedtuple object
        
        Returns:
            a 7*7*30 numpy array
        """

        target = np.zeros((self.S, self.S, self.B * 5 + len(self.classes)))
        for box in boxes:
            cls_id, x, y, w, h = self._denormalize(box, image.shape)
            col_index = int(x / self.cell_size)
            row_index = int(y / self.cell_size)

            # assign confidence score

            #"""Formally we define confidence as Pr(Object) ∗ IOU truth
            #pred . If no object exists in that cell, the confidence 
            #scores should be zero. Otherwise we want the confidence score 
            #to equal the intersection over union (IOU) between the 
            #predicted box and the ground truth."""
            target[row_index, col_index, 4] = 1
            target[row_index, col_index, 9] = 1

            #assign class probs

            #"""Each grid cell also predicts C conditional class proba-
            #bilities, Pr(Class i |Object)."""
            target[row_index, col_index, 10 + cls_id] = 1

            #assign x,y,w,h
            target[row_index, col_index, :4] = box.x, box.y, box.w, box.h
            target[row_index, col_index, 5:9] = box.x, box.y, box.w, box.h


            
        return target

    def resize(self, image, boxes, image_shape):
        """resize image and boxes to certain
        shape
        Args:
            image: a numpy array(BGR)
            boxes: bounding boxes
            image_shape: two element tuple(width, height)
        
        Returns:
            resized image and boxes
        """

        origin_shape = image.shape
        x_factor = image_shape[1] / float(origin_shape[1])
        y_factor = image_shape[0] / float(origin_shape[0])

        #resize_image
        if (image.shape[1], image.shape[0]) != image_shape:
            image = cv2.resize(image, image_shape)

        #resize_box
        result = []
        for box in boxes:
            cls_id, x, y, w, h = plot_tools.unnormalize_box_params(box, origin_shape)

            x *= x_factor
            w *= x_factor
            y *= y_factor 
            h *= y_factor

            #clamping the box board, make sure box inside the image, 
            #not on the board
            tl_x = x - w / 2
            tl_y = y - h / 2
            br_x = x + w / 2
            br_y = y + h / 2

            tl_x = min(max(0, tl_x), self.img_size - 1)
            tl_y = min(max(0, tl_y), self.img_size - 1)
            br_x = max(min(self.img_size - 1, br_x), 0)
            br_y = max(min(self.img_size - 1, br_y), 0)

            w = br_x - tl_x
            h = br_y - tl_y
            x = (br_x + tl_x) / 2
            y = (br_y + tl_y) / 2

            Box = type(box)
            box = Box(cls_id, x, y, w, h)
            result.append(plot_tools.normalize_box_params(box, image.shape))

        return image, result 


    def random_hue(self, image):
        """randomly change the hue of an image
        Args:
            image: an image numpy array(BGR)
        """

        if random.random() < self.args['augmentation_probability']:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            factor = random.uniform(1. - self.args['hue'], 1. + self.args['hue'])
            h = h * factor
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return image

    def random_saturation(self, image):

        if random.random() < self.args['augmentation_probability']:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v= cv2.split(hsv)
            factor = random.uniform(1. - self.args['saturation'], 
                                    1. + self.args['saturation'])
            s = s * factor
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return image


    def random_bright(self, image):
        """randomly brightten an image
        Args:
            image: an image numpy array(BGR)
        """

        if random.random() < self.args['augmentation_probability']:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            factor = random.uniform(1. - self.args['brightness'], 1. + self.args['brightness'])
            v = v * factor
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return image

