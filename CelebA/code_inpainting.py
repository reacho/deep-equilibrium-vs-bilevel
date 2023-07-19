import torch
import torch.nn as nn
import numpy as np
import math
from math import prod

class Subsampling2:

    def __init__(self, mask:bool, transpose=False, device=None):
        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.mask = mask.to(self.device)
        self.transpose = transpose

    def __matmul__(self, argument):
        return self.mask * argument
    @property
    def shape(self):
        no_of_rows = prod(self.mask.shape)
        return [no_of_rows, no_of_rows]

    @property
    def T(self):
        return Subsampling2(self.mask, transpose=not self.transpose, device=self.device)
    
    
class Subsampling2Specific(Subsampling2):
    def __init__(self, image, x_min_perc, x_max_perc, y_min_perc, y_max_perc, transpose=False, device=None):
        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        self.transpose=transpose
        
        #if any([x_min_perc, x_max_perc, y_min_perc, y_max_perc > 1.0]):
        if any([i>1.0 for i in [x_min_perc, x_max_perc, y_min_perc, y_max_perc]]):
            x_min_perc/=100.0
            x_max_perc/=100.0
            y_min_perc/=100.0
            y_max_perc/=100.0
            
        # Check if it's just one image
        if len(image.shape)>4:
            raise Exception("Please check again the image size")
        elif len(image.shape)==4:
            print("!!Warning!!\nyou may have passed a batch of images rather than a single image. I will select the first image\n")
            mask_image = image[0].clone()
        elif len(image.shape)==3:
            mask_image = image.clone()
        elif len(image.shape)==2:
            print("!!Warning!!\nno channel is given. I will consider this as a grayscale image\n")
            mask_image = image.unsqueeze(0).clone()
        elif len(image.shape)==1:
            raise Exception("Error, the image has only one dimension. Please check the image size")
        
        _, pixels_height, pixels_width = mask_image.shape
        
        x_min = math.floor(x_min_perc*pixels_width)
        x_max = math.ceil( x_max_perc*pixels_width)
        y_min = math.floor(y_min_perc*pixels_height)
        y_max = math.ceil( y_max_perc*pixels_height)
        
        mask = torch.ones_like(mask_image, dtype=bool)
        mask = mask.to(self.device)
        block_dimensions = (slice(y_min,y_max), slice(x_min,x_max))
        
        for cont, layer in enumerate(mask):
            layer[block_dimensions] = 0

        super().__init__(mask, transpose=self.transpose, device=self.device)