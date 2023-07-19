#!/usr/bin/env python
# coding: utf-8

import sys
print(sys.version_info)

n_config = sys.argv[-1]
print(f'n_config is {n_config}\n')

#############################################################################
### All Imports #############################################################
#####################

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.autograd as autograd

import numpy as np

import json
import time
beginTime = time.time()

import os
import pathlib
from shutil import copy, copyfile

import traceback
import datetime

import functools
print = functools.partial(print, flush=True)


with open(f'./config/config_{n_config}.json', "r") as f:
    exp = json.loads(f.read())

from pprint import pprint

pprint(exp)

import pathlib
pathlib.Path(f'./saved_data/').mkdir(parents=True, exist_ok=True)

import os

from code_inpainting import Subsampling2, Subsampling2Specific
from code_motion_blur import MotionBlur, MotionBlurSpecific 
from code_denoising import Denoising_operator


def power_iteration_torch_T(conv_A, conv_A_T, num_iterations: int, eps: float, b_inf=None):
    v_norm = []
    
    if b_inf is None:
        ch_in = conv_A.in_channels
        wid, hei = conv_A.kernel_size
        wid += 2
        hei += 3
        b_k = torch.rand([1, ch_in, wid, hei]) + eps
        b_k = b_k.to(torch.device(conv_A.weight.device))
    else:
        b_k = b_inf.detach().clone()
        
    for _ in range(num_iterations):
        b_k1 = conv_A_T(conv_A(b_k))

        # calculate the norm
        b_k1_norm = torch.linalg.vector_norm(b_k1)

        # re normalize the vector
        b_k = b_k1 / b_k1_norm

        v_norm.append( (torch.linalg.vector_norm( conv_A_T(conv_A(b_k)) ) / torch.linalg.vector_norm(b_k)).item() ) 
        if len(v_norm)>=2:
            my_rel_sensitivity = abs(v_norm[-1]-v_norm[-2])/v_norm[-1]
        
            if my_rel_sensitivity < 1e-15:
                break
    return b_k, v_norm


def load_dataset(dataset_name):
    if dataset_name=='MNIST':
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)) ])
        trainset = datasets.MNIST("../data", train=True,  download=False, transform=transform)
        testset  = datasets.MNIST("../data", train=False, download=False, transform=transform)
        batch_size = 100

    if dataset_name=='CELEBA':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(),
            transforms.Normalize((0.5,), (0.5,)) ])
        
        dataset_path = '../data/celeba/reduced_10'
        celeba_data = datasets.ImageFolder(dataset_path, transform=transform)
        trainset, testset = torch.utils.data.random_split(celeba_data, [int(10), int(10)])
        batch_size = 10

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=7, pin_memory=True)
    testloader  = torch.utils.data.DataLoader(testset,  batch_size=batch_size, shuffle=True, num_workers=7, pin_memory=True)
        
    return trainloader, testloader


torch.manual_seed(0)
np.random.seed(124)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

unique_ID = exp['uniqueID']

# Inpainting
percentage                      = exp['percentage']

# Denoising
epsilon                         = exp['epsilon']

# Deblurring
blur_kernel_size                = exp['blur_kernel_size']

# K (operator)
out_channels                    = exp['out_channels']
kernel_size                     = exp['kernel_size']
stride                          = exp['stride']
dilation=1 
padding = int((-1 + kernel_size + (kernel_size-1)*(dilation-1))/2)

eps_sq = exp['eps_sq']

# Other parameters
gamma                           = exp['gamma']
lambd                           = exp['lambd']
inpainting,denoising,deblurring = exp['inpainting_denoising_deblurring']
tau                             = exp['tau']
max_epochs                      = exp['max_epochs']

# Anderson acceleration / forward iteration
tol                             = exp['tol']
max_iter                        = exp['max_iter']
step_method                     = exp['step_method']

sigma                           = exp['sigma']
dataset_name                    = exp['dataset']
DEQ_model                       = exp['DEQ_model']

# Define \sigma (regularizer)

if sigma=='identity':
    sigma = nn.Identity()
elif sigma=='softshrink':
    sigma = nn.Softshrink(tau)
elif sigma=='relu':
    sigma = nn.ReLU()
elif sigma=='tanh':
    sigma = nn.Tanh()
else:
    raise Exception(f'Variable sigma "{sigma}" has not been defined yet.')


trainloader, testloader = load_dataset(dataset_name)

for i, data in enumerate(trainloader, 0):
    inputs_train, _ = data
    break

for i, data in enumerate(testloader, 0):
    inputs_test, _ = data
    break

print(f"There are {inputs_train.shape[0]} images per (train) batch")

in_channels = inputs_train.shape[-3]

inputs_train = inputs_train.to(device)
inputs_test  =  inputs_test.to(device)


# define K_operator
if inpainting:
    x_min_perc, x_max_perc, y_min_perc, y_max_perc = (1-percentage)/2, (1+percentage)/2, (1-percentage)/2, (1+percentage)/2 # 'central block' inpainting
    K_operator = Subsampling2Specific(inputs_train[0], x_min_perc, x_max_perc, y_min_perc, y_max_perc, transpose=False, device=device)
elif deblurring:
    padding_mode = 'circular'
    blur_type = 'diagonal'
    K_operator = MotionBlurSpecific(1, 1, blur_kernel_size, blur_type=blur_type, transpose=False, device=device)
else: # denoising:
    K_operator = Denoising_operator(transpose=False)

torch_image_train = inputs_train.detach().clone().to(device)
torch_image_test = inputs_test.detach().clone().to(device)

delta_train_test = torch.randn(torch_image_train.shape).to(device)

noisy_image_train = (K_operator@torch_image_train + epsilon*delta_train_test).to(device)
noisy_image_test = (K_operator@torch_image_test + epsilon*delta_train_test).to(device)

class net_test_TV(nn.Module):
    def __init__(self, device, in_channels = 3, out_channels = 16, kernel_size = 3, stride = 1, padding = 1, gamma = None, sigma=None, DEQ_model=False, eps_sq=0.01, TV_like_init=None, BOOLEAN_UPDATE_NORMS=None, SECOND_ORDER=None):
        super(net_test_TV, self).__init__()
        
        if BOOLEAN_UPDATE_NORMS is None:
            self.BOOLEAN_UPDATE_NORMS = False
        else:
            self.BOOLEAN_UPDATE_NORMS = BOOLEAN_UPDATE_NORMS

        if TV_like_init is None:
            TV_like_init = False
        if SECOND_ORDER is None:
            SECOND_ORDER = False
            
        self.DEQ_model = DEQ_model
        self.device = device

        self.ConvLayer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False) # A
        
        self.ConvLayer_T = nn.ConvTranspose2d(out_channels, in_channels, kernel_size, stride, padding, bias=False) # A^T or C^T

        if DEQ_model:    
            self.ConvLayer_C = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False) # C
            self.ConvLayer_AT = nn.ConvTranspose2d(out_channels, in_channels, kernel_size, stride, padding, bias=False) # A^T

        if TV_like_init:
            self.ConvLayer.weight = torch.nn.Parameter(torch.zeros(self.ConvLayer.weight.shape))

            if DEQ_model:    
                self.ConvLayer_C.weight = torch.nn.Parameter(torch.zeros(self.ConvLayer_C.weight.shape))

            with torch.no_grad():
                if SECOND_ORDER:
                    self.ConvLayer.weight[0,0,kernel_size//2,kernel_size//2-1] =  1.0
                    self.ConvLayer.weight[0,0,kernel_size//2,kernel_size//2]   = -2.0
                    self.ConvLayer.weight[0,0,kernel_size//2,kernel_size//2+1] =  1.0
                    self.ConvLayer.weight[1,0,kernel_size//2-1,kernel_size//2] =  1.0
                    self.ConvLayer.weight[1,0,kernel_size//2,kernel_size//2]   = -2.0
                    self.ConvLayer.weight[1,0,kernel_size//2+1,kernel_size//2] =  1.0
                    
                else: #1st-order TV
                    self.ConvLayer.weight[0,0,kernel_size//2,kernel_size//2] = -1.0
                    self.ConvLayer.weight[0,0,kernel_size//2,1+kernel_size//2] =  1.0
                    self.ConvLayer.weight[1,0,kernel_size//2,kernel_size//2] = -1.0
                    self.ConvLayer.weight[1,0,1+kernel_size//2,kernel_size//2] =  1.0


        with torch.no_grad():
            if DEQ_model:    
                self.ConvLayer_T.weight = nn.Parameter(self.ConvLayer.weight.detach().clone()) # copy values, without linking them

                self.ConvLayer_AT.weight = self.ConvLayer.weight
                self.ConvLayer_C.weight  = self.ConvLayer_T.weight
            else:
                self.ConvLayer_T.weight = self.ConvLayer.weight
        
        self.eps_sq = eps_sq

        if sigma is None:
            print("!!!!!!!!!! WARNING !!!!!!!!!!")
            print("Variable sigma is None. It will be put equal to nn.Identity()\n\n\n")
            self.sigma = nn.Identity()
        else:
            self.sigma = sigma

        if gamma is not None:
            self.gamma = gamma
        else:
            self.gamma = nn.Parameter(torch.Tensor([1.0]))
            
        self.chi = 1.0
        
        if DEQ_model:
            self.b_k_A, v_norm_test = power_iteration_torch_T(self.ConvLayer, self.ConvLayer_AT, int(1e3), 50.0)
        else:
            self.b_k_A, v_norm_test = power_iteration_torch_T(self.ConvLayer, self.ConvLayer_T, int(1e3), 50.0)
        
        self.b_k_A = self.b_k_A.to(self.device)
        self.norm_A = (v_norm_test[-1])**0.5
        self.v_norm_A = []
        self.v_norm_A.append(self.norm_A)        

        if DEQ_model:
            self.b_k_C, v_norm_test = power_iteration_torch_T(self.ConvLayer_C, self.ConvLayer_T, int(1e3), 50.0)
        else:
            self.b_k_C = self.b_k_A

        self.b_k_C = self.b_k_C.to(self.device)
        self.norm_C = (v_norm_test[-1])**0.5
        self.v_norm_C = []
        self.v_norm_C.append(self.norm_C)

        print("Initial norms:")
        print(f'norm_A = {self.norm_A}')
        print(f'norm_C = {self.norm_C}')
        print() 

    def update_norms(self):
        if self.BOOLEAN_UPDATE_NORMS:
            if self.DEQ_model:
                self.b_k_A, v_norm_test = power_iteration_torch_T(self.ConvLayer, self.ConvLayer_AT, int(1e3), 50.0)
            else:
                self.b_k_A, v_norm_test = power_iteration_torch_T(self.ConvLayer, self.ConvLayer_T, int(1e3), 50.0) 

            self.b_k_A = self.b_k_A.to(self.device)
            self.norm_A = (v_norm_test[-1])**0.5
            self.v_norm_A = []
            self.v_norm_A.append(self.norm_A)        

            if self.DEQ_model:
                self.b_k_C, v_norm_test = power_iteration_torch_T(self.ConvLayer_C, self.ConvLayer_T, int(1e3), 50.0) 
            else:
                self.b_k_C = self.b_k_A

            self.b_k_C = self.b_k_C.to(self.device)
            self.norm_C = (v_norm_test[-1])**0.5
            self.v_norm_C = []
            self.v_norm_C.append(self.norm_C)
        else:
            self.norm_A = 1.0
            self.norm_C = 1.0
        
        
    
    def forward(self, x):
        # self.update_norms()
        if self.BOOLEAN_UPDATE_NORMS:
            cTV = 2.7300944 
        else:
            cTV = 1.0
        x = self.gamma*( cTV*(1/self.norm_C)*self.ConvLayer_T(self.sigma(self.chi * self.ConvLayer(x) * (1/self.norm_A)*cTV)) ) 
        return x


kernel_size = exp['kernel_size'] 
out_channels = exp['out_channels'] 
TV_like_init = exp['TV_like_init'] 
SECOND_ORDER = exp['SECOND_ORDER']
BOOLEAN_UPDATE_NORMS = exp['BOOLEAN_UPDATE_NORMS'] 

LOAD_CHECKPOINT = exp['LOAD_CHECKPOINT'] 
checkpoint_ID = exp['checkpoint_ID'] 

DEQ_model = exp['DEQ_model'] 

CHANGED_TASK = exp['CHANGED_TASK'] 

opt_lr = exp['opt_lr']

lambd = 18.6307
lambd_3x3 = 18.1560

class next_step_u(nn.Module):
    def __init__(self, K, grad_R, tau=0.1, lambd=1.0, train_lambd=False):
        super().__init__()

        # Define parameters
        self.tau   = tau
        
        if train_lambd:
            self.lambd = nn.parameter.Parameter(torch.Tensor([lambd]), requires_grad=True) 
        else:
            self.lambd = lambd

        self.K = K
        self.grad_R = grad_R

    def forward(self, u, f_delta):  
        return u - self.tau*(self.lambd * (self.K.T @ (self.K @ u - f_delta)) + self.grad_R(u))


def forward_iteration(f, x0, max_iter=50, tol=1e-2):
    f0 = f(x0)
    res = []
    with torch.no_grad():
        n_pixels = torch.prod(torch.Tensor(list(x0.shape)))
    for k in range(max_iter):
        x = f0
        f0 = f(x)
        res.append((f0 - x).norm().item() / (1e-5 + f0.norm().item())) # Relative residual
        if (res[-1] < tol):
            break
    return f0, res, k


class DEQFixedPoint(nn.Module):
    def __init__(self, f, solver, **kwargs):
        super().__init__()
        self.f = f
        self.solver = solver
        self.kwargs = kwargs

        self.k_forward  = 0
        self.k_backward = 0

    def forward(self, ydelta):
        # compute forward pass and re-engage autograd tape
        with torch.no_grad():
            z, self.forward_res, self.k_forward = self.solver(lambda z : self.f(z, ydelta), ydelta.detach().clone(), **self.kwargs)
        z = self.f(z,ydelta)

        # set up Jacobian vector product (without additional forward calls)
        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0,ydelta)
        
        def backward_hook(grad):
            g, self.backward_res, self.k_backward = self.solver(lambda y : autograd.grad(f0, z0, y, retain_graph=True)[0] + grad, grad, **self.kwargs)
            return g

        try:
            z.register_hook(backward_hook)
        except RuntimeError:
            print("RuntimeError: cannot register a hook on a tensor that doesn't require gradient")
            print("I will continue without doing the backward_hook")
        return z


def epoch(loader, model, device, epsilon=0.0, opt=None, lr_scheduler=None):
    first_time_print_backward = False
    loss_function = nn.MSELoss()
    total_loss, total_err = 0.,0.
    sum_anderson_forward, sum_anderson_backward = 0.,0.
    sum_anderson_forward_res, sum_anderson_backward_res = 0.,0.

    cnt = 0
    model.eval() if opt is None else model.train()
    EVERYTHING_IS_FINE = 1
    
    for X,_ in loader:
        cnt+=1
        X = X.to(device)

        with torch.no_grad():
            gaussian_noise = torch.randn(X.size()).to(device)
            y_delta = model.f.K @ X.detach().clone() + gaussian_noise*epsilon

        if opt:
            opt.zero_grad()
        yp = model(y_delta)
        loss = loss_function(yp,X)
        
        # Check on NaN and Inf values
        with torch.no_grad():
            if loss.isnan():
                raise Exception('Error: Loss value is nan')
                break
            if loss.isinf():
                raise Exception('Error: Loss value is inf')
                break
                
        if opt:
            loss.backward()
            
            with torch.no_grad():        
                for param in model.parameters():
                    if type(param.grad) is None:
                        EVERYTHING_IS_FINE = 0
                    else:
                        EVERYTHING_IS_FINE *= (1-any(param.grad.isnan().flatten()))
                    if not EVERYTHING_IS_FINE:
                        print("There is at least one NaN value in the gradient")
                        total_err=-99
                        break
                    EVERYTHING_IS_FINE *= (1-any(param.grad.isinf().flatten()))
                    if not EVERYTHING_IS_FINE:
                        print("There is at least one Inf value in the gradient")
                        total_err=-99
                        break
        
            if not EVERYTHING_IS_FINE:
                break
            opt.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            model.f.grad_R.update_norms()

        total_loss += loss.item() * X.shape[0]

        sum_anderson_forward  += model.k_forward
        sum_anderson_backward += model.k_backward
        sum_anderson_forward_res  += model.forward_res[-1]
        try: 
            sum_anderson_backward_res += model.backward_res[-1]
        except:
            if not first_time_print_backward:
                print('model.backward_res[-1] cannot be accessed. I put the value to 100')
                first_time_print_backward = True
            sum_anderson_backward_res += 100

    return total_err / len(loader.dataset), total_loss / len(loader.dataset),           sum_anderson_forward/cnt, sum_anderson_backward/cnt, sum_anderson_forward_res/cnt, sum_anderson_backward_res/cnt 


v_train_error_perc = []
v_train_loss       = []
v_test_error_perc  = []
v_test_loss        = []
v_average_anderson_forw_train = []
v_average_anderson_back_train = []
v_average_anderson_forw_test  = []
v_average_anderson_back_test  = []
v_time_per_epoch   = []
v_average_anderson_forw_res_train = []
v_average_anderson_back_res_train = []
v_average_anderson_forw_res_test = []
v_average_anderson_back_res_test = []

net = net_test_TV(device, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, gamma=gamma, sigma=sigma, DEQ_model=DEQ_model, TV_like_init=TV_like_init, BOOLEAN_UPDATE_NORMS=BOOLEAN_UPDATE_NORMS, SECOND_ORDER=SECOND_ORDER).to(device)
net.chi = 100.0
net


update_function = next_step_u(K_operator, net, tau=1.0/8.0/net.chi, lambd=lambd_3x3, train_lambd=False)
if deblurring:
    update_function.tau/=4
    max_iter*=4
    
model = DEQFixedPoint(update_function, eval(step_method), tol=tol, max_iter=max_iter).to(device)


if LOAD_CHECKPOINT:
    checkpoint = torch.load(f'./saved_data/{device.type}_run_{checkpoint_ID}_checkpoint.pt')


try:
    del opt
except:
    pass

opt = optim.Adam(model.parameters(), lr=opt_lr) 
if deblurring:
    lr_scheduler = optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=1e-1, total_iters=max_epochs, last_epoch=-1, verbose=True)
else:
    lr_scheduler = optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=1e-2, total_iters=max_epochs, last_epoch=-1, verbose=True)

this_epoch = 0
if LOAD_CHECKPOINT:
    model.load_state_dict(checkpoint['model'])
    if not CHANGED_TASK:    
        this_epoch = checkpoint['this_epoch']
        this_epoch += 1
        opt.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        v_train_error_perc  = checkpoint['v_train_error_perc']
        v_train_loss        = checkpoint['v_train_loss']
        v_test_error_perc   = checkpoint['v_test_error_perc']
        v_test_loss         = checkpoint['v_test_loss']
        v_average_anderson_forw_train  = checkpoint['v_average_anderson_forw_train']
        v_average_anderson_back_train  = checkpoint['v_average_anderson_back_train']
        v_average_anderson_forw_test   = checkpoint['v_average_anderson_forw_test']
        v_average_anderson_back_test   = checkpoint['v_average_anderson_back_test']
        v_average_anderson_forw_res_train = checkpoint['v_average_anderson_forw_res_train']
        v_average_anderson_back_res_train = checkpoint['v_average_anderson_back_res_train']
        v_average_anderson_forw_res_test = checkpoint['v_average_anderson_forw_res_test']
        v_average_anderson_back_res_test = checkpoint['v_average_anderson_back_res_test']
        v_time_per_epoch               = checkpoint['v_time_per_epoch']
    print("A checkpoint has been loaded")


with torch.no_grad():
    train_x_plot = torch_image_train.detach().clone()
    train_y_delta_plot = noisy_image_train.detach().clone()
    train_f_y_plot = model(noisy_image_train).detach().clone()
    
    test_x_plot = torch_image_test.detach().clone()
    test_y_delta_plot = noisy_image_test.detach().clone()
    test_f_y_plot = model(noisy_image_test).detach().clone()



# TRAINING
if LOAD_CHECKPOINT:
    initial = ''
else:
    initial = 'initial_'
    this_epoch=0

if CHANGED_TASK:
    initial = 'initial_'
    this_epoch=0


i = this_epoch
j = this_epoch
while (i<max_epochs+1) and (j<1.5*max_epochs):
    start_time = time.time()    

    if initial == 'initial_': 
        train_error_perc, train_loss, average_anderson_forw_train, average_anderson_back_train, average_anderson_forw_res_train, average_anderson_back_res_train = epoch(trainloader, model, device, epsilon)
    else:
        train_error_perc, train_loss, average_anderson_forw_train, average_anderson_back_train, average_anderson_forw_res_train, average_anderson_back_res_train = epoch(trainloader, model, device, epsilon, opt, lr_scheduler)
    
    if train_error_perc<0:
        print(f'I will run again Epoch {i}, after decreasing tau and increasing max_iter.')
        model.f.tau = model.f.tau*0.1
        model.kwargs['max_iter'] = int(10*model.kwargs['max_iter'])
        print(f'tau = {model.f.tau}')
        print(f'max_iter = {model.kwargs["max_iter"]}')
        print()
        
    else:
        test_error_perc, test_loss, average_anderson_forw_test, average_anderson_back_test, average_anderson_forw_res_test, average_anderson_back_res_test = epoch(testloader,  model, device, epsilon)

        v_train_error_perc.append(train_error_perc)
        v_train_loss.append(train_loss)
        v_test_error_perc.append(test_error_perc)
        v_test_loss.append(test_loss)
        v_average_anderson_forw_train.append(average_anderson_forw_train)
        v_average_anderson_back_train.append(average_anderson_back_train)
        v_average_anderson_forw_test.append(average_anderson_forw_test )
        v_average_anderson_back_test.append(average_anderson_back_test )

        v_average_anderson_forw_res_train.append(average_anderson_forw_res_train)
        v_average_anderson_back_res_train.append(average_anderson_back_res_train)
        v_average_anderson_forw_res_test.append(average_anderson_forw_res_test )
        v_average_anderson_back_res_test.append(average_anderson_back_res_test )

        v_time_per_epoch.append(time.time() - start_time)

        print(f'End of Epoch {i}')
        print(f'Train loss: {train_loss}')
        print(f'Test loss: {test_loss}') 
        print(f'Time: {time.time() - start_time}', flush=True)
        print()

        if initial == 'initial_':
            initial=''

        i=i+1
    j=j+1

    if i%20==0:
        ## Save checkpoint
        checkpoint = { 
            'this_epoch': i,
            'model': model.state_dict(),
            'optimizer': opt.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'v_train_error_perc': v_train_error_perc, 
            'v_train_loss': v_train_loss,       
            'v_test_error_perc': v_test_error_perc,  
            'v_test_loss': v_test_loss,        
            'v_average_anderson_forw_train': v_average_anderson_forw_train, 
            'v_average_anderson_back_train': v_average_anderson_back_train, 
            'v_average_anderson_forw_test': v_average_anderson_forw_test,  
            'v_average_anderson_back_test': v_average_anderson_back_test,  
            'v_average_anderson_forw_res_train': v_average_anderson_forw_res_train,
            'v_average_anderson_back_res_train': v_average_anderson_back_res_train,
            'v_average_anderson_forw_res_test': v_average_anderson_forw_res_test,
            'v_average_anderson_back_res_test': v_average_anderson_back_res_test,
            'v_time_per_epoch': v_time_per_epoch   
        }
        torch.save(checkpoint, f'./saved_data/{device.type}_run_{unique_ID:03}_checkpoint.pt')

        print(f'Saved checkpoint in the following path:')
        print(f'./saved_data/{device.type}_run_{unique_ID:03}_checkpoint.pt')

## Save checkpoint 
checkpoint = { 
    'this_epoch': i,
    'model': model.state_dict(),
    'optimizer': opt.state_dict(),
    'lr_scheduler': lr_scheduler.state_dict(),
    'v_train_error_perc': v_train_error_perc, 
    'v_train_loss': v_train_loss,       
    'v_test_error_perc': v_test_error_perc,  
    'v_test_loss': v_test_loss,        
    'v_average_anderson_forw_train': v_average_anderson_forw_train, 
    'v_average_anderson_back_train': v_average_anderson_back_train, 
    'v_average_anderson_forw_test': v_average_anderson_forw_test,  
    'v_average_anderson_back_test': v_average_anderson_back_test,  
    'v_average_anderson_forw_res_train': v_average_anderson_forw_res_train,
    'v_average_anderson_back_res_train': v_average_anderson_back_res_train,
    'v_average_anderson_forw_res_test': v_average_anderson_forw_res_test,
    'v_average_anderson_back_res_test': v_average_anderson_back_res_test,
    'v_time_per_epoch': v_time_per_epoch   
}
torch.save(checkpoint, f'./saved_data/{device.type}_run_{unique_ID:03}_checkpoint.pt')

print(f'Saved FINAL checkpoint in the following path:')
print(f'./saved_data/{device.type}_run_{unique_ID:03}_checkpoint.pt')

