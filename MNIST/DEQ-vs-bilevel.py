#!/usr/bin/env python
# coding: utf-8

#############################################################################
### All Imports #############################################################
#####################

import sys
print(sys.version_info)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np

import json
import time
beginTime = time.time()

import os
import pathlib
from shutil import copy

torch.manual_seed(0)
np.random.seed(124)

import traceback
import datetime

import functools
print = functools.partial(print, flush=True)

def elapsed_time_from_beginning():
    return time.time() - beginTime

def main(**exp):
    
    #############################################################################
    ### Load values from exp ################################################
    #################################
    
    my_conv = None
    my_conv_t = None
    
    uniqueID = exp['uniqueID']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    path_images = f'./images/{device.type}'
    pathlib.Path(f'{path_images}/{uniqueID:03}').mkdir(parents=True, exist_ok=True)

    pathlib.Path(f'./saved_data/{device.type}_{uniqueID:03}').mkdir(parents=True, exist_ok=True)

    with open(f'./saved_data/{device.type}_{uniqueID:03}/config_{uniqueID:03}.json', 'w', encoding='utf-8') as f:
        json.dump(exp, f, ensure_ascii=False, indent=4)
        
    #inpainting
    percentage                        = exp['percentage']
    #denoising
    epsilon                           = exp['epsilon']
    #delurring
    blur_kernel_size                  = exp['blur_kernel_size']
    inpainting, denoising, deblurring = exp['inpainting_denoising_deblurring']

    tau                               = exp['tau']
    lambd                             = exp['lambd'] 
    max_epochs                        = exp['max_epochs']
    number_of_layers                  = exp['number_of_layers']
    number_of_pixels                  = exp['number_of_pixels']  
    tol                               = exp['tol'] 
    max_iter                          = exp['max_iter'] 
    chosen_model                      = exp['chosen_model']
    gamma                             = exp['gamma']
    
    # initialised later
    chosen_regulariser                = exp['chosen_regulariser']

    
    if deblurring:
        padding_mode = 'circular'
        blur_type = 'diagonal'
        
    print(f'epsilon:: {epsilon}')
    print(f'device: {device.type}')
    print(f'uniqueID: {uniqueID}')
    print(f'inpainting: {inpainting}')
    print(f'denoising: {denoising}')
    print(f'deblurring: {deblurring}')

    print(f'chosen_model: {chosen_model}')
    print(f'tau: {tau}')
    print(f'lambd: {lambd}')
    print(f'gamma: {gamma}')
    print(f'max_epochs: {max_epochs}')
    print()
    print('Anderson acceleration')
    print(f'tol: {tol}')
    print(f'max_iter: {max_iter}')
    print()
    
    
    #############################################################################
    ### All Regularisers ########################################################
    ########################

    def grad_regulariser_4(u, K, b, gamma=1, tau=0.1):
        sigma = nn.ReLU()
        return (gamma*sigma(u@K.t()+b)@K).reshape(u.size())

    def grad_regulariser_4_DEQ(u, parameters_list, gamma=1, tau=0.1): 
        sigma = nn.ReLU()
        return (gamma*parameters_list[-1](sigma(parameters_list[0](u)))).reshape(u.size())

    def grad_regulariser_5(u, K, b, gamma=1, tau=0.1): 
        sigma = nn.Softshrink(tau)
        return (gamma*sigma(u@K.t()+b)@K).reshape(u.size())

    def grad_regulariser_5_DEQ(u, parameters_list, gamma=1, tau=0.1): 
        sigma = nn.Softshrink(tau)
        return (gamma*parameters_list[-1](sigma(parameters_list[0](u)))).reshape(u.size())

    def grad_regulariser_6(u, K, b, gamma=1, tau=0.1): 
        return (gamma*(u@K.t()+b)@K).reshape(u.size())

    def grad_regulariser_6_DEQ(u, parameters_list, gamma=1, tau=0.1): 
        return (gamma*parameters_list[-1](parameters_list[0](u))).reshape(u.size())

    
    chosen_regulariser = eval(chosen_regulariser)
    
    #############################################################################
    ### All Models ##############################################################
    #####################

    # When < DEQ_bilevel_optimisation > is called, just pass the image with whatever size, either
    # > [100, 1, 1, 784]
    # > [100, 1, 784]
    # > [100, 784]
    # > [100, 1, 28, 28]
    # > [100, 28, 28]
    # The model will change the size of the input with .view() accordingly to the variable "chosen_architecture"

    class DEQ_bilevel_optimisation(nn.Module):
        def __init__(self, number_of_pixels, K=None, tau=1, lambd=1, gamma=1, R=None, chosen_architecture=None):
            super().__init__()
            # Define parameters
            self.weight_dimensions = number_of_pixels*number_of_pixels 

            if chosen_architecture is None:
                print("Warning: variable chosen_architecture is None")
                print("It will be set to < 'affine' > now")
                chosen_architecture = 'affine'

            self.tau   = tau
            self.lambd = lambd
            self.gamma = gamma
            self.K = K

            if R.__name__ == 'grad_regulariser_4_DEQ':
                number_of_matrices = 1

            if R.__name__ == 'grad_regulariser_5_DEQ':
                number_of_matrices = 1

            if R.__name__ == 'grad_regulariser_6_DEQ':
                number_of_matrices = 1

            self.layers_list = nn.ModuleList([])

            for k in range(number_of_matrices):
                self.layers_list.append(nn.Linear(self.weight_dimensions, self.weight_dimensions, bias=True))
            self.layers_list.append(nn.Linear(self.weight_dimensions, self.weight_dimensions, bias=False))

            self.correct_size = [-1, 1, 1, 28*28]
            self.correct_size_final = [-1, 1, 1, 28*28]

            self.R = R

        def forward(self, u, f_delta):
            u       =       u.view(self.correct_size)
            f_delta = f_delta.view(self.correct_size)
            this_R = self.R(u, self.layers_list, self.gamma, self.tau)

            return (u - self.tau*(self.lambd*self.K((self.K(u) - f_delta), transpose=True) + this_R)).view(self.correct_size_final)
        
    
    class bilevel_optimisation(nn.Module):
        def __init__(self, number_of_pixels, K=None, tau=1, lambd=1, gamma=1, R=None):
            super().__init__()
            # Define parameters
            self.weight_dimensions = number_of_pixels*number_of_pixels

            self.tau   = tau
            self.lambd = lambd
            self.gamma = gamma
            self.K = K

            self.affine = nn.Linear(self.weight_dimensions, self.weight_dimensions, bias=True)
            self.correct_size = [-1, 1, 1, 28*28]
            self.correct_size_final = [-1, 1, 1, 28*28]
            self.R = R


        def forward(self, u, f_delta):
            u       =       u.view(self.correct_size)
            f_delta = f_delta.view(self.correct_size)

            this_R = self.R(u, self.affine.weight, self.affine.bias, self.gamma, self.tau)
                
            return (u - self.tau*(self.lambd*self.K((self.K(u) - f_delta), transpose=True) + this_R)).view(self.correct_size_final)
                                 
    
    #############################################################################
    ### Operator Class  ################################################
    #################################
    class op_K(nn.Module):
        def __init__(self, deblurring=False, denoising=False, inpainting=False, my_conv=None, my_conv_t=None, K=None):
            super().__init__()
            self.deblurring = deblurring
            self.denoising = denoising
            self.inpainting = inpainting

            if deblurring:
                self.my_conv = my_conv
                self.my_conv_t = my_conv_t
            self.K = K

        def forward(self, x, transpose=False):
            if self.deblurring:
                if not transpose:
                    return (self.my_conv(x.view(-1,1,28,28))).view(-1,1,1,28**2)
                return (self.my_conv_t(x.view(-1,1,28,28))).view(-1,1,1,28**2)
            if self.denoising:
                return x
            if self.inpainting:
                return x.view(-1,1,1,28**2)@K
            
    
    #############################################################################
    ### Functions for Inpainting ################################################
    #################################

    ####################################
    ## HORIZONTAL MASK - WITH EXAMPLE ##
    ####################################
    # To do: implement pixel_random_sampling=False, row_random_sampling=False
    def mask_horizontal(percentage, K, reverse=False, pixel_random_sampling=False, row_random_sampling=False):
        # percentage has to be between 0.0 and 1.0
        if percentage<0.0:
            print("WARNING: Variable 'percentage' is smaller than 0.0\nI set percentage=0.0")
            percentage = 0.0
        if percentage>1.0:
            print("WARNING: Variable 'percentage' is larger than 1.0\nI set percentage=1.0")
            percentage = 1.0

        elements_K_to_zero = int(np.sqrt(K.size()[0])*torch.ceil(torch.sqrt(torch.tensor(K.size()[0]))*percentage))    

        if reverse==False:
            for cnt in range(elements_K_to_zero):
                K[cnt][cnt] = 0.0
        else:
            N = K.size()[0]
            for cnt in range(elements_K_to_zero):
                K[N-1-cnt][N-1-cnt] = 0.0    

        return K    

    #############
    ## Example ##
    ####################################

    # K = torch.eye(x.view(-1).size()[0])
    # K = mask_horizontal(percentage,K)
    # print((K@x).view( int(np.sqrt(x.size()[0])) , -1 ))

    ########################################################################
    ########################################################################

    ##################################
    ## VERTICAL MASK - WITH EXAMPLE ##
    ##################################

    # IMPORTANT: this works with 1 image at a time
    def apply_mask_vertical(K,x):
        if len(x.size()) == 1:
            x = torch.t(x.view( int(np.sqrt(x.size()[0])) , -1 ))    
        else:
            x = torch.t(x)
        x = torch.reshape(x, (-1,))

        return torch.t((K@x).view( int(np.sqrt(x.size()[0])) , -1 ))

    #############
    ## Example ##
    ####################################

    # K = torch.eye(x.view(-1).size()[0])
    # K = mask_horizontal(percentage,K)
    # print((apply_mask_vertical(K,x)).view( int(np.sqrt(x.size()[0])) , -1 ))

    ########################################################################
    ########################################################################


    
    #############################################################################
    ### Functions for Deblurring ################################################
    #################################

    if deblurring:
        my_conv   = nn.Conv2d(1, 1, blur_kernel_size, padding='same', padding_mode=padding_mode, bias=False).requires_grad_(False)
        my_conv.weight = nn.Parameter(torch.eye(blur_kernel_size).view(1,1,blur_kernel_size,blur_kernel_size), requires_grad=False)
        if blur_type == 'diagonal':
            my_conv.weight = nn.Parameter(torch.eye(blur_kernel_size).view(1,1,blur_kernel_size,blur_kernel_size), requires_grad=False)

        average_weight = torch.sum( torch.abs( my_conv.weight) ) 
        my_conv.weight = nn.Parameter(my_conv.weight / average_weight, requires_grad=False)

        my_conv_t = nn.Conv2d(1, 1, blur_kernel_size, padding='same', padding_mode=padding_mode, bias=False).requires_grad_(False)
        my_conv_t.weight = nn.Parameter(torch.flip(my_conv.weight.view(-1), [0]).view(my_conv.weight.size()), requires_grad=False)


    #############################################################################
    ### Anderson acceleration ###################################################
    #################################
    def anderson(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-2, beta = 1.0):
        """ Anderson acceleration for fixed point iteration. """
        bsz, d, H, W = x0.shape
        X = torch.zeros(bsz, m, d*H*W, dtype=x0.dtype, device=x0.device)
        F = torch.zeros(bsz, m, d*H*W, dtype=x0.dtype, device=x0.device)
        X[:,0], F[:,0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
        X[:,1], F[:,1] = F[:,0], f(F[:,0].view_as(x0)).view(bsz, -1)

        H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device)
        H[:,0,1:] = H[:,1:,0] = 1
        y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device)
        y[:,0] = 1

        res = []
        for k in range(2, max_iter):
            n = min(k, m)
            G = F[:,:n]-X[:,:n]
            H[:,1:n+1,1:n+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n, dtype=x0.dtype,device=x0.device)[None]
            alpha = torch.solve(y[:,:n+1], H[:,:n+1,:n+1])[0][:, 1:n+1, 0]   # (bsz x n)

            X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0]
            F[:,k%m] = f(X[:,k%m].view_as(x0)).view(bsz, -1)
            res.append((F[:,k%m] - X[:,k%m]).norm().item()/(1e-5 + F[:,k%m].norm().item()))
            if (res[-1] < tol):
                break
        return X[:,k%m].view_as(x0), res, k



    def forward_iteration(f, x0, max_iter=50, tol=1e-2):
        f0 = f(x0)
        res = []
        for k in range(max_iter):
            x = f0
            f0 = f(x)
            res.append((f0 - x).norm().item() / (1e-5 + f0.norm().item()))
            if (res[-1] < tol):
                break
        return f0, res


    class DEQFixedPoint(nn.Module):
        def __init__(self, f, solver, **kwargs):
            super().__init__()
            self.f = f
            self.solver = solver
            self.kwargs = kwargs

            self.k_forward  = 0
            self.k_backward = 0

        def forward(self, x):
            # compute forward pass and re-engage autograd tape
            with torch.no_grad():
                z, self.forward_res, self.k_forward = self.solver(lambda z : self.f(z, x), torch.zeros_like(x), **self.kwargs)
            z = self.f(z,x)

            # set up Jacobian vector product (without additional forward calls)
            z0 = z.clone().detach().requires_grad_()
            f0 = self.f(z0,x)
            def backward_hook(grad):
                g, self.backward_res, self.k_backward = self.solver(lambda y : autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
                                                   grad, **self.kwargs)
                return g

            z.register_hook(backward_hook)
            return z
    
    #############################################################################
    ### Initialise Parameters ###################################################
    #################################
    
    K = torch.eye(number_of_pixels**2).to(device)
    if inpainting:
        K = mask_horizontal(percentage,K)

    my_K = op_K(deblurring=deblurring, denoising=denoising, inpainting=inpainting, my_conv=my_conv, my_conv_t=my_conv_t, K=K).to(device)

    if chosen_model=='bilevel_optimisation':
        print("Loading bilevel_optimisation model")
        f = bilevel_optimisation(number_of_pixels, my_K, tau, lambd, gamma, R=chosen_regulariser)
        
    if chosen_model=='DEQ_bilevel_optimisation':
        print("Loading DEQ_bilevel_optimisation model")
        f = DEQ_bilevel_optimisation(number_of_pixels, my_K, tau, lambd, gamma, R=chosen_regulariser)

    try: 
        print(f'Activation function: {f.activation_function}')
    except:
        print()


    model = DEQFixedPoint(f, anderson, tol=tol, max_iter=max_iter, m=5).to(device)
    
    initial = 'initial_'

    #############################################################################
    ### Load dataset ############################################################
    #################################

    # MNIST data loader
    transform_MNIST = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])
    MNIST_train = datasets.MNIST("./data", train=True, download=True, transform=transform_MNIST) 
    MNIST_test  = datasets.MNIST("./data", train=False, download=True, transform=transform_MNIST) 
    train_loader = DataLoader(MNIST_train, batch_size = 100, shuffle=True, num_workers=0)
    test_loader  = DataLoader(MNIST_test,  batch_size = 100, shuffle=True, num_workers=0) 


    #############################################################################
    ### Train/Test function #####################################################
    #################################
    # standard training or evaluation loop

    def epoch(loader, model, epsilon=0.0, K=None, opt=None, lr_scheduler=None):
        loss_function = nn.MSELoss()
        total_loss, total_err = 0.,0.
        sum_anderson_forward, sum_anderson_backward = 0.,0.
        sum_anderson_forward_res, sum_anderson_backward_res = 0.,0.
        
        cnt = 0
        model.eval() if opt is None else model.train()
        for X,_ in loader:
            cnt+=1
            X = X.view(X.size()[0], 1, 1, number_of_pixels**2).to(device)
            y_delta = X.clone() 
            
            gaussian_noise = torch.randn(X.size()).to(device)
            if inpainting or deblurring:
                with torch.no_grad():
                    y_delta = f.K(X)
            
                    
            y_delta += gaussian_noise*epsilon

            yp = model(y_delta).view(X.size())
            loss = loss_function(yp,X)
            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
                lr_scheduler.step()

            total_loss += loss.item() * X.shape[0]

            sum_anderson_forward  += model.k_forward
            sum_anderson_backward += model.k_backward
            sum_anderson_forward_res  += model.forward_res[-1]
            try: 
                sum_anderson_backward_res += model.backward_res[-1]
            except:
                print('model.backward_res[-1] cannot be accessed. I put the value to 100')
                sum_anderson_backward_res += 100
                
        return total_err / len(loader.dataset), total_loss / len(loader.dataset), sum_anderson_forward/cnt, sum_anderson_backward/cnt, sum_anderson_forward_res/cnt, sum_anderson_backward_res/cnt 


    #############################################################################
    ### Choose optimiser ########################################################
    #################################

    opt = optim.Adam(model.parameters(), lr=1e-3)
    print("# Parameters: ", sum(a.numel() for a in model.parameters()))

    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, max_epochs*len(train_loader), eta_min=1e-6)


    #############################################################################
    ### Initialise memory variables #############################################
    #########################################
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
    
    
    #############################################################################
    ### Load checkpoint if it exists ############################################
    #########################################
    
    pathlib.Path(f'./saved_data/{device.type}_{uniqueID:03}').mkdir(parents=True, exist_ok=True)
    
    initial = 'initial_'
    this_epoch = 0
    try:
        checkpoint = torch.load(f'./saved_data/{device.type}_{uniqueID:03}/checkpoint.pt')
        this_epoch = checkpoint['this_epoch']
        this_epoch += 1
        model.load_state_dict(checkpoint['model'])
        opt.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
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
        print("Checkpoint found.")
        initial = ''

    except:
        print("Checkpoint not found. I will look for an initial checkpoint")
        try:
            checkpoint = torch.load(f'./saved_data/{device.type}_{uniqueID:03}/{initial}checkpoint.pt')
            this_epoch = checkpoint['this_epoch']
            this_epoch += 1
            model.load_state_dict(checkpoint['model'])
            opt.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
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
            print("Checkpoint found")
            initial = ''

        except:
            print("No checkpoint found.\nThe initial checkpoint will be computed")
            max_epochs = 0 # Compute the first epoch only

            


    #############################################################################
    ### Start Training/Testing ##################################################
    #################################

    for i in range(this_epoch, max_epochs+1):
        start_time = time.time()    
        
        if max_epochs==0:
            train_error_perc, train_loss, average_anderson_forw_train, average_anderson_back_train, average_anderson_forw_res_train, average_anderson_back_res_train = epoch(train_loader, model, epsilon, K)
        else:
            train_error_perc, train_loss, average_anderson_forw_train, average_anderson_back_train, average_anderson_forw_res_train, average_anderson_back_res_train = epoch(train_loader, model, epsilon, K, opt, scheduler)
        test_error_perc, test_loss, average_anderson_forw_test, average_anderson_back_test, average_anderson_forw_res_test, average_anderson_back_res_test = epoch(test_loader,  model, epsilon, K)

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
        print(f'Time: {time.time() - start_time}')
        print()
        
        ## Save checkpoint
        checkpoint = { 
            'this_epoch': i,
            'model': model.state_dict(),
            'optimizer': opt.state_dict(),
            'scheduler': scheduler.state_dict(),
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
        torch.save(checkpoint, f'./saved_data/{device.type}_{uniqueID:03}/{initial}checkpoint.pt')

    return model


def run_main(**exp):
    start_total_time = time.time()
    _ = main(**exp)
    end_total_time = time.time()
    true_total_time = elapsed_time_from_beginning()
    print()
    print("Main:")
    print(f'The total amount of time for this config file was {datetime.timedelta(seconds=end_total_time-start_total_time)} (h:mm:ss)')
    print(f'or {(end_total_time-start_total_time):.2f} seconds.')
    print()
    print("Overall:")
    print(f'The total amount of time for this config file was {datetime.timedelta(seconds=true_total_time)} (h:mm:ss)')
    print(f'or {(true_total_time):.2f} seconds.')
    print('')#, flush=True)

if __name__ == "__main__":
    try:
        with open(sys.argv[-1], 'r') as f:
            exp = json.loads(f.read())
    except:
        print("No valid json file has been given in input")
        print("I will go through *THE FIRST* config file.")#, flush=True)
        for element in sorted(os.listdir('./config')):
            if element.endswith('.json'):
                with open(f'config/{element}', "r") as f:
                    exp = json.loads(f.read())

                try:
                    run_main(**exp)

                except:
                    print()
                    print(f'Something was wrong with config file, ID = {exp["uniqueID"]}')
                    print(f'Error: {sys.exc_info()[0]}')
                    traceback.print_exc()

                print('')#, flush=True)
                break
    else:
        run_main(**exp)