#!/usr/bin/env python
# coding: utf-8

# Create all config files

import itertools
import json
import pathlib

pathlib.Path('./config').mkdir(parents=True, exist_ok=True)


#inpainting
v_percentage = [1/3]
# #denoising
v_epsilon = [0.0, 0.05, 0.1, 0.5, 1.0] 

v_inpainting_denoising_deblurring = [[True, False, False],[False, True, False],[False, False, True]]
v_tau = [0.01, 0.1, 0.9, 1.1, 2.1]
v_lambd = [1.0]
v_max_epochs = [100] 
v_number_of_pixels = [28] 
v_tol = [1e-3] 
v_max_iter = [50]
v_gamma = [0.1, 0.5, 1.0]

v_blur_kernel_size = [5]

idx=0

v_number_of_layers = [1]
v_chosen_model = ['bilevel_optimisation']
v_chosen_regulariser = []
add_DEQ = ''
for cont in range(4,6+1):
    v_chosen_regulariser.append(f'grad_regulariser_{cont}{add_DEQ}')

for _, values in enumerate(itertools.product(v_percentage,
                                             v_epsilon,
                                             v_inpainting_denoising_deblurring,
                                             v_tau,
                                             v_lambd,
                                             v_max_epochs,
                                             v_number_of_layers,
                                             v_number_of_pixels,
                                             v_tol,
                                             v_max_iter,
                                             v_chosen_model,
                                             v_blur_kernel_size,
                                             v_gamma,
                                             v_chosen_regulariser)):

    idx += 1
    
    data = {}
    data['uniqueID'] = idx

    data['percentage']                      = values[0]
    data['epsilon']                         = values[1]
    data['inpainting_denoising_deblurring'] = values[2]
    data['tau']                             = values[3]
    data['lambd']                           = values[4]
    data['max_epochs']                      = values[5]
    data['number_of_layers']                = values[6]
    data['number_of_pixels']                = values[7]
    data['tol']                             = values[8]
    data['max_iter']                        = values[9]
    data['chosen_model']                    = values[10]
    data['blur_kernel_size']                = values[11]
    data['gamma']                           = values[12]
    data['chosen_regulariser']              = values[13]

    json_filename = f'config/config_{idx}.json'
    print(f'{idx}\n{data}\n')

    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

        
        
# Add DEQ_bilevel_optimisation
v_number_of_layers = [1]
v_chosen_model = ['DEQ_bilevel_optimisation']

v_chosen_regulariser = []
add_DEQ = '_DEQ'    
for cont in range(4,6+1):
    v_chosen_regulariser.append(f'grad_regulariser_{cont}{add_DEQ}')

idx_bilevel = idx

for _, values in enumerate(itertools.product(v_percentage,
                                             v_epsilon,
                                             v_inpainting_denoising_deblurring,
                                             v_tau,
                                             v_lambd,
                                             v_max_epochs,
                                             v_number_of_layers,
                                             v_number_of_pixels,
                                             v_tol,
                                             v_max_iter,
                                             v_chosen_model,
                                             v_blur_kernel_size,
                                             v_gamma,
                                             v_chosen_regulariser)):

    idx += 1
    
    data = {}
    data['uniqueID'] = idx

    data['percentage']                      = values[0]
    data['epsilon']                         = values[1]
    data['inpainting_denoising_deblurring'] = values[2]
    data['tau']                             = values[3]
    data['lambd']                           = values[4]
    data['max_epochs']                      = values[5]
    data['number_of_layers']                = values[6]
    data['number_of_pixels']                = values[7]
    data['tol']                             = values[8]
    data['max_iter']                        = values[9]
    data['chosen_model']                    = values[10]
    data['blur_kernel_size']                = values[11]
    data['gamma']                           = values[12]
    data['chosen_regulariser']              = values[13]

    
    json_filename = f'config/config_{idx}.json'
    print(f'{idx}\n{data}\n')

    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


print(f'Last Bilevel idx: {idx_bilevel}')
print(f'Last DEQ idx: {idx}')