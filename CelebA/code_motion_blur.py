import torch
import torch.nn as nn

class MotionBlur:
    def __init__(self, kernel_blur, transpose=False, device=None):
        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.transpose = transpose
        conv_n_chan_out, conv_n_chan_in, conv_kernel_size, _ = kernel_blur.shape
        this_padding = conv_kernel_size//2
        
        if not self.transpose:
            self.useThisConv = nn.Conv2d (conv_n_chan_in, conv_n_chan_out, conv_kernel_size, padding = this_padding, bias = False).to(self.device).requires_grad_(False)
        else:
            self.useThisConv = torch.nn.ConvTranspose2d (conv_n_chan_out, conv_n_chan_in, conv_kernel_size, padding = this_padding, bias = False).to(self.device).requires_grad_(False)
        
        with torch.no_grad():
            _ = self.useThisConv.weight.copy_(kernel_blur)
        

    def __matmul__(self, argument):
        arg_shape = argument.shape
        return (self.useThisConv(argument.view(-1, 1, arg_shape[-2],arg_shape[-1]))).view(arg_shape)
    

    @property
    def T(self):
        return MotionBlur(self.useThisConv.weight, transpose=not self.transpose, device=self.device)

class MotionBlurSpecific(MotionBlur):
    def __init__(self, conv_n_chan_in, conv_n_chan_out, conv_kernel_size, blur_type='diagonal', padding_type='circular', transpose=False, device=None):
        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.transpose=transpose
        
        if blur_type == 'diagonal':
            kernel_matrix = torch.eye(conv_kernel_size)
        elif blur_type == 'random':
            torch.random.manual_seed(2022)
            kernel_matrix = torch.rand(conv_n_chan_out, conv_n_chan_in, conv_kernel_size, conv_kernel_size)
        else:
            raise Exception(f'Blur {blur_type} not implemented yet')
        
        kernel_matrix = ((kernel_matrix/kernel_matrix.sum())/conv_n_chan_out).to(self.device) # normalization

        kernel_matrix = torch.tile(kernel_matrix, (conv_n_chan_out, conv_n_chan_in,1,1))
        
        super().__init__(kernel_matrix, transpose=self.transpose, device=self.device)
