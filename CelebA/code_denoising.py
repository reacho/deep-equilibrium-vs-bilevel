import torch
import torch.nn

class Denoising_operator:

    def __init__(self, transpose=False): 
        self.transpose = transpose

    def __matmul__(self, argument):
        return argument

    @property
    def T(self):
        return Denoising_operator(transpose=not self.transpose)