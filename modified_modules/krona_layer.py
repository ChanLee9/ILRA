import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List


class KronALayer():
    def __init__(
        self, 
        krona_dim: int, 
        krona_alpha: int, 
        krona_dropout: float,
        merge_weights: bool,
    ):
        self.krona_dim = krona_dim
        self.krona_alpha = krona_alpha
        
        # Optional dropout
        if krona_dropout > 0.:
            self.krona_dropout = nn.Dropout(p=krona_dropout)
        else:
            self.krona_dropout = lambda x: x
            
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class Linear(nn.Linear, KronALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        krona_dim: int = 0, 
        krona_alpha: int = 1, 
        krona_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        KronALayer.__init__(self, krona_dim=krona_dim, krona_alpha=krona_alpha, krona_dropout=krona_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        
        # Actual trainable parameters
        if krona_dim > 0:
            self.krona_A = nn.Parameter(self.weight.new_zeros((krona_dim, out_features//krona_dim)))
            self.krona_B = nn.Parameter(self.weight.new_zeros((in_features//krona_dim, krona_dim)))
            self.scaling = self.krona_alpha / self.krona_dim
            
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)
    
    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'krona_A'):
            # initialize krona_A the same way as the default for nn.Linear and krona_B to zero
            nn.init.kaiming_uniform_(self.krona_A, a=math.sqrt(5))
            nn.init.zeros_(self.krona_B)

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.krona_dim > 0:
                    self.weight.data -= self.compute_krona_matrix()
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.krona_dim > 0:
                    self.weight.data += self.compute_krona_matrix()
                self.merged = True       

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.krona_dim > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            krona_product_matrix = self.compute_krona_matrix()*self.scaling
            # print(krona_product_matrix)
            result += self.krona_dropout(x)@krona_product_matrix
            # add residual Connection
            # return result + X
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)
    
    def compute_krona_product(self, x):
        krona_res = self.krona_A.transpose(0, 1)@self.krona_dropout(x.view(x.size(0), x.size(1), self.krona_dim, -1))@self.krona_B
        return krona_res.view(x.size(0), x.size(1), -1)
    
    def compute_krona_matrix(self):
        # compute kroneker matrix
        krona_product = torch.einsum('ij,kl->ikjl', self.krona_A, self.krona_B)   
        krona_product = krona_product.view(self.krona_A.size(0) * self.krona_B.size(0), self.krona_A.size(1) * self.krona_B.size(1)) 
        return krona_product*self.scaling

def mark_only_krona_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'krona_' not in n:
            p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'krona_only':
        for m in model.modules():
            if isinstance(m, KronALayer) and \
                hasattr(m, 'bias') and \
                m.bias is not None:
                    m.bias.requires_grad = True
    else:
        raise NotImplementedError
