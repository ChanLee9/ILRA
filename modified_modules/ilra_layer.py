import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math
from typing import Optional, List

from dataclasses import dataclass
from transformers import RobertaModel
from sklearn.decomposition import PCA


class IlraDim():
    def __init__(self, config) -> None:
        self.config = config
        self.model = RobertaModel.from_pretrained(config.model_name_or_path)
        self.ilra_dims = self.get_ilra_dims(config)
    
    def get_ilra_dims(self, config):
        """
        ilra_dims: {layer_idx: {module: dim}}
        """
        ilra_dims = {}  
        for i in range(config.num_hidden_layers):
            ilra_dims[i] = {module: 0 for module in config.modules_to_apply}
        for ly in range(self.config.num_hidden_layers):
            for module in self.config.modules_to_apply:
                ilra_dims[ly][module] = self.get_single_dim(ly, module, config.k)
        return ilra_dims

    def get_single_dim(self, ly, module, k):
        # get the turning dim of current matrix
        pca = PCA(n_components=10)
        if module == "query":
            mx = self.model.encoder.layer[ly].attention.self.query.weight.detach().cpu().numpy()
        elif module == "key":
            mx = self.model.encoder.layer[ly].attention.self.key.weight.detach().cpu().numpy()
        elif module == "value":
            mx = self.model.encoder.layer[ly].attention.self.value.weight.detach().cpu().numpy()
        elif module == "output":
            mx = self.model.encoder.layer[ly].attention.output.dense.weight.detach().cpu().numpy()
        elif module == "ffn1":
            mx = self.model.encoder.layer[ly].intermediate.dense.weight.detach().cpu().numpy()
        elif module == "ffn2":
            mx = self.model.encoder.layer[ly].output.dense.weight.detach().cpu().numpy()
        else:
            raise NotImplementedError
        pca.fit(mx)
        variance_ratio = pca.explained_variance_ratio_ 
        variance_ratio_diff = variance_ratio[1:] - variance_ratio[:-1]
        
        d_c = 0
        for j in range(len(variance_ratio_diff)-1):
            if variance_ratio_diff[j+1] / variance_ratio_diff[j] < k:
                d_c = j
                break
        # breakpoint()
        v_m = np.mean(variance_ratio_diff[d_c:])
            
        for j in range(d_c, len(variance_ratio_diff)):
            if variance_ratio_diff[j] >= v_m:
                pca_dim = j
                break
        
        v_m = np.mean(variance_ratio_diff[pca_dim:])
        
        for j in range(d_c, len(variance_ratio_diff)):
            if variance_ratio_diff[j] >= v_m:
                return j + 1
        
        return len(variance_ratio_diff)

    def check_ilra_dim(self, ilra_dim, in_features, out_features):
        # make sure that krona_dim is divisible by both in_features and out_features
        min_dim = min(in_features, out_features)
        ilra_dim_left, ilra_dim_right = ilra_dim, ilra_dim
        while min_dim % ilra_dim_left and min_dim % ilra_dim_right:
            ilra_dim_left -= 1
            ilra_dim_right += 1
        if min_dim % ilra_dim_left == 0:
            ilra_dim = ilra_dim_left
        elif min_dim % ilra_dim_right == 0:
            ilra_dim = ilra_dim_right
        return ilra_dim


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


# class Linear(nn.Linear, KronALayer):
#     # LoRA implemented in a dense layer
#     def __init__(
#         self, 
#         in_features: int, 
#         out_features: int, 
#         krona_dim: int = 0, 
#         krona_alpha: int = 1, 
#         krona_dropout: float = 0.,
#         fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
#         merge_weights: bool = True,
#         **kwargs
#     ):
#         nn.Linear.__init__(self, in_features, out_features, **kwargs)
#         KronALayer.__init__(self, krona_dim=krona_dim, krona_alpha=krona_alpha, krona_dropout=krona_dropout,
#                            merge_weights=merge_weights)

#         self.fan_in_fan_out = fan_in_fan_out
        
#         # Actual trainable parameters
#         if krona_dim > 0:
#             self.krona_A = nn.Parameter(self.weight.new_zeros((krona_dim, out_features//krona_dim)))
#             self.krona_B = nn.Parameter(self.weight.new_zeros((in_features//krona_dim, krona_dim)))
#             self.scaling = self.krona_alpha / self.krona_dim
            
#             # Freezing the pre-trained weight matrix
#             self.weight.requires_grad = False
#         self.reset_parameters()
#         if fan_in_fan_out:
#             self.weight.data = self.weight.data.transpose(0, 1)
    
#     def reset_parameters(self):
#         nn.Linear.reset_parameters(self)
#         if hasattr(self, 'krona_A'):
#             # initialize krona_A the same way as the default for nn.Linear and krona_B to zero
#             nn.init.kaiming_uniform_(self.krona_A, a=math.sqrt(5))
#             nn.init.zeros_(self.krona_B)

#     def train(self, mode: bool = True):
#         nn.Linear.train(self, mode)
#         if mode:
#             if self.merge_weights and self.merged:
#                 # Make sure that the weights are not merged
#                 if self.krona_dim > 0:
#                     mx = self.compute_krona_matrix()
#                     self.weight.data -= mx.transpose(0, 1)
#                 self.merged = False
#         else:
#             if self.merge_weights and not self.merged:
#                 # Merge the weights and mark it
#                 if self.krona_dim > 0:
#                     mx = self.compute_krona_matrix()
#                     self.weight.data += mx.transpose(0, 1)
#                 self.merged = True       

#     def forward(self, x: torch.Tensor):
#         def T(w):
#             return w.transpose(0, 1) if self.fan_in_fan_out else w
#         if self.krona_dim > 0 and not self.merged:
#             result = F.linear(x, T(self.weight), bias=self.bias)
#             mx = self.compute_krona_matrix()
#             krona_product_matrix = mx*self.scaling
#             result += self.krona_dropout(x)@krona_product_matrix
#             # add residual Connection
#             # return result + X
#             return result
#         else:
#             return F.linear(x, T(self.weight), bias=self.bias)
    
#     def compute_krona_matrix(self):
#         # compute kroneker matrix
#         krona_product = torch.einsum('ij,kl->ikjl', self.krona_A, self.krona_B)   
#         krona_product = krona_product.view(self.krona_A.size(0) * self.krona_B.size(0), self.krona_A.size(1) * self.krona_B.size(1)) 
#         return krona_product*self.scaling

def mark_only_ilra_as_trainable(model: nn.Module, bias: str = 'none') -> None:
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


if __name__ == "__main__":
    @dataclass
    class Config:
        dataset_name = "MRPC"
        batch_size = 4
        k = 0.2
        max_length = 512
        model_name_or_path = "../pretrained_models/roberta"
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        lr = 1e-5
        epochs = 3
        num_tags = 2
        task_type = "NLU"
        num_hidden_layers = 12
        hidden_size = 768
        intermediate_size = 3072
        modules_to_apply = "query,value,key,output,ffn1,ffn2"
    config = Config()
    config.modules_to_apply = config.modules_to_apply.split(',')
    ilra_dim = IlraDim(config)
    print(ilra_dim.ilra_dims)
