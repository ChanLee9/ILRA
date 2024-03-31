import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math
from typing import Optional, List

from dataclasses import dataclass
from transformers import RobertaModel
from sklearn.decomposition import PCA


class PilraDim():
    def __init__(self, config) -> None:
        self.config = config
        self.model = RobertaModel.from_pretrained(config.model_name_or_path)
        self.pilra_dims = self.get_pilra_dims(config)
    
    def get_pilra_dims(self, config):
        """
        pilra_dims: {layer_idx: {module: [d_c, d_eid, d_m]}}
        """
        pilra_dims = {}  
        for i in range(config.num_hidden_layers):
            pilra_dims[i] = {module: 0 for module in config.modules_to_apply}
        for ly in range(self.config.num_hidden_layers):
            for module in self.config.modules_to_apply:
                dims = self.get_single_dim(ly, module, config.k)
                pilra_dims[ly][module] = dims
        return pilra_dims

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
        
        d_c = 1
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
                d_eid = j
                break
        
        d_m = np.argmax(variance_ratio_diff)
        d_m = max(d_m, d_eid+1)
        return [d_c, d_eid, d_m]

    def check_pilra_dim(self, ilra_dim, in_features, out_features):
        # make sure that krona_dim is divisible by both in_features and out_features
        # special case
        if ilra_dim in [1, 2, 3, 4, 6, 8]:
            return ilra_dim
        elif ilra_dim == 5:
            return 4
        elif ilra_dim == 7:
            return 6
        else:
            return 8


class PilraLayer():
    def __init__(
        self, 
        pilra_dims: List[int], 
        pilra_alpha: int, 
        pilra_dropout: float,
        merge_weights: bool,
    ):
        self.pilra_dims = pilra_dims
        self.pilra_alpha = pilra_alpha
        
        # Optional dropout
        if pilra_dropout > 0.:
            self.pilra_dropout = nn.Dropout(p=pilra_dropout)
        else:
            self.pilra_dropout = lambda x: x
            
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class Linear(nn.Linear, PilraLayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        pilra_dims: List[int], 
        pilra_alpha: int = 1, 
        pilra_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        PilraLayer.__init__(self, pilra_dims=pilra_dims, pilra_alpha=pilra_alpha, pilra_dropout=pilra_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        
        # Actual trainable parameters
        
        d_c, d_eid, d_m = sorted(pilra_dims)
        
        if d_c < d_eid < d_m:
            if d_c > 0:
                self.pilra_d_c_A = nn.Parameter(self.weight.new_zeros((d_c, in_features)))
                self.pilra_d_c_B = nn.Parameter(self.weight.new_zeros((out_features, d_c)))
            
            self.pilra_d_eid_A = nn.Parameter(self.weight.new_zeros((d_eid, in_features)))
            self.pilra_d_eid_B = nn.Parameter(self.weight.new_zeros((out_features, d_eid)))
            
            self.pilra_d_m_A = nn.Parameter(self.weight.new_zeros((d_m, in_features)))
            self.pilra_d_m_B = nn.Parameter(self.weight.new_zeros((out_features, d_m)))
        elif d_c == d_eid < d_m:
            if d_c > 0:
                self.pilra_d_c_A = nn.Parameter(self.weight.new_zeros((d_c, in_features)))
                self.pilra_d_c_B = nn.Parameter(self.weight.new_zeros((out_features, d_c)))
            
            self.pilra_d_m_A = nn.Parameter(self.weight.new_zeros((d_m, in_features)))
            self.pilra_d_m_B = nn.Parameter(self.weight.new_zeros((out_features, d_m)))
        elif d_c < d_eid == d_m:
            if d_c > 0:
                self.pilra_d_c_A = nn.Parameter(self.weight.new_zeros((d_c, in_features)))
                self.pilra_d_c_B = nn.Parameter(self.weight.new_zeros((out_features, d_c)))
            
            self.pilra_d_eid_A = nn.Parameter(self.weight.new_zeros((d_eid, in_features)))
            self.pilra_d_eid_B = nn.Parameter(self.weight.new_zeros((out_features, d_eid)))
        elif d_c == d_eid == d_m:
            if d_c > 0:
                self.pilra_d_c_A = nn.Parameter(self.weight.new_zeros((d_c, in_features)))
                self.pilra_d_c_B = nn.Parameter(self.weight.new_zeros((out_features, d_c)))
        else:
            breakpoint()
        
        # self.scaling = self.pilra_alpha / self.pilra_dim
            
            # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)
    
    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'pilra_d_c_A'):
            # initialize d_c_A the same way as the default for nn.Linear and d_c_B to zero
            nn.init.kaiming_uniform_(self.pilra_d_c_A, a=math.sqrt(5))
            nn.init.zeros_(self.pilra_d_c_B)
        if hasattr(self, 'pilra_d_eid_A'):
            # initialize d_eid_A the same way as the default for nn.Linear and d_eid_B to zero
            nn.init.kaiming_uniform_(self.pilra_d_eid_A, a=math.sqrt(5))
            nn.init.zeros_(self.pilra_d_eid_B)
        if hasattr(self, 'pilra_d_m_A'):
            # initialize d_m_A the same way as the default for nn.Linear and d_m_B to zero
            nn.init.kaiming_uniform_(self.pilra_d_m_A, a=math.sqrt(5))
            nn.init.zeros_(self.pilra_d_m_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if hasattr(self, 'pilra_d_c_A'):
                    self.weight.data -= T(self.pilra_d_c_B @ self.pilra_d_c_A) * self.pilra_alpha
                if hasattr(self, 'pilra_d_eid_A'):
                    self.weight.data -= T(self.pilra_d_eid_B @ self.pilra_d_eid_A) * self.pilra_alpha
                if hasattr(self, 'pilra_d_m_A'):
                    self.weight.data -= T(self.pilra_d_m_B @ self.pilra_d_m_A) * self.pilra_alpha
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if hasattr(self, 'pilra_d_c_A'):
                    self.weight.data += T(self.pilra_d_c_B @ self.pilra_d_c_A) * self.pilra_alpha
                if hasattr(self, 'pilra_d_eid_A'):
                    self.weight.data += T(self.pilra_d_eid_B @ self.pilra_d_eid_A) * self.pilra_alpha
                if hasattr(self, 'pilra_d_m_A'):
                    self.weight.data += T(self.pilra_d_m_B @ self.pilra_d_m_A) * self.pilra_alpha
                self.merged = True       

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if hasattr(self, 'pilra_d_c_A'):
                result += (self.pilra_dropout(x) @ self.pilra_d_c_A.transpose(0, 1)) @ self.pilra_d_c_B.transpose(0, 1) * self.pilra_alpha
            if hasattr(self, 'pilra_d_eid_A'):
                result += (self.pilra_dropout(x) @ self.pilra_d_eid_A.transpose(0, 1)) @ self.pilra_d_eid_B.transpose(0, 1) * self.pilra_alpha
            if hasattr(self, 'pilra_d_m_A'):
                result += (self.pilra_dropout(x) @ self.pilra_d_m_A.transpose(0, 1)) @ self.pilra_d_m_B.transpose(0, 1) * self.pilra_alpha
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)
    

def mark_only_pilra_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'pilra_' not in n:
            p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'pilra_only':
        for m in model.modules():
            if isinstance(m, PilraLayer) and \
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
    ilra_dim = PilraDim(config)
    print(ilra_dim.ilra_dims)
