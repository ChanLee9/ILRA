import pandas as pd
import torch
import torch.nn as nn
import torch.functional as F
from transformers import RobertaModel, AutoTokenizer

class MyModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.base_model = RobertaModel.from_pretrained(config.model_name_or_path)
        self.fc = nn.Linear(768, config.num_tags)
    
    def loss_fn():
        criterion = nn.BCELoss()
        
