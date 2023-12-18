import pandas as pd
import torch
import torch.nn as nn
import torch.functional as F
from customed_transformers import RobertaModel, AutoTokenizer

class NLU_Model(nn.Module):
    def __init__(self, base_model, config) -> None:
        super().__init__()
        self.base_model = base_model
        hidden_size = self.base_model.pooler.dense.weight.shape[1]
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act_fn = nn.Tanh()
        self.dropout = nn.Dropout(config.dropout)    
        self.fc = nn.Linear(hidden_size, config.num_tags)
        self.device = config.device
        self.max_length = config.max_length
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    
    def loss_fn(self, item):
        criterion = nn.CrossEntropyLoss()
        y_true = torch.LongTensor(item['label']).to(self.device)
        logits, y_pred = self.forward(item)
        loss = criterion(logits, y_true)
        return loss
                
    def forward(self, item):
        sentences1 = item['sentence1']    
        if "sentence2" in item.keys():
            sentences2 = item['sentence2']
            text_encodings = self.tokenizer(sentences1,
                                            sentences2,
                                            truncation=True,
                                            padding=True,
                                            max_length=self.max_length,
                                            return_tensors='pt')
        else:
            text_encodings = self.tokenizer(sentences1,
                                            truncation=True,
                                            padding=True,
                                            max_length=self.max_length,
                                            return_tensors='pt')
        text_encodings = text_encodings.to(self.device)
        logits = self.base_model(**text_encodings)
        logits = self.dropout(logits.pooler_output)
        logits = self.dense(logits)
        logits = self.act_fn(logits)
        logits = self.dropout(logits)
        logits = self.fc(logits)
        y_pred = torch.argmax(logits, dim=1)
        return logits, y_pred
            

class NLG_Model(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
    
