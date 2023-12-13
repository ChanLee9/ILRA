import json
import pandas as pd
import numpy as np
import torch
import loralib as lora
from tqdm import tqdm
import torch.nn as nn
from dataclasses import dataclass
from transformers import get_scheduler
from transformers import RobertaModel, AutoTokenizer

from sklearn.metrics import classification_report

from preprocess import *
from model import *

WEIGHT_DECAY = 1e-2

def get_optimizer(model, dataloader, config):
    optimizer = torch.optim.AdamW(filter(lambda p:p.requires_grad, model.parameters()), lr=config.lr, weight_decay=WEIGHT_DECAY)
    lr_scheduler = get_scheduler(
        'linear',
        optimizer,
        num_warmup_steps=200,
        num_training_steps=config.num_epochs*len(dataloader)
    )
    return optimizer, lr_scheduler
    
def train_loop(dataloader, model, optimizer, lr_scheduler, epoch):     # 一轮训练
    total_loss = 0.
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'epoch: {epoch}, loss: {0:>4f}')
    
    model.train()  
    for batch, item in enumerate(dataloader):
        loss = model.loss_fn(item)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        optimizer.zero_grad()
        
        total_loss += loss.item()
        progress_bar.set_description(f'epoch: {epoch+1}, loss: {total_loss/(batch+1)}')
        progress_bar.update(1)
    return total_loss


def dev_loop(dataloader, model):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'validating... ')
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for batch, item in enumerate(dataloader):
            y_true.extend(item['label'])
            y_pred.extend(model(item)[1].cpu().numpy())
            progress_bar.update(1)
        print(f'dev result: \n {classification_report(y_true, y_pred)}')
    return y_pred


def test_loop(dataloader, model):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'testing... ')
    model.eval()
    y_pred = []
    with torch.no_grad():
        for batch, item in enumerate(dataloader):
            y_pred = model(item).cpu().numpy()
        return y_pred

def print_trainable_params(model):
    """打印模型可训练参数量占比

    Args:
        model (_type_): _description_

    """
    total_params = sum([p.numel() for p in model.parameters()])
    trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    trainable_ratio = trainable_params / total_params
    print(f'total parameters: {total_params} || trainable parameters: {trainable_params} || trainable ratio: {100*trainable_ratio:.2f}%'
        )

def apply_lora(base_model, config):
    if config.task_type == "NLU":
        num_layers = len(base_model.encoder.layer)
        for ly in range(num_layers):
            # query
            query_dim = base_model.encoder.layer[ly].attention.self.query.weight.shape
            base_model.encoder.layer[ly].attention.self.query = lora.Linear(
                in_features=query_dim[0],
                out_features=query_dim[1],
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.dropout
            )
            # value
            value_dim = base_model.encoder.layer[ly].attention.self.value.weight.shape
            base_model.encoder.layer[ly].attention.self.value = lora.Linear(
                in_features=value_dim[0],
                out_features=value_dim[1],
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.dropout
            )
    
    # mark only lora module as trainable
    lora.mark_only_lora_as_trainable(base_model)
  
def apply_bit_fit(base_model, config):
    # freeze all parameters except bias
    if config.task_type == "NLU":
        # freeze all parameters
        for _, p in base_model.named_parameters():
            p.requires_grad = False
        
        # unfreeze bias
        num_layers = len(base_model.encoder.layer)
        for ly in range(num_layers):
            base_model.encoder.layer[ly].attention.self.query.bias.requires_grad = True
            base_model.encoder.layer[ly].attention.self.value.bias.requires_grad = True
            base_model.encoder.layer[ly].attention.self.key.bias.requires_grad = True
            base_model.encoder.layer[ly].attention.output.dense.bias.requires_grad = True
            base_model.encoder.layer[ly].intermediate.dense.bias.requires_grad = True
            base_model.encoder.layer[ly].output.dense.bias.requires_grad = True
        base_model.pooler.dense.bias.requires_grad = True

def get_base_model(config):
    if config.task_type == "NLU":
        base_model = RobertaModel.from_pretrained(config.model_name_or_path)
    return base_model


def get_customed_model(config):
    if config.task_type == "NLU":
        base_model = RobertaModel.from_pretrained(config.model_name_or_path)
        if config.method == "full-finetuning":
            pass
        elif config.method == "lora":
            apply_lora(base_model, config)
        elif config.method == "bit_fit":
            apply_bit_fit(base_model, config)
        model = NLU_Model(base_model, config).to(config.device)
    return model

if __name__ == "__main__":
    @dataclass
    class Config:
        dataset_name = "MRPC"
        batch_size = 4
        max_length = 512
        model_name_or_path = "../pretrained_models/roberta"
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        lr = 1e-5
        epochs = 3
        num_tags = 2
        task_type = "NLU"
    
    config = Config()
    
    train, dev, test = read_data(config)
    train_dataset = MyDataset(config, train)
    test_dataset = MyDataset(config, test)
    dev_dataset = MyDataset(config, dev)
    
    train_dataloader = MyDataLoader(train_dataset, config, shuffle=True)
    test_dataloader = MyDataLoader(test_dataset, config, shuffle=False)
    dev_dataloader = MyDataLoader(dev_dataset, config, shuffle=False)
    
    model = get_customed_model(config)
    optimizer, lr_scheduler = get_optimizer(model, train_dataloader, config)
    
    for epoch in range(config.epochs):
        train_loop(train_dataloader, model, optimizer, lr_scheduler, epoch)
        dev_loop(dev_dataloader, model)
    
