import json
import pandas as pd
import numpy as np
import torch
import loralib as lora
from tqdm import tqdm
import torch.nn as nn
from dataclasses import dataclass
from transformers import get_scheduler
from customed_transformers import RobertaModel, RobertaConfig

from sklearn.metrics import classification_report, matthews_corrcoef, accuracy_score

from preprocess import *
from model import *
from methods import *

def get_optimizer(model, dataloader, config):
    
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.lr)
    total_warmup_steps=config.num_epochs*len(dataloader)
    lr_scheduler = get_scheduler(
        'linear',
        optimizer,
        num_warmup_steps=total_warmup_steps,
        num_training_steps=100
    )
    # print(f'num_training_steps: {int(0.08 * total_warmup_steps)}')
    return optimizer, lr_scheduler

def train_loop(dataloader, model, optimizer, lr_scheduler, epoch):     # 一轮训练
    total_loss = 0.
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'epoch: {epoch}, loss: {0:>4f}')
    
    model.train()  
    for batch, item in enumerate(dataloader):
        loss = model.loss_fn(item)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        total_loss += loss.item()
        progress_bar.set_description(f'epoch: {epoch+1}, loss: {total_loss/(batch+1):>4f}')
        progress_bar.update(1)
    return total_loss


def dev_loop(dataloader, model, config):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'validating... ')
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for batch, item in enumerate(dataloader):
            y_true.extend(item['label'])
            y_pred.extend(model(item)[1].cpu().numpy())
            progress_bar.update(1)
        if config.dataset_name == "CoLA":
            res = matthews_corrcoef(y_true, y_pred)
            print(f'dev result: \n {res}')
        else:
            res = accuracy_score(y_true, y_pred)
            print(f'dev result: \n {classification_report(y_true, y_pred)}')
    return y_pred, y_true, res


def test_loop(dataloader, model):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'testing... ')
    model.eval()
    y_pred = []
    with torch.no_grad():
        for batch, item in enumerate(dataloader):
            y_pred.extend(model(item)[1].cpu().numpy())
        return y_pred

def print_trainable_params(model):
    """打印模型可训练参数量占比

    Args:
        model (_type_): _description_

    """
    print_trainable_layers = False
    total_params = sum([p.numel() for p in model.parameters()])
    trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    trainable_ratio = trainable_params / total_params
    
    if print_trainable_layers:
        for n, p in model.named_parameters():
            if p.requires_grad:
                print(n, p.shape)
    print(f'total parameters: {total_params} || trainable parameters: {trainable_params} || trainable ratio: {100*trainable_ratio:.2f}%'
        )
    return trainable_params

def get_base_model(config):
    if config.task_type == "NLU":
        base_model = RobertaModel.from_pretrained(config.model_name_or_path)
    return base_model

def freeze_model(model):
    for n, p in model.named_parameters():
        p.requires_grad = False

def get_customed_model(config):
    if config.task_type == "NLU":
        roberta_config = RobertaConfig.from_pretrained(config.model_name_or_path)
        # add customed config to roberta_config before load roberta base model
        roberta_config.update(config.__dict__)
        base_model = RobertaModel.from_pretrained(config.model_name_or_path, config=roberta_config)
        base_model.config.method = config.method
        if config.method == "fft":
            pass
        elif config.method == "lora" or config.method == "ilra" or config.method == "pa":
            lora.mark_only_lora_as_trainable(base_model)
        elif config.method == "bit_fit":
            # freeze all parameters except bias (LayerNorm.bias excluded)
            for n, p in base_model.named_parameters():
                if "bias" not in n:
                    p.requires_grad = False
                elif "LayerNorm.bias" in n:
                    p.requires_grad = False
        elif config.method == "krona":
            krona.mark_only_krona_as_trainable(base_model)
        # freeze_model(base_model)
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
    
