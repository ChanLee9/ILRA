import torch
import argparse

from training import *
from preprocess import *
from model import *

def get_args():
    parser = argparse.ArgumentParser()
    
    # data
    parser.add_argument("--dataset_name", default="MRPC", type=str, help="dataset name")
    parser.add_argument("--task_type", default="NLU", type=str, help="task type")
    parser.add_argument("--batch_size", default=4, type=int, help="batch size")
    parser.add_argument("--max_length", default=512, type=int, help="max length of input")
    parser.add_argument("--num_tags", default=2, type=int, help="number of tags")
    
    # training
    parser.add_argument("--model_name_or_path", default="../pretrained_models/roberta", type=str, help="model name or path")
    parser.add_argument("--device", default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), type=str, help="device")
    parser.add_argument("--lr", default=1e-5, type=float, help="learning rate")
    parser.add_argument("--num_epochs", default=3, type=int, help="epochs")
    
    # evaluation
    
    return parser.parse_args()


if __name__ == '__main__':
    config = get_args()
    train, dev, test = read_data(config)
    train_dataset = MyDataset(config, train)
    test_dataset = MyDataset(config, test)
    dev_dataset = MyDataset(config, dev)
    
    train_dataloader = MyDataLoader(train_dataset, config, shuffle=True)
    test_dataloader = MyDataLoader(test_dataset, config, shuffle=False)
    dev_dataloader = MyDataLoader(dev_dataset, config, shuffle=False)
    
    model = get_model(config)
    optimizer, lr_scheduler = get_optimizer(model, train_dataloader, config)
    
    for epoch in range(config.num_epochs):
        train_loop(train_dataloader, model, optimizer, lr_scheduler, epoch)
        dev_loop(dev_dataloader, model)