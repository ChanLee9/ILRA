import time
from datetime import datetime
import json

import torch
import argparse

from training import *
from preprocess import *
from model import *

from sklearn.metrics import classification_report

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
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--num_epochs", default=10, type=int, help="epochs")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="weight decay")
    # parser.add_argument("--warmup_steps", default=100, type=int, help="warmup steps")
    parser.add_argument("--method", type=str, help="method")
    parser.add_argument("--dropout", default=0.1, type=float, help="dropout ratio")
    parser.add_argument("--modules_to_apply", default="query,value", type=str, help="modules to apply")
    parser.add_argument("--scaling_alpha", default=8, type=float, help="lora alpha")
    parser.add_argument("--scale", default=1, type=float, help="whether to scale")
    
    # test
    parser.add_argument("--do_test", default=0, type=int, help="do test")
    
    # lora
    parser.add_argument("--lora_r", default=2, type=int, help="lora r")
    
    # krona
    parser.add_argument("--krona_dim", default=2, type=int, help="lora r")
    parser.add_argument("--krona_dropout", default=0.1, type=float, help="dropout ratio")
    
    
    # evaluation
    
    return parser.parse_args()


if __name__ == '__main__':
    start_time = time.time()
    config = get_args()
    if config.method == "ilra":
        config.residual_connection = False
    else:
        config.residual_connection = False
    config.modules_to_apply = config.modules_to_apply.split(',')
    
    train, dev, test = read_data(config)
    train_dataset = MyDataset(config, train)
    test_dataset = MyDataset(config, test)
    dev_dataset = MyDataset(config, dev)
    
    train_dataloader = MyDataLoader(train_dataset, config, shuffle=True)
    test_dataloader = MyDataLoader(test_dataset, config, shuffle=True)
    dev_dataloader = MyDataLoader(dev_dataset, config, shuffle=True)
    
    model = get_customed_model(config)
    
    # print configs
    print("\n"*3)
    print(f"configs listed below:\n")
    for key in config.__dict__: 
        print(f"{key}: {config.__dict__[key]}")
    if config.method == "ilra":
        config.ilra_dims = model.base_model.encoder.ilra_dims
        for ly in config.ilra_dims:
            print(f"layer{ly}: ilra dims: {config.ilra_dims[ly]}")
    print_trainable_params(model)

    optimizer, lr_scheduler = get_optimizer(model, train_dataloader, config)
    # training
    for epoch in range(config.num_epochs):
        train_loop(train_dataloader, model, optimizer, lr_scheduler, epoch)
        y_pred, y_true = dev_loop(dev_dataloader, model)
    
    # evaluation
    y_pred, y_true = dev_loop(dev_dataloader, model)
    dev_res = classification_report(y_true, y_pred, output_dict=True)
    print(f'final dev res\n')
    for key in dev_res:
        print(f"{key}: {dev_res[key]}")
    f1_score = round(dev_res["weighted avg"]["f1-score"], 5)
    
    # test 
    if config.do_test:
        y_pred = test_loop(test_dataloader, model)
    
    end_time = time.time()  
    consumed_time = end_time - start_time
    
    # save result
    # cur_time = datetime.now().strftime(r"%Y-%m-%d_%H_%M")
    config_dict = config.__dict__
    config_dict["consumed_time"] = consumed_time
    config_dict.update(dev_res)
    with open(f"result/{config.dataset_name}/{config.method}_{str(f1_score)}.json", "w") as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=4)
        print(f"save result to result/{config.dataset_name}/{config.method}_{str(f1_score)}.json")
    
