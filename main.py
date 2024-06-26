import time
import copy
from datetime import datetime
import json

import torch
import argparse

from training import *
from preprocess import *
from model import *

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import classification_report, matthews_corrcoef, accuracy_score, f1_score

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
    
    # ilra
    parser.add_argument("--k", default=1/5, type=float, help="k of ilra")
    
    parser.add_argument("--ablation", default=0, type=int, help="ablation study")
    
    return parser.parse_args()


if __name__ == '__main__':
    start_time = time.time()
    config = get_args()
    
    assert config.method in ["fft", "lora", "ilra", "pa", "bit_fit", "krona", "pilra"]
    
    if config.method == "ilra":
        config.residual_connection = False
    else:
        config.residual_connection = False
    config.modules_to_apply = config.modules_to_apply.split(',')
    if config.dataset_name in ["CoLA", "SST-2", "MRPC", "QQP", "QNLI", "RTE", "WNLI"]:
        config.num_tags = 2
    elif config.dataset_name in ["MNLI-m", "MNLI-mm"]:
        config.num_tags = 3
    
    train, dev = read_data(config)
    train_dataset = MyDataset(config, train)
    # test_dataset = MyDataset(config, test)
    dev_dataset = MyDataset(config, dev)
    
    train_dataloader = MyDataLoader(train_dataset, config, shuffle=True)
    # test_dataloader = MyDataLoader(test_dataset, config, shuffle=True)
    dev_dataloader = MyDataLoader(dev_dataset, config, shuffle=True)
    
    model = get_customed_model(config)
    
    # print configs
    print("\n"*3)
    print(f"configs listed below:\n")
    for key in config.__dict__: 
        print(f"{key}: {config.__dict__[key]}")
    # if config.method == "ilra":
    #     config.ilra_dims = model.base_model.encoder.ilra_dims
    #     for ly in config.ilra_dims:
    #         print(f"layer{ly}: ilra dims: {config.ilra_dims[ly]}")
    # elif config.method == "pilra":
    #     config.pilra_dims = model.base_model.encoder.pilra_dims
    #     for ly in config.pilra_dims:
    #         print(f"layer{ly}: pilra dims: {config.pilra_dims[ly]}")
    trainable_params, trainable_ratio = print_trainable_params(model)
    config.num_trainable_params = trainable_params
    config.trainable_ratio = trainable_ratio

    optimizer, lr_scheduler = get_optimizer(model, train_dataloader, config)
    
    # training
    best_res = 0.
    best_acc = 0.
    best_f1 = 0.
    for epoch in range(config.num_epochs):
        train_loop(train_dataloader, model, optimizer, lr_scheduler, epoch)
        y_pred, y_true = dev_loop(dev_dataloader, model)
        if config.dataset_name == "CoLA":
            res = matthews_corrcoef(y_true, y_pred)
            print(f'dev result: \n {res}')
        elif config.dataset_name in ["RTE", "SST-2", "MNLI-m", "MNLI-mm", "QNLI", "WNLI"]:
            res = accuracy_score(y_true, y_pred)
            print(f'dev result: \n {classification_report(y_true, y_pred)}')
        elif config.dataset_name in ["MRPC", "QQP"]:
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            res = (acc + f1) / 2
            if res > best_res:
                best_acc, best_f1 = acc, f1
            print(f'dev result: \n {classification_report(y_true, y_pred)}')
        
        elif config.dataset_name == "STS-B":
            pearsonr_res = pearsonr(y_true, y_pred)
            spearmanr_res = spearmanr(y_true, y_pred)
            res = (pearsonr_res[0] + spearmanr_res[0]) / 2
            print(f'dev result: \n pearsonr: {pearsonr_res[0]}, spearmanr: {spearmanr_res[0]}')
        
        if res > best_res:
            best_res = res
            # print("copy best model state dict\n")
            # best_model_states = copy.deepcopy(model.state_dict())
    
    # print("testing...\n")
    # # testing 
    # if config.do_test == 1:
    #     print("loading best model state dict...\n")
    #     model.load_state_dict(best_model_states)
    #     y_pred, y_true = dev_loop(test_dataloader, model)
    #     if config.dataset_name == "CoLA":
    #         res = matthews_corrcoef(y_true, y_pred)
    #         print(f'test result: \n {res}')
    #     elif config.dataset_name == "RTE":
    #         res = accuracy_score(y_true, y_pred)
    #         print(f'test result: \n {classification_report(y_true, y_pred)}')
    #     elif config.dataset_name == "MRPC":
    #         acc = accuracy_score(y_true, y_pred)
    #         f1 = f1_score(y_true, y_pred)
    #         res = (acc + f1) / 2
    #         print(f'test result: \n {classification_report(y_true, y_pred)}')
    
    if config.dataset_name == "CoLA":
        config.matthews_corrcoef_res= best_res
    elif config.dataset_name in ["RTE", "SST-2", "MNLI-m", "MNLI-mm", "QNLI", "WNLI"]:
        config.acc = best_res
    elif config.dataset_name in ["MRPC", "QQP"]:
        config.acc = best_acc
        config.f1 = best_f1
    elif config.dataset_name == "STS-B":
        config.pearsonr = pearsonr_res[0]
        config.spearmanr = spearmanr_res[0]
        
    end_time = time.time()  
    consumed_time = end_time - start_time
    config_dict = config.__dict__
    config_dict["consumed_time"] = consumed_time
    if config.ablation == 1:
        with open(f"ablation/{config.dataset_name}/{config.method}_{str(best_res)}.json", "w") as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=4)
            print(f"save result to ablation/{config.dataset_name}/{config.method}_{str(best_res)}.json")
    else:
        with open(f"result/{config.dataset_name}/{config.method}_{str(round(best_res, 6))}.json", "w") as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=4)
            print(f"save result to result/{config.dataset_name}/{config.method}_{str(round(best_res, 6))}.json")
    