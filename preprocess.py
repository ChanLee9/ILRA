import pandas as pd
import os

from sklearn.model_selection import train_test_split
  
def read_data(data_dir, dataset_name):
    if dataset_name == "RTE":
        train_path = os.path.join(data_dir, "train.tsv")
        test_path = os.path.join(data_dir, "dev.tsv")
        dev_path = os.path.join(data_dir, "test.tsv")
        train_data = pd.read_csv(train_path, sep='\t', header=0)
        test_data = pd.read_csv(test_path, sep='\t', header=0)
        dev_data = pd.read_csv(dev_path, sep='\t', header=0)
        
    elif dataset_name == "CoLA":
        train_path = os.path.join(data_dir, "train.tsv")
        test_path = os.path.join(data_dir, "dev.tsv")
        dev_path = os.path.join(data_dir, "test.tsv")
        train_data = pd.read_csv(train_path, sep='\t', header=None, names=['p1', 'label', 'p2', 'sentence'])
        test_data = pd.read_csv(test_path, sep='\t', header=None, names=['p1', 'label', 'p2', 'sentence'])
        dev_data = pd.read_csv(dev_path, sep='\t', header=None, names=['p1', 'label', 'p2', 'sentence'])
    elif dataset_name == "MRPC":
        split_mrpc_data(data_dir)
        
