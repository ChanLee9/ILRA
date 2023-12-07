import os
import pandas as pd

import torch
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
  
def read_data(config: object) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    read data and transform it to pandas.DataFrame for Dataset and DataLoader
    """
    dataset_name = config.dataset_name
    data_dir = os.path.join("datasets", config.dataset_name)
    train_path = os.path.join(data_dir, "train.tsv")
    test_path = os.path.join(data_dir, "test.tsv")
    dev_path = os.path.join(data_dir, "dev.tsv")
    
    if dataset_name == "CoLA":
        # The Corpus of Linguistic Acceptability, single sentence classification, 
        # for train and dev set, we only need sentence and label, for test set, we need sentence, and id
        train_data = pd.read_csv(train_path, sep='\t', header=None, names=['p1', 'label', 'p2', 'sentence1'])
        test_data = pd.read_csv(test_path, sep='\t', header=0, names=['index', 'sentence1'])
        dev_data = pd.read_csv(dev_path, sep='\t', header=None, names=['p1', 'label', 'p2', 'sentence1'])
        return (train_data[['sentence1', 'label']], dev_data[['sentence1', 'label']], test_data)
        
    elif dataset_name == "RTE":
        # The Recognizing Textual Entailment datasets, two sentences classification
        # note that the label is str in ['entailment', 'not_entailment'], so we need to convert it to int
        train_data = pd.read_csv(train_path, sep='\t', header=0, names=['id', 'sentence1', 'sentence2', 'label'])
        test_data = pd.read_csv(test_path, sep='\t', header=0, names=['index', 'sentence1', 'sentence2'])
        dev_data = pd.read_csv(dev_path, sep='\t', header=0, names=['id', 'sentence1', 'sentence2', 'label'])
        
        # convert label to int
        train_data['label'] = train_data['label'].apply(lambda x: 1 if x == 'entailment' else 0)
        dev_data['label'] = dev_data['label'].apply(lambda x: 1 if x == 'entailment' else 0)
        return (train_data[['sentence1', 'sentence2', 'label']], dev_data[['sentence1', 'sentence2', 'label']], test_data)
        
    elif dataset_name == "MRPC":
        # The Microsoft Research Paraphrase Corpus, two sentences classification
        # The MRPC dataset has labels, so we can evaluate the model offline
        train_data = pd.read_csv(train_path, sep='\t', header=0, names=["label", "id1", "id2", "sentence1", "sentence2"])
        test_data = pd.read_csv(test_path, sep='\t', header=0, names=["label", "id1", "id2", "sentence1", "sentence2"])
        dev_data = pd.read_csv(dev_path, sep='\t', header=0, names=["label", "id1", "id2", "sentence1", "sentence2"])
        return (train_data[['sentence1', 'sentence2', 'label']], dev_data[['sentence1', 'sentence2', 'label']], \
            test_data[['sentence1', 'sentence2', 'label']])


class MyDataset(Dataset):
    def __init__(self, config, data) -> None:
        super().__init__()
        self.dataset_name = config.dataset_name
        self.data = data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index):
        cur_series = self.data.iloc[index]
        res = {
            "sentence1": cur_series['sentence1'],
        }
        if "label" in cur_series.index:
            res["label"] = cur_series["label"]
        
        if "sentence2" in cur_series.index:
            res["sentence2"] = cur_series["sentence2"]
        
        if "index" in cur_series.index:
            res["index"] = cur_series["index"]
        return res


class MyDataLoader(DataLoader):
    def __init__(self, dataset, config, shuffle=False) -> None:
        super().__init__(dataset, batch_size=config.batch_size, shuffle=shuffle, collate_fn=self.collate_fn)
        self.max_length = config.max_length
        self.device = config.device
    
    def collate_fn(self, batch):
        res = {}
        for key in batch[0].keys():
            res[key] = [item[key] for item in batch]
        
        return res


if __name__ == "__main__":
    @dataclass
    class Config:
        dataset_name = "RTE"
        batch_size = 4
        max_length = 256
        model_name_or_path = "../pretrained_models/roberta"
        device = "cpu"
        
    config = Config()
    
    # test for read data
    config.dataset_name = "MRPC"
    train, dev, test = read_data(config)
    
    # test for MyDataset
    train, dev, test = read_data(config)
    train_dataset = MyDataset(config, train)
    test_dataset = MyDataset(config, test)
    dev_dataset = MyDataset(config, dev)
    
    train_dataloader = MyDataLoader(train_dataset, config, shuffle=True)
    test_dataloader = MyDataLoader(test_dataset, config, shuffle=False)
    dev_dataloader = MyDataLoader(dev_dataset, config, shuffle=False)
    breakpoint()
