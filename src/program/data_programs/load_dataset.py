from datasets import load_dataset
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import random


class GetDataset:
    def __init__(self, dataset_name, seed):
        self.dataset_name = dataset_name
        self.seed = seed

    ######## seed 固定 ##########
    
    def fix_seed(self):
        """
        seed固定
        """
        # random
        random.seed(self.seed)
        # Numpy
        np.random.seed(self.seed)
        # Pytorch
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
    
    ######## データのロード #########

    def _load_dataset(self):
        """
        データのロード
        """
        dataset = load_dataset(self.dataset_name)
        return dataset
    
    def dataset_cleaner(self, dataset):
        """
        フォーマットを合わせる
        """
        del dataset['test']
        dataset = dataset.remove_columns('idx')
        return dataset
    
    def devide_train_for_dev(self, dataset):
        """
        trainを分割
        """
        self.fix_seed()
        dataset_new = dataset['train'].train_test_split(test_size=0.1, shuffle=True)
        dataset_new["validation"] = dataset_new['test']
        del dataset_new['test']
        dataset_new['test'] = dataset['validation']
        return dataset_new
    
    def load_clean_dataset(self):
        """
        main関数
        """
        dataset = self._load_dataset()
        if self.dataset_name == "sst2":
            dataset = self.devide_train_for_dev(self.dataset_cleaner(dataset))
        if self.dataset_name == "anli":
            dataset = self.clean_dataset_for_anli(dataset)
        return dataset
    
    def clean_dataset_for_anli(self, dataset):
        """
        anli のデータセットを整形
        """
        dataset['train'] = dataset['train_r2'].remove_columns('uid').remove_columns('reason')
        dataset['validation'] = dataset['dev_r2'].remove_columns('uid').remove_columns('reason')
        dataset['test'] = dataset['test_r2'].remove_columns('uid').remove_columns('reason')

        del dataset['train_r1']
        del dataset['train_r2']
        del dataset['train_r3']

        del dataset['dev_r1']
        del dataset['dev_r2']
        del dataset['dev_r3']

        del dataset['test_r1']
        del dataset['test_r2']
        del dataset['test_r3']

        return dataset
    
    
    def class_mapped_dataset(self, dataset, classes):
        dataset = dataset.map(
            lambda x: {"class label": [classes[label] for label in x['label']]},
            batched=True,
            num_proc=1,
        )
        return dataset
    
    def get_dataset(self):
        dataset = self.load_clean_dataset()
        if self.dataset_name == "sst2":
            classes = self.get_classes()
            dataset = self.class_mapped_dataset(dataset, classes)
        return dataset

    ####### ラベルの取得 #########
    def get_classes(self):
        dataset = self.load_clean_dataset()
        classes = [k.replace("_", " ") for k in dataset['train'].features['label'].names]
        return classes
    