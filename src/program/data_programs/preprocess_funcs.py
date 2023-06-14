import torch
from transformers import default_data_collator
from torch.utils.data import Dataset, DataLoader

class PreprocessData:
    def __init__(self, tokenizer, dataset_name, dataset, batch_size):  # foranli タスクモードを指定して前処理の場合分けする
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        if self.dataset_name == "sst2":
            self.max_length = 64
        elif self.dataset_name == "anli":
            self.max_length = 100
        self.dataset = dataset
        self.batch_size = batch_size
    
    def verbalizer(self, label):
        """
        バーバライザー
        """
        if self.dataset_name == "gpt2mix/sst2":
            if label == 0:
                return "positive"
            elif label == 1:
                return "negative"
        elif self.dataset_name == "sst2":
            if label == 0:
                return "negative"
            elif label == 1:
                return "positive"
        elif self.dataset_name == "anli":
            if label == 0:
                return "entailment"
            elif label == 1:
                return "neutral"
            elif label == 2:
                return "contradiction"
    
    def preprocess_function(self, examples):
        """
        データの前処理
        """
        # if self.task_mode == sst2:     # foranli タスクがsst2ならこの処理
        batch_size = len(examples['sentence'])
        inputs = [f"{x} It is " for x in examples['sentence']]
        targets = [self.verbalizer(x) for x in examples['label']]
        model_inputs = self.tokenizer(inputs)
        labels = self.tokenizer(targets)
        for i in range(batch_size):  # 各入力文に対してのトークナイズしたIDとそれに対応したラベルのIDを用意する
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i] + [self.tokenizer.pad_token_id]
            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]  # i文目の入力文のID
            label_input_ids = labels["input_ids"][i]  # i文目のラベルの部分のID(ラベルのID以外の部分は-100に設定)
            model_inputs["input_ids"][i] = [self.tokenizer.pad_token_id] * (
                self.max_length - len(sample_input_ids)
            ) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (self.max_length - len(sample_input_ids)) + model_inputs[
                "attention_mask"
            ][i]
            labels["input_ids"][i] = [-100] * (self.max_length - len(sample_input_ids)) + label_input_ids
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:self.max_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:self.max_length])
            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:self.max_length])
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs
    
    def preprocess_function_for_anli(self, examples):
        """
        ANLI 用のデータの前処理
        """
        batch_size = len(examples['premise'])
        inputs = [f"premise: {premise}\nhypothesis: {hypothesis}\nprediction: " for premise, hypothesis in zip(examples['premise'], examples['hypothesis'])]
        targets = [self.verbalizer(x) for x in examples['label']]
        model_inputs = self.tokenizer(inputs)
        labels = self.tokenizer(targets)
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i] + [self.tokenizer.pad_token_id]
            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i]
            model_inputs["input_ids"][i] = [self.tokenizer.pad_token_id] * (
                self.max_length - len(sample_input_ids)
            ) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (self.max_length - len(sample_input_ids)) + model_inputs[
                "attention_mask"
            ][i]
            labels["input_ids"][i] = [-100] * (self.max_length - len(sample_input_ids)) + label_input_ids
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:self.max_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:self.max_length])
            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:self.max_length])
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs
    
    def get_preprocess_datasets(self):
        if self.dataset_name == "sst2":
            processed_datasets = self.dataset.map(
                self.preprocess_function,
                batched=True,
                num_proc=1,
                remove_columns=self.dataset['train'].column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset",
            )
        elif self.dataset_name == "anli":
            processed_datasets = self.dataset.map(
                self.preprocess_function_for_anli,
                batched=True,
                num_proc=1,
                remove_columns=self.dataset['train'].column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset",
            )
        return processed_datasets
    
    def get_preprocess_datasets_train_eval(self):
        processed_datasets = self.get_preprocess_datasets()
        return processed_datasets['train'], processed_datasets['validation']
    
    def get_preprocess_dataloader_train_eval(self):
        train_dataset, eval_dataset = self.get_preprocess_datasets_train_eval()
        train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=self.batch_size, pin_memory=True)
        eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=self.batch_size, pin_memory=True)
        return train_dataloader, eval_dataloader