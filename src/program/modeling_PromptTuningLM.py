import json
import logging
import os
import random
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
from datasets import load_dataset

class PromptTuningLM(nn.Module):
    def __init__(
        self,
        model_name: str,
        n_prompt_tokens: int,
        config: AutoConfig,
        soft_prompt_path: str = None,
    ):
        super(PromptTuningLM, self).__init__()
        self.model_name = model_name
        self.n_prompt_tokens = n_prompt_tokens 
        # 事前学習済みのGPTの呼び出し
        self.lm = AutoModelForCausalLM.from_pretrained(model_name, config=config)  # 今回は japanese-gpt2-medium

        # Promptに対する埋め込みベクトルの作成
        self.soft_prompt = nn.Embedding(n_prompt_tokens, config.hidden_size)
        torch.nn.init.xavier_uniform_(self.soft_prompt.weight)  # soft prompt を初期化

        # GPTの重みを固定
        for param in self.lm.parameters():
            param.requires_grad = False

        # [推論時] Promptに対する学習済みの埋め込みベクトルをロード ???
        if soft_prompt_path is not None: 
            print(f"Set soft prompt. ({n_prompt_tokens} tokens)")
            self.soft_prompt = torch.load(soft_prompt_path)

    def _extend_inputs(self, input_ids) -> torch.Tensor:
        """
        Promptに対する埋め込みベクトルを付与する
        """
        # input_idsをベクトルに変換する（事前学習モデルが異なる場合は変更する必要あり）
        if "gpt" in self.model_name:
            inputs_embeds = self.lm.transformer.wte(input_ids)
        elif "opt" in self.model_name:
            inputs_embeds = self.lm.model.decoder.embed_tokens(input_ids)
        # inputs_embeds = self.lm.model.decoder.embed_tokens(input_ids)
        if len(list(inputs_embeds.shape)) == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)
        # Promptに対する埋め込みベクトルとinputs_embedsを連結する
        batch_size = inputs_embeds.size(0)
        learned_embeds = self.soft_prompt.weight.repeat(batch_size, 1, 1)
        extended_embeds = torch.cat([learned_embeds, inputs_embeds], dim=1)  # prompt + input
        return extended_embeds

    def _extend_labels(self, labels, ignore_index=-100) -> torch.Tensor:
        """
        inputに合わせて正解ラベルにPromptに対するラベルを付与する
        """
        if len(list(labels.shape)) == 1:
            labels = labels.unsqueeze(0)
        n_batches = labels.shape[0]
        # Promptに対してignore_indexを付与（-100に設定していれば損失が計算されない）
        prompt_labels = torch.full((n_batches, self.n_prompt_tokens), 
                                    ignore_index).to(labels.device)  # -100 で作られた (n_batches, self.n_prompt_tokens) を生成　ラベルの
        # Promptに対するラベルと元の正解ラベルを連結する
        extended_labels = torch.cat([prompt_labels, labels], dim=1)  # prompt ver. のラベル
        return extended_labels

    def save_soft_prompt(self, path: str, filename: str, logger):
        """
        Promptに対する埋め込みベクトルの保存
        """
        torch.save(self.soft_prompt, os.path.join(path, filename))  # soft prompt を保存
        logger.info(f"Saved soft prompt: {os.path.join(path, filename)}")

    def forward(self, input_ids, labels=None, return_dict=None):
        # Promptを付与したベクトル
        inputs_embeds = self._extend_inputs(input_ids)  # input に prompt を付与
        if labels is not None:
            labels = self._extend_labels(labels)

        return self.lm(
            inputs_embeds=inputs_embeds,
            labels=labels,
            return_dict=return_dict,
        )  # 事前学習済みモデルを実行
    
    def forward_hard(self, input_ids, labels=None, return_dict=None):
        # ここを hard に変える
        # inputs_embeds = self._extend_inputs(input_ids)
        if "gpt" in self.model_name:
            inputs_embeds = self.lm.transformer.wte(input_ids)
        elif "opt" in self.model_name:
            inputs_embeds = self.lm.model.decoder.embed_tokens(input_ids)
        if len(list(inputs_embeds.shape)) == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)
        
        if labels is not None:
            labels = self._extend_labels(labels)
        
        return self.lm(
            inputs_embeds=inputs_embeds,
            labels=labels,
            return_dict=return_dict,
        )


    def generate(self, input_text, tokenizer, max_new_tokens, eos_token_id, device):
        """
        [推論時]自己回帰で回答を生成する
        """
        input_ids = tokenizer.encode(input_text, add_special_tokens=False)
        cur_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
        # 最大でmax_new_tokensだけ単語を生成する
        for _ in range(max_new_tokens):
            outputs = self.forward(cur_ids)
            softmax_logits = torch.softmax(outputs.logits[0,-1], dim=0)
            result = self.check_posinega(softmax_logits)
        return result

    def generate_hard(self, input_text, tokenizer, max_new_tokens, eos_token_id, device, harded_prompt):
        """
        [推論時]自己回帰で回答を生成する,hard
        """
        input_text = harded_prompt + " " + input_text
        # print(input_text)
        input_ids = tokenizer.encode(input_text, add_special_tokens=False)
        cur_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
        for _ in range(max_new_tokens):
            outputs = self.forward_hard(cur_ids)
            softmax_logits = torch.softmax(outputs.logits[0,-1], dim=0)
            # next_token_id = int(softmax_logits.argmax().to('cpu'))
            # print(next_token_id)
            result = self.check_posinega(softmax_logits)
        return result
            
    def check_posinega(self, logits):
        if "gpt" in self.model_name:
            if logits[3967] >= logits[4633]:
                result = "positive"
            else:
                result = "negative"
        elif "opt" in self.model_name:
            if logits[1313] >= logits[2430]:
                result = "positive"
            else:
                result = "negative"
        return result


@dataclass
class InputExample():
    text: str
    label: str

def create_examples(dataset, dataset_name):
    examples = []
    if dataset_name == "gpt3mix/sst2":
        for sent, lab in zip(dataset['text'], dataset['label']):
            if lab == 0:
                posinega = "positive"
            elif lab == 1:
                posinega = "negative"
            examples.append(InputExample(
                text = sent,
                label = posinega))
            
    elif dataset_name == "sst2":
        for sent, lab in zip(dataset['sentence'], dataset['label']):
            if lab == 0:
                posinega = "negative"
            elif lab == 1:
                posinega = "positive"
            examples.append(InputExample(
                text = sent,
                label = posinega))
    
    return examples

class CustomDataset(torch.utils.data.IterableDataset):
    def __init__(self, tokenizer, generator):
        super().__init__()
        self._tokenizer = tokenizer
        self._generator = generator

    @classmethod
    def from_texts(cls, tokenizer, texts):
        return cls(tokenizer=tokenizer, generator=lambda: texts)

    def __iter__(self):
        for text in self._generator():
            ids = self._tokenizer.encode(text)
            yield {"input_ids": ids, "labels": ids}