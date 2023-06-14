import argparse
import random
from typing import Dict, List

import numpy as np
import torch
from data.preprocess import load_create_prompt_function, tokenize_function, tokenize_function_ft
from datasets import load_dataset
from models.load_model import load_model, load_peft_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, precision_score
from soft_prompt_trainer import PEFTTrainer, SoftPromptTrainer
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    DataCollatorForLanguageModeling,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    default_data_collator,
    get_linear_schedule_with_warmup,
)
from peft.src.peft import (
    PeftConfig,
    PeftModel,
    PromptTuningConfig,
    PromptTuningInit,
    TaskType,
    get_peft_model,
    prepare_model_for_int8_training,
)
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import math

class Measure_LM:
    def __init__(self, sim_index, seed, num_virtual_tokens, device):
        self.sim_index = sim_index
        self.seed = seed
        self.num_virtual_tokens = num_virtual_tokens
        self.device = device

    
    def sim_in_opt(self, model, size):
        """
        OPTの中のコサイン類似度
        """
        heatmap_path = f'/home/kyotaro/peft_clean/heatmap/opt_similarity_{size}_{self.seed}_{self.sim_index}.png'
        title = f"OPT vocabulary (random {size}samples)"
        similarity_list = np.zeros((size, size))
        wte = model.base_model.model.decoder.embed_tokens.weight

        wte_vocab = wte.shape[0]
        wte_sample_id = random.sample(list(range(wte_vocab)), size)
        for i in tqdm(range(len(wte_sample_id))):
            for j in range(len(wte_sample_id)):
                if self.sim_index == "cosine":
                    similarity = self.cos_sim_measure_internal(wte[wte_sample_id[i]], wte[wte_sample_id[j]])
                elif self.sim_index == "eucrid":
                    similarity = self.eucrid_measure_internal(wte[wte_sample_id[i]], wte[wte_sample_id[j]])
                similarity_list[i][j] = round(similarity.item(), 3)
        self.make_similality_heatmap(similarity_list, heatmap_path, title)
        return similarity_list

    def sim_in_gpt2(self, size):
        """
        GPT2の中のコサイン類似度
        """
        heatmap_path = f'/home/kyotaro/peft_test/heatmap/gpt2_similarity_{size}_{self.seed}_{self.sim_index}.png'
        title = f"GPT2 vocabulary (random {size}samples)"
        # similarity_list = [[0 for a in range(size)] for b in range(size)]
        similarity_list = np.zeros((size, size))
        model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
        wte = model.transformer.wte.weight

        wte_vocab = wte.shape[0]
        wte_sample_id = random.sample(list(range(wte_vocab)), size)
        for i in tqdm(range(len(wte_sample_id))):
            for j in range(len(wte_sample_id)):
                if self.sim_index == "cosine":
                    similarity = self.cos_sim_measure_internal(wte[wte_sample_id[i]], wte[wte_sample_id[j]])
                elif self.sim_index == "eucrid":
                    similarity = self.eucrid_measure_internal(wte[wte_sample_id[i]], wte[wte_sample_id[j]])
                similarity_list[i][j] = round(similarity.item(), 3)
        self.make_similality_heatmap(similarity_list, heatmap_path, title)
        return similarity_list
        
    def sim_in_opt_all(self, model):
        """
        OPTの中のコサイン類似度（全語彙）
        """
        heatmap_path = f'/home/kyotaro/peft_test/heatmap/opt_similarity_all_{self.sim_index}_{self.seed}.png'
        title = f'OPT vocabulary ({self.sim_index})'
        vocab_size = model.base_model.config.vocab_size // 100
        similarity_list = np.zeros((vocab_size, vocab_size))
        # similarity_list = [[0 for a in range(vocab_size)] for b in tqdm(range(vocab_size))]
        wte = model.base_model.model.decoder.embed_tokens.weight

        # wte_vocab = wte.shape[0]
        # wte_sample_id = random.sample(list(range(wte_vocab)), vocab_size)
        for i in tqdm(range(vocab_size)):
            for j in range(vocab_size):
                if self.sim_index == "cosine":
                    similarity = self.cos_sim_measure_internal(wte[i], wte[j])
                elif self.sim_index == "eucrid":
                    similarity = self.eucrid_measure_internal(wte[i], wte[j])
                similarity_list[i][j] = round(similarity.item(), 3)
        self.make_similality_heatmap(similarity_list, heatmap_path, title)
        return similarity_list
    
    def sim_in_soft_prompt(self, model, peft_model_id):
        """
        ソフトプロンプトの中のコサイン類似度
        """
        peft_model_id = peft_model_id.replace("/", "_")
        heatmap_path = f'/home/kyotaro/peft_test/heatmap/soft_prompt_similarity_{self.sim_index}_{peft_model_id}.png'
        title = f'OPT softprompt ({self.sim_index})'
        similarity_list = [[0 for a in range(10)] for b in range(10)]
        soft_prompt = model.prompt_encoder.embedding.weight
        for i in range(self.num_virtual_tokens):
            for j in range(self.num_virtual_tokens):
                if self.sim_index == "cosine":
                    similarity = self.cos_sim_measure_internal(soft_prompt[i], soft_prompt[j])
                elif self.sim_index == "eucrid":
                    similarity = self.eucrid_measure_internal(soft_prompt[i], soft_prompt[j])
                similarity_list[i][j] = round(similarity.item(), 2)
        self.make_similality_heatmap(similarity_list, heatmap_path, title)
        return similarity_list

    def cos_sim_measure_internal(self, vector1, vector2):
        """
        中の類似度測る用
        """
        vector1 = vector1.to(self.device)
        vector2 = vector2.to(self.device)
        vector1 = vector1.unsqueeze(0) # [1, 768]
        vector2 = vector2.unsqueeze(0) # [1, 768]
        dot = vector1 @ vector2.T
        vector_norm = (vector1 * vector1).sum(axis=1, keepdims=True) ** .5
        matrix_norm = (vector2 * vector2).sum(axis=1, keepdims=True) ** .5
        cos_sim = dot / vector_norm / matrix_norm.T
        return cos_sim
    
    def eucrid_measure_internal(self, vector1, vector2):
        """
        ユークリッド距離の確認
        """
        vector1 = vector1.to(self.device)
        vector2 = vector2.to(self.device)
        vector1 = vector1.unsqueeze(0)
        vector2 = vector2.unsqueeze(0)
        L2_dist = torch.dist(vector1,vector2,p=2)
        return L2_dist
    
    def make_similality_heatmap(self, sim_list, heatmap_path, title):
        """
        類似度をヒートマップにする
        """
        plt.figure()
        sns.heatmap(sim_list, vmin=0, vmax=20.0, annot=True)  
        plt.title(title)
        plt.savefig(heatmap_path)
        plt.close('all')