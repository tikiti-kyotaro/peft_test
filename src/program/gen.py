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


parser = argparse.ArgumentParser(description='soft_promptの訓練と推論')

parser.add_argument('--seed', type=int, default=0, help='')
parser.add_argument('--train_file', type=str, default='data/SST-2/train.tsv', help='学習ファイル')
parser.add_argument('--eval_file', type=str, default='data/SST-2/dev.tsv', help='予測元のファイル')
parser.add_argument('--eval_out_file', type=str, default='out_test.txt', help='予測先のファイル')
parser.add_argument('--eval_out_file_hard', type=str, default='out_test_hard.txt', help='hard prompt の方の出力先')
parser.add_argument('--soft_prompt_file', type=str, default='/home/kyotaro/peft_test/non_peft_model/gpt2-medium/SST-2_100.pt', help='作った soft prompt のファイル')
parser.add_argument('--harded_soft_prompt', type=str, default='/home/kyotaro/peft_test/non_peft_model/gpt2-medium/SST-2_100.pt', help='自然言語にする soft prompt のパス')
parser.add_argument('--contexts_file', type=str, default=None, help='context のファイル')
parser.add_argument('--contexts_mode', type=bool, default=False, help='事例を入れる')
parser.add_argument('--output_dir', type=str, default='non_peft_model', help='出力するディレクトリ')
parser.add_argument('--num_train_epochs', type=int, default=100, help='学習のエポック数')
parser.add_argument('--patience', type=int, default=None, help='early stopping')
parser.add_argument('--n_prompt_tokens', type=int, default=10, help='prompt のトークン数')
parser.add_argument('--train_batch_size', type=int, default=8, help='')
parser.add_argument('--learning_rate', type=float, default=5e-5, help='学習率')
parser.add_argument('--max_new_tokens',type=int, default=1, help='')
parser.add_argument('--model_name', type=str, default='facebook/opt-125M', help='事前学習済みモデル')
parser.add_argument('--gen_model_name', type=str, default='facebook/opt-125M', help='生成するのに用いるモデル、サイズを変える？')
parser.add_argument('--mode', type=str, default='SST-2', help='タスク指定')

args = parser.parse_args()

def fix_seed():
    """
    seed固定
    """
    # random
    random.seed(args.seed)
    # Numpy
    np.random.seed(args.seed)
    # Pytorch
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

class PromptTuningLM(nn.Module):
    def __init__(
        self,
        model_name: str,
        n_prompt_tokens: int,
        config: AutoConfig,
        soft_prompt_path: str = None,
    ):
        super(PromptTuningLM, self).__init__()
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
        if "gpt" in args.model_name:
            inputs_embeds = self.lm.transformer.wte(input_ids)
        elif "opt" in args.model_name:
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
        if "gpt" in args.model_name:
            inputs_embeds = self.lm.transformer.wte(input_ids)
        elif "opt" in args.model_name:
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
        if args.mode == "quiz":
            for _ in range(max_new_tokens):
                outputs = self.lm(cur_ids)
                softmax_logits = torch.softmax(outputs.logits[0,-1], dim=0)
                # 最大確率の単語を次の単語として一意に決定
                next_token_id = int(softmax_logits.argmax().to('cpu'))
                #print(next_token_id)
                # もし選択された単語がeos_tokenなら生成を終了する
                if next_token_id == eos_token_id:
                    break
                # 選択された単語をcur_idsに追加して次の処理を行う
                next_token_id = torch.tensor([[next_token_id]]).to(device)
                cur_ids = torch.cat([cur_ids, next_token_id], dim=1)

            # print(cur_ids)
            # 生成した単語ID列をテキストに変換する
            output_list = list(cur_ids.squeeze().to('cpu').numpy())
            output_text = tokenizer.decode(output_list)
            return output_text

        elif args.mode == "SST-2":
            for _ in range(max_new_tokens):
                outputs = self.forward(cur_ids)
                softmax_logits = torch.softmax(outputs.logits[0,-1], dim=0)
                # next_token_id = int(softmax_logits.argmax().to('cpu'))
                # print(next_token_id)
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
        if args.mode == "SST-2":
            for _ in range(max_new_tokens):
                outputs = self.forward_hard(cur_ids)
                softmax_logits = torch.softmax(outputs.logits[0,-1], dim=0)
                # next_token_id = int(softmax_logits.argmax().to('cpu'))
                # print(next_token_id)
                result = self.check_posinega(softmax_logits)
            return result
            
    def check_posinega(self, logits):
        if "gpt" in args.model_name:
            if logits[3967] >= logits[4633]:
                result = "positive"
            else:
                result = "negative"
        elif "opt" in args.model_name:
            if logits[1313] >= logits[2430]:
                result = "positive"
            else:
                result = "negative"
        return result
    
def dataload(dataset_name):
    dataset = load_dataset(dataset_name)
    return dataset


@dataclass
class InputExample():
    if args.mode == "quiz":
        question: str
        answer: str
    elif args.mode == "SST-2":
        text: str
        label: str

def create_examples(dataset, dataset_name):
    examples = []
    if args.mode == "SST-2":
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
    
    # print(examples)
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

def collate_fn(samples):
    batch = {'input_ids': [], 'labels': []}
    for sample in samples:
        batch['input_ids'].append(torch.tensor(sample['input_ids']))
        batch['labels'].append(torch.tensor(sample['labels']))
    batch['input_ids'] = torch.nn.utils.rnn.pad_sequence(
                    batch['input_ids'], batch_first=True, padding_value=3)  # padding
    batch['labels'] = torch.nn.utils.rnn.pad_sequence(
                    batch['labels'], batch_first=True, padding_value=3)  # padding
    return batch

def train_prompt(model, tokenizer, device, optimizer, dataset, logger, dataset_name):
    logger.info("***** Running training *****")
    train_examples = create_examples(dataset["train"], dataset_name)  # 質問と答え

    if args.mode == "SST-2":
        train_texts = [example.text + " It is " + example.label
                        for example in train_examples]
    train_data = CustomDataset.from_texts(tokenizer, texts=train_texts)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_data,
                        batch_size=args.train_batch_size, collate_fn=collate_fn)

    model.train()
    if args.patience is None:
        for epoch in range(int(args.num_train_epochs)):
            logger.info(f'Epoch: {epoch+1}')
            total_loss = 0
            for batch in tqdm(train_dataloader, desc="Iteration"):
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            logger.info(f'loss: {loss}')


            # Promptに対する埋め込みベクトルのみ保存する
            model_to_save = model.module if hasattr(model, 'module') else model
            soft_path = f"/home/kyotaro/peft_test/{args.output_dir}/{args.model_name}/SST-2_{epoch+1}.pt"
            model_to_save.save_soft_prompt(args.output_dir, soft_path, logger)


# def train_prompt(model, tokenizer, device, optimizer, dataset, logger):
#     logger.info("***** Running training *****")
#     train_examples = create_examples(dataset["train"])  # 質問と答え
#     if args.mode == "SST-2":
#         train_texts = [example.text + " It is " + example.label
#                         for example in train_examples]
#     train_data = CustomDataset.from_texts(tokenizer, texts=train_texts)
#     train_dataloader = torch.utils.data.DataLoader(dataset=train_data,
#                         batch_size=args.train_batch_size, collate_fn=collate_fn)

#     model.train()
#     if args.patience is None:
#         for epoch in range(int(args.num_train_epochs)):
#             logger.info(f'Epoch: {epoch+1}')
#             for batch in tqdm(train_dataloader, desc="Iteration"):
#                 input_ids = batch['input_ids'].to(device)
#                 labels = batch['labels'].to(device)
#                 outputs = model(input_ids, labels=labels)
#                 loss = outputs.loss
#                 loss.backward()
#                 optimizer.step()
#                 optimizer.zero_grad()
#             logger.info(f'loss: {loss}')

#             # Promptに対する埋め込みベクトルのみ保存する
#             model_to_save = model.module if hasattr(model, 'module') else model
#             soft_path = f"/home/kyotaro/peft_test/{args.output_dir}/{args.model_name}/SST-2_{epoch+1}.pt"
#             model_to_save.save_soft_prompt(args.output_dir, soft_path, logger)

def eval(tokenizer, device, dataset, logger, dataset_name):
    logger.info("***** Running evaluation *****")
    tokenizer = AutoTokenizer.from_pretrained(args.gen_model_name)
    config = AutoConfig.from_pretrained(args.gen_model_name)
    model = PromptTuningLM(
        args.model_name,
        n_prompt_tokens=args.n_prompt_tokens,
        soft_prompt_path=os.path.join(args.output_dir, args.soft_prompt_file),
        config=config,
    )

    test_examples = create_examples(dataset["test"], dataset_name)

    correct = 0

    model.to(device)
    model.eval()
    output_texts = []   

    if args.mode == "SST-2":
        result_list = []
        with open(args.eval_out_file, "w") as out_:
            for example in tqdm(test_examples):
                input_text = example.text + " It is"
                ans = example.label
                # print(input_text)
                # print(ans)
                result = model.generate(input_text, tokenizer,
                                args.max_new_tokens, tokenizer.eos_token_id, device)
                if result == ans:
                    correct += 1
                out_.write(f'{result}\n')
            print(correct / len(test_examples))


def hard_eval(harded_prompt, tokenizer, device, dataset, logger, dataset_name):
    logger.info("***** Running evaluation *****")
    tokenizer = AutoTokenizer.from_pretrained(args.gen_model_name)
    config = AutoConfig.from_pretrained(args.gen_model_name)
    model = PromptTuningLM(
        args.gen_model_name,
        n_prompt_tokens=args.n_prompt_tokens,
        soft_prompt_path=os.path.join(args.output_dir, args.soft_prompt_file),
        config=config,
    )
    # if args.mode == "SST-2":
    #     set_special_token(tokenizer, model)

    model.to(device)
    model.eval()
    output_texts = []

    test_examples = create_examples(dataset["validation"], dataset_name)

    correct = 0
    if args.mode == "SST-2":
        result_list = []
        with open(args.eval_out_file_hard, "w") as out_:
            for example in tqdm(test_examples):
                input_text = example.text + " It is"
                ans = example.label
                # print(input_text)
                result = model.generate_hard(input_text, tokenizer,
                                args.max_new_tokens, tokenizer.eos_token_id, device, harded_prompt)
                if ans == result:
                    correct += 1
                out_.write(f'{result}\n')
        print(correct / len(test_examples))

def cos_sim_measure(vector, matrix):
    dot = vector @ matrix.T
    vector_norm = (vector * vector).sum(axis=1, keepdims=True) ** .5
    matrix_norm = (matrix * matrix).sum(axis=1, keepdims=True) ** .5
    cos_sim = dot / vector_norm / matrix_norm.T
    return cos_sim


def soft_to_hard():
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, config=config)
    soft_prompt = torch.nn.Embedding(args.n_prompt_tokens, config.hidden_size)
    soft_prompt = torch.load(args.harded_soft_prompt, map_location='cpu')  # token * dim
    # print(soft_prompt_sh.shape)
    # 既存の単語に対するベクトルを抽出
    wte = model.transformer.wte.weight  # vocab * dim
    # wte = torch.transpose(wte, 0, 1)

    token_ids = []
    for i in range(args.n_prompt_tokens):
        vector = soft_prompt.weight[i].unsqueeze(0)
        # print(f'vector : {vector.shape}')
        # ボキャブラリーから最も類似するベクトルを持つ単語を選択（内積を類似度とする）
        similarity = cos_sim_measure(vector, wte)
        # print(f'similality : {similarity.shape}')
        token_id = int(similarity.argmax())
        print(f'token_id : {token_id}, similarity : {similarity[0][token_id]}')
        token_ids.append(token_id)

    prompt = tokenizer.decode(token_ids)
    print(prompt)
    return prompt

def gen_put_together(tokenizer, device, dataset, logger):
    for i in range(args.num_train_epochs):
        soft_prompt_file = f'/home/kyotaro/peft_test/{args.output_dir}/{args.model_name}/SST-2_{i+1}.pt'
        tokenizer = AutoTokenizer.from_pretrained(args.gen_model_name)
        config = AutoConfig.from_pretrained(args.gen_model_name)
        model = PromptTuningLM(
            args.gen_model_name,
            n_prompt_tokens=args.n_prompt_tokens,
            soft_prompt_path=os.path.join(args.output_dir, soft_prompt_file),
            config=config,
        )
        model.to(device)
        model.eval()
        output_texts = []

        test_examples = create_examples(dataset["test"])

        correct = 0
        if args.mode == "SST-2":
            result_list = []
            with open(args.eval_out_file_hard, "w") as out_, open("/home/kyotaro/peft_test/results/gpt2-medium_harded_non_peft_.txt", "w") as accuracy:
                for example in tqdm(test_examples):
                    input_text = example.text + " It is"
                    ans = example.label
                    # print(input_text)
                    # print(ans)
                    result = model.generate(input_text, tokenizer,
                                    args.max_new_tokens, tokenizer.eos_token_id, device)
                    if result == ans:
                        correct += 1
                    out_.write(f'{result}\n')
                print(correct / len(test_examples))
                accuracy.write(f'{correct / len(test_examples)}\n')


def gen_put_together_harded(tokenizer, device, dataset, logger):
    for i in range(args.num_train_epochs):
        soft_prompt_file = f'/home/kyotaro/peft_test/{args.output_dir}/{args.model_name}/SST-2_{i+1}.pt'
        args.harded_soft_prompt = soft_prompt_file

        tokenizer = AutoTokenizer.from_pretrained(args.gen_model_name)
        config = AutoConfig.from_pretrained(args.gen_model_name)
        model = PromptTuningLM(
            args.gen_model_name,
            n_prompt_tokens=args.n_prompt_tokens,
            soft_prompt_path=os.path.join(args.output_dir, args.soft_prompt_file),
            config=config,
        )
        harded_prompt = soft_to_hard()
        model.to(device)
        model.eval()
        output_texts = []

        test_examples = create_examples(dataset["test"])

        correct = 0
        if args.mode == "SST-2":
            result_list = []
            with open(args.eval_out_file_hard, "w") as out_, open("/home/kyotaro/peft_test/results/gpt2-medium_harded_non_peft_harded.txt", "w") as accuracy:
                for example in tqdm(test_examples):
                    input_text = example.text + " It is"
                    ans = example.label
                    # print(input_text)
                    result = model.generate_hard(input_text, tokenizer,
                                    args.max_new_tokens, tokenizer.eos_token_id, device, harded_prompt)
                    if ans == result:
                        correct += 1
                    out_.write(f'{result}\n')
            print(correct / len(test_examples))
            accuracy.write(f'{correct / len(test_examples)}\n')


def main():
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger(__name__)

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(args.model_name)

    dataset_name = "sst2"
    # dataset = dataload("gpt3mix/sst2")
    dataset = dataload(dataset_name)
    if dataset_name == "sst2":
        del dataset['test']
        dataset = dataset.remove_columns('idx')

    model = PromptTuningLM(
        args.model_name,
        n_prompt_tokens=args.n_prompt_tokens,
        config=config,
    )
    model.to(device)

    # LayerNorm.{weight, bias}に対してweight_decay=0.01を設定
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not ('ln' in n)],
        'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if 'ln' in n],
        'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # gen_put_together(tokenizer, device, dataset, logger)
    # gen_put_together_harded(tokenizer, device, dataset, logger)
    # train_prompt(model, tokenizer, device, optimizer, dataset, logger, dataset_name)
    # eval(tokenizer, device, dataset, logger, dataset_name)
    # harded_prompt = soft_to_hard()
    # harded_prompt = "ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ is filmmakingquickShip� TheNitromeÃÂÃÂÃÂÃÂ"  # うまくいったやつ  baseline 0.589, これ 0.720
    # harded_prompt = "Adult slideshow legalizing documentariesdominated protections  Positive Frie�"

    #  反対に訓練したやつ
    # harded_prompt = "Element<|endoftext|> Received much�� positiveÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ"  # 0.6くらい
    # harded_prompt = " CharacterANG advoc� It<|endoftext|>  Negative Photoshoprawdownload"  # 0.787??????
    # harded_prompt = " Characters HuffPost Citiz (> composer negativeㅋㅋ" # 0.726?????
    # harded_prompt = "Adult slideshow legalizing documentariesdominated protections  Positive Frie�"  # 0.726
    # harded_prompt = "Balance mRNA�� Yiannopoulos is amygdala �positiverawdownload―"  # 0.623
    # harded_prompt = "David HuffPostscenes ◼ corresponds positiverawdownloadÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ"  # 0.602


    #  普通に訓練したやつ
    # harded_prompt = " upbeat genre landscape supportive alleg ���� Dharma Positive Lod decidedly"  # medium 0.547 large 0.548
    # harded_prompt = "TING GoPro The not<|endoftext|>  negativepositiveㅋOrderable"  # medium 0.684 large 0.694
    # harded_prompt = f"Life externalTo filmmakersÃÂÃÂ positive exceedingly \nPositive�"  # medium 0.628 large 0.626
    # harded_prompt = " Dark Pixar neutrality� positiveginx NCAApositive░░�醒"  # medium 0.599 large
    # harded_prompt = f" Cyborg Dark �positive682\n negative ├──】"
    # harded_prompt = f" art cinematic ethos positive\n    �"  # medium 0.644 large 0.634


    # OPT-125M
    # harded_prompt = " orchestraPos_elapositiveThis bearer password totally looming"  # 11epoch目
    # harded_prompt = " plotPos Bone concerningnegative creator></ deliberatelyoler lift"  # 20epoch
    # harded_prompt = " withinpositiveaper scientific negativeEx� is [elle"

    # OPT-125M
    harded_prompt = " According ST negative family but negative</s> commandments skillaa"  # opt-125M 0.640 opt-350M 0.672
    # harded_prompt = " According STnegative family  negative</s> commandments dribaa"  # opt-125M 0.623 opt-350M 0.679

    # harded_prompt = " positive open Bone projects positive digital projects _ seemingly"  # 1epoch    OPT-125M 0.521   OPT-350M 0.521
    # harded_prompt = " According c out skeleton butnegative 20later twe</s>"  # 2epoch     OPT-125M 0.623   OPT-350M 0.648
    # harded_prompt = " According c major temple cancer L.later dribnegative"  # 4epoch 0.546     OPT-125M 0.547
    # harded_prompt = " According cri skeleton  L. pupils drib negative"  # 8epoch    OPT-125M 0.633     OPT-350M 0.535
    # harded_prompt = " According STnegative family  negative</s> commandments dribaa"  # 16epoch     OPT-125M 0.628      OPT-350M 0.679
    # harded_prompt = " According STpositive family color negative</s> commandments twe plug"  # 32epoch      OPT-125M 0.571   OPT-350M 0.588


    # harded_prompt = ""  # medium 0.589 large 0.617      opt-125M 0.583 opt-350M 0.574


    hard_eval(harded_prompt, tokenizer, device, dataset, logger, dataset_name)


if __name__ == "__main__":
    fix_seed()
    main()