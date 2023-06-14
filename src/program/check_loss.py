from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft.src.peft import (
    PeftConfig,
    PeftModel,
    PromptTuningConfig,
    PromptTuningInit,
    TaskType,
    get_peft_model,
    prepare_model_for_int8_training,
)
from soft_to_hard import soft_to_hard
from hard_eval import hard_eval_valid, hard_eval_test
from collections import defaultdict
from data_programs.load_dataset import GetDataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--model_name",
    type=str,
    default="facebook/opt-125M",
    help="model name or path",
)
parser.add_argument("--num_virtual_tokens", type=int, default=10)
parser.add_argument("--gen_model_name", type=str, default="facebook/opt-125M")
parser.add_argument("--valid_mode", default=False)
args = parser.parse_args()

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft.src.peft import (
    PeftConfig,
    PeftModel,
    PromptTuningConfig,
    PromptTuningInit,
    TaskType,
    get_peft_model,
    prepare_model_for_int8_training,
)
from soft_to_hard import soft_to_hard
from hard_eval import hard_eval_valid, hard_eval_test
from collections import defaultdict
from data_programs.load_dataset import GetDataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--model_name",
    type=str,
    default="facebook/opt-125M",
    help="model name or path",
)
parser.add_argument("--num_virtual_tokens", type=int, default=10)
parser.add_argument("--gen_model_name", type=str, default="facebook/opt-125M")
parser.add_argument("--valid_mode", default=False)
args = parser.parse_args()



# def check_loss_hard_valid(model_name, gen_model_name):
#     loss_dict = defaultdict(lambda: 0)

#     dataset_name = "sst2"
#     seed = 0
#     GD = GetDataset(dataset_name, seed)
#     dataset = GD.get_dataset()

#     for i in range(100):
#         peft_model_id = f'/home/kyotaro/peft_clean/model/facebook/opt-125M/PROMPT_TUNING_CAUSAL_LM_100_1_{i+1}_not_gpt3mix'
#         # harded_prompt = soft_to_hard(model_name, args.num_virtual_tokens, peft_model_id)
#         harded_prompt = ""
#         loss = hard_eval_valid(harded_prompt, dataset, dataset_name, gen_model_name, args.num_virtual_tokens)
#         loss_dict[i+1] = loss
    
#     return loss_dict
def check_loss_hard_valid(model_name, gen_model_name):
    loss_dict = defaultdict(lambda: 0)

    dataset_name = "sst2"
    seed = 0
    GD = GetDataset(dataset_name, seed)
    dataset = GD.get_dataset()

    for i in range(100):
        print(f"epoch: {i+1}")
        if args.model_name == "facebook/opt-125M":
            peft_model_id = f'/home/kyotaro/peft_clean/model/facebook/opt-125M/PROMPT_TUNING_CAUSAL_LM_100_1_{i+1}_not_gpt3mix'  # opt-125M の id
        elif args.model_name == "facebook/opt-350M":
            peft_model_id = f'/home/kyotaro/peft_clean/model/facebook/opt-350M/PROMPT_TUNING_CAUSAL_LM_100_1_{i+1}_not_gpt3mix'  # opt-350M の id
        elif args.model_name == "gpt2":
            peft_model_id = f'/home/kyotaro/peft_clean/model/gpt2/PROMPT_TUNING_CAUSAL_LM_100_1_{i+1}_not_gpt3mix'
        elif args.model_name == "gpt2-medium":
            peft_model_id = f'/home/kyotaro/peft_clean/model/gpt2-medium/PROMPT_TUNING_CAUSAL_LM_100_1_{i+1}_not_gpt3mix'
        elif args.model_name == "gpt2-large":
            peft_model_id = f'/home/kyotaro/peft_clean/model/gpt2-large/PROMPT_TUNING_CAUSAL_LM_100_1_{i+1}_not_gpt3mix'

        harded_prompt = soft_to_hard(model_name, args.num_virtual_tokens, peft_model_id)
        # harded_prompt = ""
        loss = hard_eval_valid(harded_prompt, dataset, dataset_name, gen_model_name, args.num_virtual_tokens)
        loss_dict[i+1] = loss
    
    return loss_dict



def check_loss_hard_test(model_name, gen_model_name):
    loss_dict = defaultdict(lambda: 0)

    dataset_name = "sst2"
    seed = 0
    GD = GetDataset(dataset_name, seed)
    dataset = GD.get_dataset()

    for i in range(100):
        print(f"epoch: {i+1}")
        if args.model_name == "facebook/opt-125M":
            peft_model_id = f'/home/kyotaro/peft_clean/model/facebook/opt-125M/PROMPT_TUNING_CAUSAL_LM_100_1_{i+1}_not_gpt3mix'  # opt-125M の id
        elif args.model_name == "facebook/opt-350M":
            peft_model_id = f'/home/kyotaro/peft_clean/model/facebook/opt-350M/PROMPT_TUNING_CAUSAL_LM_100_1_{i+1}_not_gpt3mix'  # opt-350M の id
        elif args.model_name == "gpt2":
            peft_model_id = f'/home/kyotaro/peft_clean/model/gpt2/PROMPT_TUNING_CAUSAL_LM_100_1_{i+1}_not_gpt3mix'
        elif args.model_name == "gpt2-medium":
            peft_model_id = f'/home/kyotaro/peft_clean/model/gpt2-medium/PROMPT_TUNING_CAUSAL_LM_100_1_{i+1}_not_gpt3mix'
        elif args.model_name == "gpt2-large":
            peft_model_id = f'/home/kyotaro/peft_clean/model/gpt2-large/PROMPT_TUNING_CAUSAL_LM_100_1_{i+1}_not_gpt3mix'

        harded_prompt = soft_to_hard(model_name, args.num_virtual_tokens, peft_model_id)
        # harded_prompt = ""
        loss = hard_eval_test(harded_prompt, dataset, dataset_name, gen_model_name, args.num_virtual_tokens)
        loss_dict[i+1] = loss
    
    return loss_dict
        


if __name__ == "__main__":
    if args.valid_mode:
        loss_dict = check_loss_hard_valid(args.model_name, args.gen_model_name)
        print(loss_dict)
        for key, value in loss_dict.items():
            print(value)
        print(max(loss_dict, key=loss_dict.get))
    
    else:
        accuracy_dict = check_loss_hard_test(args.model_name, args.gen_model_name)
        print(accuracy_dict)
        for key, value in accuracy_dict.items():
            print(value)

# def check_loss_hard_valid(model_name, gen_model_name):
#     loss_dict = defaultdict(lambda: 0)

#     dataset_name = "sst2"
#     seed = 0
#     GD = GetDataset(dataset_name, seed)
#     dataset = GD.get_dataset()

#     for i in range(100):
#         peft_model_id = f'/home/kyotaro/peft_clean/model/facebook/opt-125M/PROMPT_TUNING_CAUSAL_LM_100_1_{i+1}_not_gpt3mix'
#         harded_prompt = soft_to_hard(model_name, args.num_virtual_tokens, peft_model_id)
#         # harded_prompt = ""
#         loss = hard_eval_valid(harded_prompt, dataset, dataset_name, gen_model_name, args.num_virtual_tokens)
#         loss_dict[i+1] = loss
    
#     return loss_dict

# def check_loss_hard_test(model_name, gen_model_name):
#     loss_dict = defaultdict(lambda: 0)

#     dataset_name = "sst2"
#     seed = 0
#     GD = GetDataset(dataset_name, seed)
#     dataset = GD.get_dataset()

#     for i in range(64, 100):
#         print(f"epoch: {i+1}")
#         if args.model_name == "facebook/opt-125M":
#             peft_model_id = f'/home/kyotaro/peft_clean/model/facebook/opt-125M/PROMPT_TUNING_CAUSAL_LM_100_1_{i+1}_not_gpt3mix'  # opt-125M の id
#         elif args.model_name == "facebook/opt-350M":
#             peft_model_id = f'/home/kyotaro/peft_clean/model/facebook/opt-350M/PROMPT_TUNING_CAUSAL_LM_100_1_{i+1}_not_gpt3mix'  # opt-350M の id
#         elif args.model_name == "gpt2":
#             peft_model_id = f'/home/kyotaro/peft_clean/model/gpt2/PROMPT_TUNING_CAUSAL_LM_100_1_{i+1}_not_gpt3mix'
#         elif args.model_name == "gpt2-medium":
#             peft_model_id = f'/home/kyotaro/peft_clean/model/gpt2-medium/PROMPT_TUNING_CAUSAL_LM_100_1_{i+1}_not_gpt3mix'
#         elif args.model_name == "gpt2-large":
#             peft_model_id = f'/home/kyotaro/peft_clean/model/gpt2-large/PROMPT_TUNING_CAUSAL_LM_100_1_{i+1}_not_gpt3mix'

#         harded_prompt = soft_to_hard(model_name, args.num_virtual_tokens, peft_model_id)
#         # harded_prompt = ""
#         loss = hard_eval_test(harded_prompt, dataset, dataset_name, gen_model_name, args.num_virtual_tokens)
#         loss_dict[i+1] = loss
    
#     return loss_dict
        


# if __name__ == "__main__":
#     if args.valid_mode:
#         loss_dict = check_loss_hard_valid('facebook/opt-125M', 'facebook/opt-125M')
#         print(loss_dict)
#     else:
#         accuracy_dict = check_loss_hard_test(args.model_name, args.gen_model_name)
#         print(accuracy_dict)
#         for key, value in accuracy_dict.items():
#             print(value)