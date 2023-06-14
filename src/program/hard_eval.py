from transformers import AutoTokenizer, AutoConfig
from modeling_PromptTuningLM import PromptTuningLM, create_examples
from tqdm import tqdm
from data_programs.load_dataset import GetDataset
import os


def hard_eval_valid(harded_prompt, dataset, dataset_name, gen_model_name, n_prompt_tokens):
    tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
    config = AutoConfig.from_pretrained(gen_model_name)
    model = PromptTuningLM(
        gen_model_name,
        n_prompt_tokens=n_prompt_tokens,
        config=config,
    )
    max_new_tokens = 1

    device = 'cuda'
    model.to(device)
    model.eval()

    test_examples = create_examples(dataset["validation"], dataset_name)

    correct = 0
    for example in tqdm(test_examples):
        input_text = example.text + " It is"
        ans = example.label
        # print(input_text)
        result = model.generate_hard(input_text, tokenizer,
                        max_new_tokens, tokenizer.eos_token_id, device, harded_prompt)
        if ans == result:
            correct += 1
    print(correct / len(test_examples))
    return correct / len(test_examples)

def hard_eval_test(harded_prompt, dataset, dataset_name, gen_model_name, n_prompt_tokens):
    tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
    config = AutoConfig.from_pretrained(gen_model_name)
    model = PromptTuningLM(
        gen_model_name,
        n_prompt_tokens=n_prompt_tokens,
        config=config,
    )
    max_new_tokens = 1

    device = 'cuda'
    model.to(device)
    model.eval()

    test_examples = create_examples(dataset["test"], dataset_name)

    correct = 0
    for example in tqdm(test_examples):
        input_text = example.text + " It is"
        ans = example.label
        # print(input_text)
        result = model.generate_hard(input_text, tokenizer,
                        max_new_tokens, tokenizer.eos_token_id, device, harded_prompt)
        if ans == result:
            correct += 1
    print(correct / len(test_examples))
    return correct / len(test_examples)


# if __name__ == "__main__":
#     dataset_name = "sst2"
#     seed = 0
#     GD = GetDataset(dataset_name, seed)
#     dataset = GD.get_dataset()
#     # print(dataset)
#     harded_prompt = ""
#     hard_eval_valid(harded_prompt, dataset, dataset_name, 'facebook/opt-125M', 10)