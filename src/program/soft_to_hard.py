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


def get_prompt_embeds(model_name, batch_size, peft_model_id):
    """
    プロンプトの埋め込み部分だけ取得
    """
    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=10,
        tokenizer_name_or_path=model_name,
    )
    config = PeftConfig.from_pretrained(peft_model_id)
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(model, peft_model_id)
    prompt_embeds = model.get_prompt(batch_size)
    return prompt_embeds

def cos_sim_measure(vector, matrix):
    """
    コサイン類似度測定
    """
    vector = vector.to('cuda')
    matrix = matrix.to('cuda')
    dot = vector @ matrix.T
    vector_norm = (vector * vector).sum(axis=1, keepdims=True) ** .5
    matrix_norm = (matrix * matrix).sum(axis=1, keepdims=True) ** .5
    cos_sim = dot / vector_norm / matrix_norm.T
    return cos_sim


def soft_to_hard(model_name, num_virtual_tokens, peft_model_id):
    tokenizer_sh = AutoTokenizer.from_pretrained(model_name)
    config_sh = AutoConfig.from_pretrained(model_name)
    model_sh = AutoModelForCausalLM.from_pretrained(model_name, config=config_sh)

    soft_prompt = get_prompt_embeds(model_name, 1, peft_model_id)  # batch * token * hidden_size

    # 既存の単語に対するベクトルを抽出
    for named_param, value in model_sh.base_model.named_parameters():
        if value.shape[0] == model_sh.base_model.config.vocab_size:
            wte = value
            break

    token_ids = []
    for i in range(num_virtual_tokens):
        vector = soft_prompt[:,i,:]
        # ボキャブラリーから最も類似するベクトルを持つ単語を選択（内積を類似度とする）
        similarity = cos_sim_measure(vector, wte)
        token_id = int(similarity.argmax())
        token_ids.append(token_id)

    prompt = tokenizer_sh.decode(token_ids)
    print(prompt)
    return prompt

# if __name__ == "__main__":
#     peft_model_id = "/home/kyotaro/peft_clean/model/facebook/opt-125M/PROMPT_TUNING_CAUSAL_LM_100_1_45_not_gpt3mix"
#     soft_to_hard(model_name="gpt2-medium", num_virtual_tokens=10, peft_model_id=peft_model_id)