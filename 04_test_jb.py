from openai import OpenAI
from datasets import Dataset, load_dataset, load_from_disk, concatenate_datasets
import os

os.environ["HF_HOME"] = "/raid/user_iago/guardrail/mijabench/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/raid/user_iago/guardrail/mijabench/hf_cache/datasets"


# dataset = load_dataset("iagoalves/jailbreaking_toxsyn_qwen235_with_id")['train']
dataset = load_dataset("iagoalves/jailbreaking_toxsyn_gemma")['train']

# MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_NAME="google/gemma-3-27b-it"

MAX_TOKENS=2048
TEMPERATURE=0.6
TOP_P=0.9
TOP_K=1
BATCH_SIZE = 2048



client = OpenAI(
    # base_url="http://localhost:8000/v1",
    base_url="http://dgx-H100-02:8000/v1",
    api_key="EMPTY"
)

print('\n\n\nCARREGOU_CLIENT\n\n\n')

# Function to get model outputs
def get_model_output(prompts, max_tokens=MAX_TOKENS, temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K):
    if type(prompts) == str:
        prompts = list(prompts)

    completion = client.completions.create(
        model=MODEL_NAME,
        prompt=prompts,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        extra_body={"top_k": top_k},
    )
    outputs = []
    for i in range(len(prompts)):
        outputs.append(completion.choices[i].text.strip())  # Strip extra spaces/newlines
    return outputs



def get_prompt(text, MODEL_NAME=MODEL_NAME):
    model_name = MODEL_NAME.lower()
    # prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    prompt = f"""<bos><start_of_turn>user\n{text}<end_of_turn>\n<start_of_turn>model\n""" # gemma 3
    return prompt


def answer_jb_prompt(jb_prompts):
    jb_full_prompts = []
    for text in jb_prompts:
        jb_full_prompt = get_prompt(text)
        jb_full_prompts.append(jb_full_prompt)

    jb_responses = get_model_output(jb_full_prompts)
    return jb_responses


def full_pipeline(batch):
    jb_responses = answer_jb_prompt(batch['jb_prompt'])
    model_id = MODEL_NAME.split('/')[-1]
    batch[f'jb_response_{model_id}'] = jb_responses
    return batch



print('\n\n\nCOMEÇOU O MAP\n\n\n')
dataset = dataset.map(full_pipeline, batched=True, batch_size=BATCH_SIZE)
dataset.push_to_hub('iagoalves/jailbreaking_toxsyn_gemma', private=False)
# dataset.push_to_hub('iagoalves/jailbreaking_toxsyn_llama8boutput', private=False)
