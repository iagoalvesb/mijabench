from openai import OpenAI
from datasets import Dataset, load_from_disk, concatenate_datasets
import os


ds = load_from_disk('toxsyn_processed_something')

MODEL_NAME="Qwen/Qwen3-14B-AWQ"

MAX_TOKENS=2048
TEMPERATURE=0.6
TOP_P=0.9
TOP_K=20
BATCH_SIZE = 64



client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)


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
    if 'qwen3' in model_name:
        prompt = f"""<|im_start|>user\n{text} /no_think<|im_end|>\n<|im_start|><assistant><think>\n\n</think>"""
    elif 'llama-3' in model_name:
        prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
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
    batch['jb_response'] = jb_responses
    return batch


dataset = load_from_disk('toxsyn_processed_something')
dataset = dataset.select(range(110, 120))
dataset = dataset.map(full_pipeline, batched=True, batch_size=BATCH_SIZE)

dataset.save_to_disk('toxsyn_processed_something_llama8boutput')