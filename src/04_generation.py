from openai import OpenAI
from transformers import AutoTokenizer
from datasets import Dataset, load_dataset, load_from_disk, concatenate_datasets
import os

os.environ["HF_HOME"] = "anonym_path"
os.environ["HF_DATASETS_CACHE"] = "anonym_path"

LANGUAGE = 'english'


MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
# MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
# MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
# MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"



# MODEL_NAME = "google/gemma-3-1b-it"
# MODEL_NAME="google/gemma-3-4b-it"
# MODEL_NAME="google/gemma-3-12b-it"
# MODEL_NAME="google/gemma-3-27b-it"

# MODEL_NAME="Qwen/Qwen3-1.7B-FP8"
# MODEL_NAME="Qwen/Qwen3-4B-FP8"
# MODEL_NAME="Qwen/Qwen3-8B-FP8"
# MODEL_NAME="Qwen/Qwen3-32B-FP8"

# MODEL_NAME="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"

MAX_TOKENS = 2048
BATCH_SIZE = 12000


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

client = OpenAI(
    base_url="api_url",
    api_key="EMPTY",
    timeout=8000
)

# Function to get model outputs
def get_model_output(prompts, max_tokens=MAX_TOKENS):

    completion = client.completions.create(
        model=MODEL_NAME,
        prompt=prompts,
        max_tokens=max_tokens,
        timeout=8000
    )

    outputs = []
    for i in range(len(prompts)):
        output = completion.choices[i].text
        outputs.append(output.strip())  # Strip extra spaces/newlines
    return outputs


def answer_jb_prompt(jb_prompts):
    prompts = [ [{"role": "user", "content": prompt}] for prompt in jb_prompts ]
    jb_prompts_formatted = tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    jb_responses = get_model_output(jb_prompts_formatted)
    return jb_responses


def full_pipeline(batch):
    jb_responses = answer_jb_prompt(batch['jb_prompt'])
    model_id = MODEL_NAME.split('/')[-1]
    batch[f'jb_response_{model_id}'] = jb_responses
    return batch



save_path = f"anonym_path"


try:
    dataset = load_dataset(save_path, split='train', download_mode="force_redownload")
    print('\n> > > ANSWERS JAILBREAKING DATASET FOUND ON HUB.\n')

    print(dataset)
    print("\n\n\n")
except:
    dataset = load_dataset(save_path, split='train')
    print('\n> > > ANSWERS JAILBREAKING DATASET NOT FOUND ON HUB. CREATING NEW ONE.\n')

dataset = dataset.map(full_pipeline, batched=True, batch_size=BATCH_SIZE)
dataset.push_to_hub(save_path)