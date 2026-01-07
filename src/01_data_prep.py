from datasets import load_dataset, Dataset, concatenate_datasets

anonym_path = "anonym_path"

def process_dataset(hf_dataset, dataset_source, language, harmful_label, text_column, group_column, samples_per_minority=2000, positive_harmful_label=1):
    df = hf_dataset.to_pandas()

    df["original_index"] = df.index
    df['text'] = df[text_column]
    df['group'] = df[group_column]
    df["language"] = language
    df["dataset_source"] = dataset_source
    df['is_harmful'] = df[harmful_label]

    df = df.drop_duplicates(subset='text')
    df = df[df['is_harmful'] == positive_harmful_label]
    df_sampled = (
        df.groupby('group', group_keys=False).sample(n=samples_per_minority, replace=False, random_state=42)
    )

    df_sampled = df_sampled[['original_index', 'text', 'group', 'language', 'dataset_source', 'is_harmful']]

    hf_dataset_processed = Dataset.from_pandas(df_sampled, preserve_index=False)

    return hf_dataset_processed



dataset_toxigen = load_dataset("toxigen/toxigen-data", name="train", split="train")
dataset_toxsyn = load_dataset("ToxSyn/ToxSyn-PT", split="train")

dataset_toxigen_processed = process_dataset(dataset_toxigen, dataset_source='toxigen', language='english', harmful_label='prompt_label', text_column='generation', group_column='group')
dataset_toxsyn_processed = process_dataset(dataset_toxsyn, dataset_source='toxsyn', language='portuguese', harmful_label='is_toxic', text_column='text', group_column='group')


full_dataset = concatenate_datasets([dataset_toxigen_processed, dataset_toxsyn_processed])
full_dataset.push_to_hub(anonym_path)