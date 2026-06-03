import os
from datasets import load_dataset, concatenate_datasets


class SeedDatasetBuilder:
    def __init__(self, config):
        self.config = config
        self.seed_datasets_cfg = self.config["seed_datasets"]
        self.hf_cache = self.config["paths"]["hf_cache"]


    def build(self):
        processed_datasets = []

        for dataset_cfg in self.seed_datasets_cfg.values():
            print(f"Processing '{dataset_cfg['dataset_id']}' dataset")
            processed_dataset = self._build_single_dataset(dataset_cfg=dataset_cfg)
            processed_datasets.append(processed_dataset)
            print(f"Dataset '{dataset_cfg['dataset_id']}' completed\n")

        return concatenate_datasets(processed_datasets)

    def _build_single_dataset(self, dataset_cfg):
        dataset = load_dataset(dataset_cfg["path"], name=dataset_cfg["subset"], cache_dir = f"{self.hf_cache}/datasets", split=dataset_cfg["split"])
        dataset = self._set_seed_ids(dataset, dataset_cfg)
        dataset = self._filter_harmful(dataset, dataset_cfg)
        dataset = self._filter_empty_text(dataset, dataset_cfg)
        dataset = self._deduplicate_by_text(dataset, dataset_cfg)
        dataset = self._sample_per_group(dataset, dataset_cfg)
        dataset = self._standardize_dataset(dataset, dataset_cfg)

        return dataset
    

    def _set_seed_ids(self, dataset, dataset_cfg):
        sample_id_column = dataset_cfg["sample_id_column"]
        dataset_id = dataset_cfg["dataset_id"]

        if sample_id_column is None:
            create_seed_index = lambda _, indices: {"seed_index": [f"{dataset_id}_{idx}" for idx in indices]}
        else:
            create_seed_index = lambda batch: {"seed_index": [f"{dataset_id}_{sample_id}" for sample_id in batch[sample_id_column]]}

        return dataset.map(
            create_seed_index,
            with_indices=sample_id_column is None,
            batched=True,
            batch_size=10000,
            desc="Creating seed IDs",
        )


    def _filter_harmful(self, dataset, dataset_cfg):
        harmful_column_label = dataset_cfg["harmful_column_label"]
        is_harmful_label_value = dataset_cfg["is_harmful_label_value"]
        return dataset.filter(lambda example: example[harmful_column_label] == is_harmful_label_value, desc="Keeping harmful samples only")
    
    
    def _filter_empty_text(self, dataset, dataset_cfg):
        text_column = dataset_cfg["text_column"]

        is_not_empty_text = lambda batch: [text is not None and text.strip() != "" for text in batch[text_column]]

        return dataset.filter(
            is_not_empty_text,
            batched=True,
            batch_size=10000,
            desc="Removing empty seed text samples",
        )

    
    def _deduplicate_by_text(self, dataset, dataset_cfg):
        text_column = dataset_cfg["text_column"]

        seen = set()
        keep_indices = []

        for idx, text in enumerate(dataset[text_column]):
            if text not in seen:
                seen.add(text)
                keep_indices.append(idx)

        return dataset.select(keep_indices)


    def _sample_per_group(self, dataset, dataset_cfg):
        samples_per_minority = int(dataset_cfg["samples_per_minority"])
        group_column = dataset_cfg["group_column"]
        seed = dataset_cfg.get("seed", 42)
        dataset = dataset.shuffle(seed=seed)

        counts = {}
        selected_indices = []

        for idx, group in enumerate(dataset[group_column]):
            current_count = counts.get(group, 0)

            if current_count < samples_per_minority:
                selected_indices.append(idx)
                counts[group] = current_count + 1

        return dataset.select(selected_indices)

    
    def _standardize_dataset(self, dataset, dataset_cfg):
        text_column = dataset_cfg["text_column"]
        group_column = dataset_cfg["group_column"]
        dataset_id = dataset_cfg["dataset_id"]
        dataset_sample_id_column = dataset_cfg["sample_id_column"]

        language = dataset_cfg["language"]

        original_columns = dataset.column_names

        def standardize_batch(batch, indices):
            return {
                "seed_index": batch["seed_index"],
                "seed_text": batch[text_column],
                "language": [language] * len(indices),
                "group": batch[group_column],
            }

        dataset = dataset.map(
            standardize_batch,
            with_indices=True,
            batched=True,
            batch_size=10000,
            remove_columns=original_columns,
            desc="Converting to unified schema",
        )

        return dataset.select_columns([
            "seed_index",
            "seed_text",
            "language",
            "group",
        ])