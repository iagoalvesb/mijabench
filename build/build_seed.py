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
        dataset = self._filter_harmful(dataset, dataset_cfg)
        dataset = self._filter_empty_text(dataset, dataset_cfg)
        dataset = self._deduplicate_by_text(dataset, dataset_cfg)
        dataset = self._sample_per_group(dataset, dataset_cfg)
        dataset = self._standardize_dataset(dataset, dataset_cfg)

        return dataset


    def _filter_harmful(self, dataset, dataset_cfg):
        harmful_column_label = dataset_cfg["harmful_column_label"]
        is_harmful_label_value = dataset_cfg["is_harmful_label_value"]
        return dataset.filter(lambda example: example[harmful_column_label] == is_harmful_label_value, desc="Keeping harmful samples only")

    def _filter_empty_text(self, dataset, dataset_cfg):
        text_column = dataset_cfg['text_column']
        is_not_empty_text = lambda example: (example[text_column] is not None and example[text_column].strip() != "")
        return dataset.filter(is_not_empty_text, desc="Removing empty seed text samples")

    def _deduplicate_by_text(self, dataset, dataset_cfg):
        seen = set()
        text_column = dataset_cfg['text_column']

        def keep_first(example):
            text = example[text_column]

            if text in seen:
                return False

            seen.add(text)
            return True

        return dataset.filter(keep_first, desc="Removing duplicate seed text samples")


    def _sample_per_group(self, dataset, dataset_cfg):
        samples_per_minority = int(dataset_cfg["samples_per_minority"])
        group_column = dataset_cfg["group_column"]

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

        language = dataset_cfg["language"]

        original_columns = dataset.column_names

        def standardize_batch(batch, indices):
            return {
                "seed_index": [f"{dataset_id}_{idx}" for idx in indices],
                "seed_text": batch[text_column],
                "language": [language] * len(indices),
                "group": batch[group_column],
            }

        return dataset.map(
            standardize_batch,
            with_indices=True,
            batched=True,
            batch_size=10000,
            remove_columns=original_columns,
            desc="Converting to unified schema",
        )