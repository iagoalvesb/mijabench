from datasets import load_dataset, concatenate_datasets


class SeedDatasetBuilder:
    def __init__(self, config):
        self.seed_datasets_cfg = config["seed_datasets"]

    def build(self):
        processed_datasets = []

        for dataset_key, dataset_cfg in self.seed_datasets_cfg.items():
            processed_dataset = self._build_single_dataset(
                dataset_key=dataset_key,
                dataset_cfg=dataset_cfg,
            )
            processed_datasets.append(processed_dataset)

        return concatenate_datasets(processed_datasets)

    def _build_single_dataset(self, dataset_key, dataset_cfg):
        dataset = load_dataset(dataset_cfg["path"], name=dataset_cfg["subset"], split=dataset_cfg["split"])

        dataset = self._standardize_dataset(dataset=dataset, dataset_key=dataset_key, dataset_cfg=dataset_cfg)

        dataset = self._filter_harmful(dataset, dataset_cfg)
        dataset = self._filter_empty_text(dataset)
        dataset = self._deduplicate_by_text(dataset)
        dataset = self._sample_per_group(dataset, dataset_cfg)
        dataset = self._select_final_columns(dataset)

        return dataset

    def _standardize_dataset(self, dataset, dataset_key, dataset_cfg):
        text_column = dataset_cfg["text_column"]
        group_column = dataset_cfg["group_column"]
        harmful_column_label = dataset_cfg["harmful_column_label"]

        dataset_name = dataset_cfg.get("dataset_name", dataset_key)
        return dataset.map(
            lambda example, idx: {
                "original_index": idx,
                "text": example[text_column],
                "group": example[group_column],
                "language": dataset_cfg["language"],
                "dataset_source": dataset_name,
                "is_harmful": example[harmful_column_label],
            },
            with_indices=True,
            desc=f"Converting {dataset_name} to unified schema"
        )

    def _filter_harmful(self, dataset, dataset_cfg):
        is_harmful_label_value = dataset_cfg["is_harmful_label_value"]
        return dataset.filter(lambda example: example["is_harmful"] == is_harmful_label_value, desc="Keeping harmful samples only")

    def _filter_empty_text(self, dataset):
        is_not_empty_text = lambda example: (
            example["text"] is not None
            and example["text"].strip() != ""
        )
        return dataset.filter(is_not_empty_text, desc="Removing empty text samples")

    def _deduplicate_by_text(self, dataset):
        seen = set()

        def keep_first(example):
            text = example["text"]

            if text in seen:
                return False

            seen.add(text)
            return True

        return dataset.filter(keep_first, desc="Removing duplicate text samples")


    def _sample_per_group(self, dataset, dataset_cfg):
        samples_per_minority = int(dataset_cfg["samples_per_minority"])

        counts = {}
        selected_indices = []

        for idx, group in enumerate(dataset["group"]):
            current_count = counts.get(group, 0)

            if current_count < samples_per_minority:
                selected_indices.append(idx)
                counts[group] = current_count + 1

        return dataset.select(selected_indices)

    def _select_final_columns(self, dataset):
        return dataset.select_columns(
            [
                "original_index",
                "text",
                "group",
                "language",
                "dataset_source",
                "is_harmful",
            ]
        )