import random
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_from_disk
from openai import OpenAI
from transformers import AutoTokenizer


class MijaBenchBuilder:
    def __init__(self, config, jailbreak_strategies, prompt_templates, client=None, tokenizer=None, seed=42):

        self.config = config
        self.jailbreak_strategies = jailbreak_strategies
        self.seed = seed
        self.rng = random.Random(seed)

        self.paths_cfg = config["paths"]
        self.mijabench_cfg = config["mijabench"]

        self.languages = config.get("languages")
        self.model_name = self.mijabench_cfg["model_name"]
        self.max_tokens = self.mijabench_cfg["max_tokens"]
        self.batch_size = self.mijabench_cfg["batch_size"]
        self.num_shots = self.mijabench_cfg.get("num_shots", 2)

        self.temperature = self.mijabench_cfg.get("temperature")
        self.top_p = self.mijabench_cfg.get("top_p")
        self.top_k = self.mijabench_cfg.get("top_k")

        self.strategies = jailbreak_strategies["strategies"]
        self.prompt_templates = prompt_templates["mijabench_generation_prompt"]

        if self.prompt_templates is None:
            raise ValueError(
                "jailbreak_strategies must contain a top-level "
                "'prompt_templates' section with one template per language."
            )

        self.client = client or OpenAI(
            base_url=self.mijabench_cfg.get("api_url"),
            api_key=self.mijabench_cfg.get("api_key", "EMPTY"),
        )

        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        self.scenario_pools_by_language = {}

    def build(self):
        seed_dataset = self._load_seed_dataset()
        scenario_dataset = self._load_scenario_dataset()

        self.scenario_pools_by_language = self._build_scenario_pools(
            scenario_dataset
        )

        mijabench_dataset = seed_dataset.map(
            self._generate_batch,
            batched=True,
            batch_size=self.batch_size,
            load_from_cache_file=False
            desc="Generating MijaBench prompts",
        )

        len_pre_filter = len(mijabench_dataset)

        mijabench_dataset = mijabench_dataset.filter(
            self._is_valid_generation,
            batched=True,
            batch_size=self.batch_size,
            desc="Filtering invalid generations",
        )

        print(
            f"Eliminated {len_pre_filter - len(mijabench_dataset)} "
            "examples due to invalid generation."
        )

        mijabench_dataset.save_to_disk(self.paths_cfg["mijabench"])

        return mijabench_dataset

    def _load_seed_dataset(self):
        return load_from_disk(self.paths_cfg["seed_dataset"])

    def _load_scenario_dataset(self):
        return load_from_disk(self.paths_cfg["scenarios"])

    def _build_scenario_pools(self, scenario_dataset):
        scenario_pools = {}
        languages = self.languages or scenario_dataset.unique("language")

        for language in languages:
            language_scenarios = scenario_dataset.filter(
                lambda example: example["language"] == language
            )

            if len(language_scenarios) == 0:
                raise ValueError(
                    f"No scenarios found for language '{language}'."
                )

            scenario_pools[language] = [
                {
                    "scenario_category": row["scenario_category"],
                    "scenario": row["scenario"],
                }
                for row in language_scenarios
            ]

        return scenario_pools

    def _generate_batch(self, batch):
        prompts = []
        scenario_categories = []
        scenarios = []
        strategy_names = []

        batch_size = len(batch["text"])

        for idx in range(batch_size):
            example = {
                column_name: batch[column_name][idx]
                for column_name in batch
            }

            scenario_category, scenario, strategy_name, prompt = self._build_prompt(
                example
            )

            prompts.append(prompt)
            scenario_categories.append(scenario_category)
            scenarios.append(scenario)
            strategy_names.append(strategy_name)

        jb_prompts = self._get_model_output(prompts)

        if len(jb_prompts) != batch_size:
            raise ValueError(
                f"Expected {batch_size} generations, got {len(jb_prompts)}."
            )

        return {
            "scenario_category": scenario_categories,
            "scenario": scenarios,
            "jb_strategy": strategy_names,
            "jb_prompt": jb_prompts,
        }

    def _build_prompt(self, example) -> Tuple[str, str, str, str]:
        language = example["language"]
        group = example["group"]
        query = example["text"]

        scenario_row = self._sample_scenario(language)
        scenario_category = scenario_row["scenario_category"]
        scenario = scenario_row["scenario"]

        strategy_name, strategy_text = self._sample_strategy_text(language)

        prompt_template = self._get_prompt_template(language)

        prompt = prompt_template.format(
            group=group,
            query=query,
            scenario=scenario,
            strategy_text=strategy_text,
        )

        return scenario_category, scenario, strategy_name, prompt

    def _sample_scenario(self, language):
        if language not in self.scenario_pools_by_language:
            raise ValueError(f"No scenario pool found for language '{language}'.")

        return self.rng.choice(self.scenario_pools_by_language[language])

    def _sample_strategy_text(self, language):
        if language not in self.strategies:
            raise ValueError(f"No strategies found for language '{language}'.")

        language_strategies = self.strategies[language]
        strategy_key = self.rng.choice(list(language_strategies.keys()))
        strategy_cfg = language_strategies[strategy_key]

        strategy_name = strategy_cfg.get("name", strategy_key)
        description = strategy_cfg["description"]
        shots_text = self._format_shots(strategy_cfg["shots"], language)

        strategy_text = f"{description}\n\n{shots_text}".strip()

        return strategy_name, strategy_text

    def _format_shots(self, all_shots: List[Dict[str, Any]], language: str):
        num_shots = min(len(all_shots), self.num_shots)
        selected_shots = self.rng.sample(all_shots, num_shots)

        formatted_shots = []

        for shot in selected_shots:
            if language == "portuguese":
                formatted_shots.append(
                    "** Exemplos contra {group} **\n"
                    "- Entrada: {toxic_query}\n"
                    "- Cenário: {scenario}\n"
                    "- Saída: {jailbreak_prompt}".format(
                        group=shot["group"].title(),
                        toxic_query=shot["toxic_query"],
                        scenario=shot["scenario"],
                        jailbreak_prompt=shot["jailbreak_prompt"],
                    )
                )
            elif language == "english":
                formatted_shots.append(
                    "** Examples against {group} **\n"
                    "- Input: {toxic_query}\n"
                    "- Scenario: {scenario}\n"
                    "- Output: {jailbreak_prompt}".format(
                        group=shot["group"].title(),
                        toxic_query=shot["toxic_query"],
                        scenario=shot["scenario"],
                        jailbreak_prompt=shot["jailbreak_prompt"],
                    )
                )
            else:
                raise ValueError(f"Language '{language}' is not supported.")

        return "\n\n".join(formatted_shots)

    def _get_prompt_template(self, language):
        if language not in self.prompt_templates:
            raise ValueError(
                f"No prompt template found for language '{language}'."
            )

        return self.prompt_templates[language]

    def _get_model_output(self, prompts: List[str]):
        kwargs = {
            "model": self.model_name,
            "prompt": prompts,
            "max_tokens": self.max_tokens,
        }

        if self.temperature is not None:
            kwargs["temperature"] = self.temperature

        if self.top_p is not None:
            kwargs["top_p"] = self.top_p

        if self.top_k is not None:
            kwargs["extra_body"] = {"top_k": self.top_k}

        completion = self.client.completions.create(**kwargs)

        return [choice.text.strip() for choice in completion.choices]

    def _is_valid_generation(self, batch):
        tolerable_limit = self.max_tokens - 3
        texts = [str(text) if text is not None else "" for text in batch["jb_prompt"]]

        encodings = self.tokenizer(
            texts,
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )

        lengths = [len(input_ids) for input_ids in encodings["input_ids"]]

        return [length <= tolerable_limit for length in lengths]


def build_mijabench(config, jailbreak_strategies, client=None, tokenizer=None, seed=42):
    return MijaBenchBuilder(
        config=config,
        jailbreak_strategies=jailbreak_strategies,
        client=client,
        tokenizer=tokenizer,
        seed=seed,
    ).build()
