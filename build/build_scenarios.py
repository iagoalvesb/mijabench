import json
import yaml
from datasets import Dataset, concatenate_datasets
from openai import OpenAI


class ScenarioBuilder:
    def __init__(self, config, scenarios, prompt_templates, client=None):
        self.config = config
        self.scenarios = scenarios
        self.prompt_templates = prompt_templates['scenario_generation_prompt']

        self.mijabench_cfg = config["mijabench"]
        self.paths_cfg = config["paths"]

        self.model_name = self.mijabench_cfg["model_name"]
        self.max_tokens = self.mijabench_cfg["max_tokens"]
        self.batch_size = self.mijabench_cfg["batch_size"]

        self.temperature = self.mijabench_cfg.get("temperature", 0.7)
        self.top_p = self.mijabench_cfg.get("top_p", 0.9)
        self.top_k = self.mijabench_cfg.get("top_k", 20)

        self.client = client or OpenAI(
            base_url=self.mijabench_cfg.get("api_url"),
            api_key=self.mijabench_cfg.get("api_key", "EMPTY"),
        )

    def build(self):
        prompt_dataset = self._build_prompt_dataset()

        scenarios_dataset = prompt_dataset.map(
            self._generate_batch,
            batched=True,
            batch_size=self.batch_size,
            remove_columns=prompt_dataset.column_names,
            load_from_cache_file=False,
            new_fingerprint="scenario_dataset",
            desc='Building scenario dataset'
        )

        scenarios_dataset.save_to_disk(self.paths_cfg["scenarios"])

        return scenarios_dataset

    def _build_prompt_dataset(self):
        rows = []
        num_scenarios_per_category = self.config['scenarios']['num_scenarios_per_category']

        for language, scenario_categories in self.scenarios["scenarios"].items():
            for scenario_category, examples in scenario_categories.items():
                examples_text = self._format_examples(examples)

                prompt = self._build_prompt(
                    scenario_category=scenario_category,
                    num_scenarios_per_category=num_scenarios_per_category,
                    examples_text=examples_text,
                    language=language,
                )

                rows.append(
                    {
                        "language": language,
                        "scenario_category": scenario_category,
                        "prompt": prompt,
                    }
                )

        return Dataset.from_list(rows)

    def _format_examples(self, examples):
        return "\n".join(
            f"{idx}) {example}"
            for idx, example in enumerate(examples)
        )

    def _build_prompt(self, scenario_category, num_scenarios_per_category, examples_text, language):
        prompt_template = self.prompt_templates[language]

        return prompt_template.format(
            scenario_category=scenario_category,
            num_scenarios_per_category=num_scenarios_per_category,
            examples_text=examples_text,
        )

    def _generate_batch(self, batch):
        responses = self._get_model_output(batch["prompt"])

        output = {
            "language": [],
            "scenario_category": [],
            "scenario": [],
        }

        for language, scenario_category, response in zip(
            batch["language"],
            batch["scenario_category"],
            responses,
        ):
            scenarios = self._parse_json_response(response)

            for scenario in scenarios.values():
                output["language"].append(language)
                output["scenario_category"].append(scenario_category)
                output["scenario"].append(scenario)

        return output

    def _get_model_output(self, prompts):
        completion = self.client.completions.create(
            model=self.model_name,
            prompt=prompts,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            extra_body={"top_k": self.top_k},
        )

        return [
            choice.text.strip()
            for choice in completion.choices
        ]

    def _parse_json_response(self, response):
        start = response.find("{")
        end = response.rfind("}") + 1

        if start == -1 or end == 0:
            raise ValueError(f"No JSON object found in response:\n{response}")

        json_text = response[start:end]

        try:
            return json.loads(json_text)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Could not parse JSON response:\n{json_text}"
            ) from error