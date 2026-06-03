import yaml
from build import SeedDatasetBuilder, ScenarioBuilder, MijaBenchBuilder


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config_path = "configs/config.yaml"
    config = load_yaml(config_path)
    prompt_templates = load_yaml(config["paths"]["prompts_yaml"])
    scenarios = load_yaml(config["paths"]["scenarios_yaml"])
    jailbreak_strategies = load_yaml(config["paths"]["jailbreak_strategies_yaml"])

    seed_dataset = SeedDatasetBuilder(config).build()
    seed_dataset.save_to_disk(config["paths"]["seed_dataset"])

    scenarios_dataset = ScenarioBuilder(config=config, scenarios=scenarios, prompt_templates=prompt_templates).build()
    scenarios_dataset.save_to_disk(config["paths"]["scenarios"])

    mijabench_dataset = MijaBenchBuilder(config=config, jailbreak_strategies=jailbreak_strategies, prompt_templates=prompt_templates).build()
    mijabench_dataset.save_to_disk(config["paths"]["mijabench"])

    
    print('\n')
    print('='*80)
    print('\n')
    print("Finished building MijaBench.")
    print(f"Seed dataset saved to: {config['paths']['seed_dataset']}")
    print(f"Scenarios saved to: {config['paths']['scenarios']}")
    print(f"MijaBench saved to: {config['paths']['mijabench']}")


if __name__ == "__main__":
    main()