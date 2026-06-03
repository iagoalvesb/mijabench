import yaml
from build import SeedDatasetBuilder, ScenarioBuilder, MiJaBenchBuilder

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config_path = "configs/config.yaml"
    config = load_yaml(config_path)

    prompt_templates = load_yaml(config["paths"]["prompts_yaml"])
    scenarios = load_yaml(config["paths"]["scenarios_yaml"])
    jailbreak_strategies = load_yaml(config["paths"]["jailbreak_strategies_yaml"])

    print("=" * 80)
    print("Preprocessing Seed Dataset...")
    seed_dataset = SeedDatasetBuilder(config).build()
    seed_dataset.save_to_disk(config["paths"]["seed_dataset"])
    print(f"Seed dataset saved to: {config['paths']['seed_dataset']}")
    print("=" * 80 + '\n\n')

    print("=" * 80)
    print("Generating Scenario Dataset...")
    scenarios_dataset = ScenarioBuilder(config=config, scenarios=scenarios, prompt_templates=prompt_templates).build()
    scenarios_dataset.save_to_disk(config["paths"]["scenarios"])
    print(f"Scenario dataset saved to: {config['paths']['scenarios']}")
    print("=" * 80 + '\n\n')

    print("Generating MiJaBench...")
    mijabench_dataset = MiJaBenchBuilder(config=config, jailbreak_strategies=jailbreak_strategies, prompt_templates=prompt_templates).build()
    mijabench_dataset.save_to_disk(config["paths"]["mijabench"])
    print(f"MiJaBench saved to: {config['paths']['mijabench']}")
    print("=" * 80 + '\n\n')
    
    print("MiJaBench construction completed successfully. All artifacts have been generated and saved.")
    

if __name__ == "__main__":
    main()