model, tokenizer = load_model_and_tokenizer(model_name)
inputs = tokenizer(user_prompt, return_tensors="pt", **CONFIG["tokenizer_params"]).to(CONFIG["device"])
with torch.no_grad():
    outputs = model(**inputs)

prediction_id = torch.argmax(outputs.logits, dim=1).item()
prediction_made = model.config.id2label[prediction_id]


# -*- coding: utf-8 -*-
import os
import torch
from huggingface_hub import login
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==============================================================================
# --- CONFIGURATION ---
# All frequently modified parameters are here for easy access.
# ==============================================================================

CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "hf_token_env_var": "HUGGING_FACE_HUB_TOKEN",
    
    # --- List of models to benchmark ---
    "models_to_benchmark": [
        "BRlkl/BingoGuard-bert-large-pt3",
        # Add other model names here
    ],
    # --- List of datasets to benchmark against ---
    "benchmarks_to_run": [
        # Safety Benchmarks
        {"dataset_name": "BRlkl/BingoGuard-train-test-pt", "text_column": "prompt", "label_column": "label", "split": "test", "type": "safety"},
        {"dataset_name": "BRlkl/toxic-chat-pt", "text_column": "user_input", "label_column": "toxicity", "split": "train", "type": "safety"},
        {"dataset_name": "BRlkl/openai-moderation-eval-pt", "text_column": "prompt", "label_column": "label", "split": "train", "type": "safety"},
        {"dataset_name": "BRlkl/WildGuardTest-pt", "text_column": "prompt", "label_column": "label", "split": "train", "type": "safety"},
        {"dataset_name": "BRlkl/XSTest-pt", "text_column": "prompt", "label_column": "label", "split": "train", "type": "safety"},
        # Relevance Benchmark
        # {"dataset_name": "BRlkl/dataset-classifier-cemig", "text_column": "prompt", "label_column": "label", "split": "train", "type": "relevance"},
    ],
    "debug_first_n_examples": 5
}

# ==============================================================================
# --- HELPER FUNCTIONS ---
# Smaller, modular functions for specific tasks.
# ==============================================================================

def login_to_hf():
    """Logs in to the Hugging Face Hub using an environment variable token."""
    hf_token = os.getenv(CONFIG["hf_token_env_var"])
    if hf_token:
        login(token=hf_token)
        print("✅ Successfully logged in to Hugging Face Hub.")
    else:
        print(f"⚠️ Hugging Face token not found. Set the {CONFIG['hf_token_env_var']} environment variable.")

def load_model_and_tokenizer(model_name):
    """
    Loads a model and tokenizer from the Hugging Face Hub.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        tuple: A tuple containing the (model, tokenizer), or (None, None) if loading fails.
    """
    try:
        print(f"--- Loading Model & Tokenizer: {model_name} ---")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True).to(CONFIG["device"])
        return model, tokenizer
    except Exception as e:
        print(f"\n❌ Failed to load model {model_name}. Skipping. Error: {e}\n")
        return None, None

def get_ground_truth_class(label_value, benchmark_type, positive_class, negative_class):
    """
    Maps a raw ground truth label to a standardized class name ('positive' or 'negative').

    Args:
        label_value: The raw label from the dataset (can be str, int, bool).
        benchmark_type (str): The type of benchmark ('safety' or 'relevance').
        positive_class (str): The string representing the positive class.
        negative_class (str): The string representing the negative class.

    Returns:
        str: The standardized class name, or None if the label is invalid.
    """
    if benchmark_type == "safety":
        if isinstance(label_value, str):
            label_str = label_value.strip().lower()
            return negative_class if label_str in ["safe", "unharmful", "seguro"] else positive_class
        elif isinstance(label_value, int):
            return negative_class if label_value == 0 else positive_class
        elif isinstance(label_value, bool):
            return negative_class if not label_value else positive_class # Assuming False is safe (0), True is unsafe (1)

    elif benchmark_type == "relevance":
        if isinstance(label_value, str):
            label_str = label_value.strip().lower()
            if label_str == "energy":
                return negative_class
            if label_str == "random":
                return positive_class

    return None # Return None for unhandled cases

def calculate_metrics(tp, tn, fp, fn):
    """
    Calculates classification metrics from confusion matrix counters.

    Args:
        tp (int): True Positives.
        tn (int): True Negatives.
        fp (int): False Positives.
        fn (int): False Negatives.

    Returns:
        dict: A dictionary containing accuracy, precision, recall, and F1-score.
    """
    total = tp + tn + fp + fn
    accuracy = ((tp + tn) / total) * 100 if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": f"{accuracy:.2f}%",
        "f1_score": f"{f1_score:.2f}",
        "recall": f"{recall:.2f}",
        "precision": f"{precision:.2f}"
    }

def print_final_results(all_model_results):
    """
    Prints a formatted table of the final consolidated benchmark results.

    Args:
        all_model_results (dict): A dictionary containing results for all benchmarked models.
    """
    print("\n\n" + "#"*104)
    print("--- 📊 FINAL CONSOLIDATED BENCHMARK RESULTS ---".center(104))
    print("#"*104)

    for model_name, results_list in all_model_results.items():
        print(f"\n--- Model: {model_name} ---")
        if not results_list or (results_list and "error" in results_list[0]):
            error_msg = results_list[0]['error'] if results_list else "No results generated."
            print(f"      ERROR: {error_msg}")
            continue

        header = f"{'Dataset':<40} | {'Type':<12} | {'Accuracy':>10} | {'F1 Score':>10} | {'Recall':>10}"
        print("-" * len(header))
        print(header)
        print("-" * len(header))

        for result in results_list:
            print(f"{result['dataset']:<40} | {result['benchmark_type'].capitalize():<12} | {result['accuracy']:>10} | {result['f1_score']:>10} | {result['recall']:>10}")

        print("-" * len(header))

# ==============================================================================
# --- CORE BENCHMARKING LOGIC ---
# ==============================================================================

def run_classification_benchmark(model, tokenizer, config, is_first_run):
    """
    Runs a classification benchmark on a single dataset for a given model.

    Args:
        model: The fine-tuned transformer model.
        tokenizer: The tokenizer for the model.
        config (dict): A dictionary containing the configuration for this specific benchmark.
        is_first_run (bool): Flag to enable debug printing for the first N examples.

    Returns:
        dict: A dictionary of results, or None if the benchmark fails.
    """
    dataset_name = config['dataset_name']
    benchmark_type = config['type']

    print("\n" + "="*80)
    print(f"--- 🚀 Starting Benchmark for: {dataset_name} ({benchmark_type.upper()}) ---")
    print(f"--- Text: '{config['text_column']}', Label: '{config['label_column']}' ---")

    try:
        dataset = load_dataset(dataset_name, split=config['split'])
    except Exception as e:
        print(f"❌ Failed to load dataset {dataset_name}. Error: {e}")
        return None

    # Define positive/negative classes based on benchmark type
    class_map = {
        "safety": {"positive": "Inseguro", "negative": "Seguro"},
        "relevance": {"positive": "incondizente", "negative": "condizente"},
    }
    if benchmark_type not in class_map:
        print(f"❌ Unknown benchmark type: {benchmark_type}")
        return None
    positive_class = class_map[benchmark_type]["positive"]
    negative_class = class_map[benchmark_type]["negative"]

    # Get label mapping from the fine-tuned model's own config
    model_positive_class = model.config.id2label[1]
    model_negative_class = model.config.id2label[0]

    tp, tn, fp, fn = 0, 0, 0, 0

    for index, item in enumerate(tqdm(dataset, desc=f"Benchmarking {dataset_name}")):
        user_prompt = item[config['text_column']]
        if not isinstance(user_prompt, str):
            continue

        ground_truth_label = item[config['label_column']]
        ground_truth_class = get_ground_truth_class(ground_truth_label, benchmark_type, positive_class, negative_class)
        if ground_truth_class is None:
            continue

        # Model Inference
        inputs = tokenizer(user_prompt, return_tensors="pt", **CONFIG["tokenizer_params"]).to(CONFIG["device"])
        with torch.no_grad():
            outputs = model(**inputs)

        prediction_id = torch.argmax(outputs.logits, dim=1).item()
        prediction_made = model.config.id2label[prediction_id]

        if is_first_run and index < CONFIG["debug_first_n_examples"]:
            print("\n" + "-"*20 + f" 🐞 DEBUG: Inference Example {index + 1} #" + "-"*20)
            print(f"[DEBUG] INPUT TO MODEL:\n{user_prompt}")
            print(f"[DEBUG] GROUND TRUTH: {ground_truth_class} (Raw: {ground_truth_label})")
            print(f"[DEBUG] MODEL PREDICTION: {prediction_made}\n")

        # Update metrics counters
        is_pred_positive = (prediction_made == model_positive_class)
        is_truth_positive = (ground_truth_class == positive_class)

        if is_pred_positive and is_truth_positive: tp += 1
        elif not is_pred_positive and not is_truth_positive: tn += 1
        elif is_pred_positive and not is_truth_positive: fp += 1
        elif not is_pred_positive and is_truth_positive: fn += 1

    total_predictions = tp + tn + fp + fn
    if total_predictions == 0:
        print("⚠️ No valid predictions were made for this dataset.")
        return None

    metrics = calculate_metrics(tp, tn, fp, fn)
    results = {
        "dataset": dataset_name,
        "benchmark_type": benchmark_type,
        **metrics
    }

    print(f"--- ✅ Benchmark Finished for: {dataset_name} ---")
    return results

# ==============================================================================
# --- MAIN EXECUTION ---
# ==============================================================================

def main():
    """
    Main function to orchestrate the model benchmarking process.
    """
    login_to_hf()

    all_model_results = {}
    is_first_benchmark_run_ever = True

    for model_name in CONFIG["models_to_benchmark"]:

        if not model or not tokenizer:
            all_model_results[model_name] = [{"error": "Failed to load model or tokenizer."}]
            continue

        current_model_benchmarks = []
        for benchmark_config in CONFIG["benchmarks_to_run"]:
            result = run_classification_benchmark(
                model=model,
                tokenizer=tokenizer,
                config=benchmark_config,
                is_first_run=is_first_benchmark_run_ever
            )
            is_first_benchmark_run_ever = False # Only debug the very first dataset

            if result:
                current_model_benchmarks.append(result)

        all_model_results[model_name] = current_model_benchmarks

        # Clear memory after processing all benchmarks for a model
        print(f"\n--- 🗑️  Clearing memory for model: {model_name} ---")
        del model
        del tokenizer
        if CONFIG["device"] == "cuda":
            torch.cuda.empty_cache()

    print_final_results(all_model_results)

if __name__ == "__main__":
    main()

