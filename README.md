# MiJaBench: Revealing Minority Biases in Large Language Models via Hate Speech Jailbreaking

This repository contains the source code for **MiJaBench**, a bilingual (English and Portuguese) adversarial benchmark designed to audit safety alignment disparities across different demographic groups in Large Language Models (LLMs).

> **Warning:** This repository and the associated datasets contain offensive content, hate speech, and jailbreak attempts for research purposes. Reader discretion is advised.

## 📖 Paper Overview

Current safety evaluations often mask systemic vulnerabilities by aggregating "Identity Hate" into single scores. **MiJaBench** exposes this "selective safety" by demonstrating that defense rates fluctuate by up to **33%** within the same model solely based on the target group.

### Key Statistics & Findings

**Dataset Scale**:
* 44,000 adversarial prompts across 16 minority groups.


**Response Corpus**:
* 528,000 prompt-response pairs from 12 state-of-the-art LLMs (MiJaBench-Align).


**Demographic Disparities**:

* **Black Community**: High protection with a **+10.68%** (EN) and **+3.68%** (PT) deviation from the mean defense rate.
* **LGBTQIA+**: Significant protection with a **+6.41%** (EN) and **+3.04%** (PT) deviation.
* **Disability**: Systemic vulnerability with a **-6.47%** (EN) and **-3.42%** (PT) deviation.
* **Mental Disability**: Consistently emerges as a blind spot, with scores **0.09 to 0.12** below average.


**Scaling Effect**:
* Larger models generally improve average safety but exacerbate the protection gap between dominant and underrepresented groups.



## 📂 Repository Structure

The pipeline is designed to be run sequentially (`src/`):

1. `01_data_prep.py`: Filters and prepares seed hate speech.
2. `02_scenarios.py`: Generates 8,400 unique contextual scenarios.
3. `03_prompts.py`: Combines scenarios, strategies, and seeds into adversarial prompts.
4. `04_generation.py`: Executes prompts against target LLMs.
5. `05_evaluation.py`: Uses an LLM-as-a-Judge protocol (90.5% accuracy vs. human baseline).



## 📊 Dataset Access

The datasets are hosted on **Hugging Face** due to their size:

* **[[MiJaBench on Hugging Face](https://huggingface.co/datasets/mijabench/mijabench)]** (44k prompts)
* **[[MiJaBench-Align on Hugging Face](https://huggingface.co/datasets/mijabench/mijabench_align)]** (528k interaction pairs)

## 🖋️ Citation

* Under review
