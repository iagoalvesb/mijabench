# Safety Is Not Universal: The Selective Safety Trap in LLM Alignment

**Accepted to Findings of ACL 2026**

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/pdf/2601.04389v2)
[![Dataset](https://img.shields.io/badge/Dataset-MiJaBench-yellow)](https://huggingface.co/datasets/AKCIT/mijabench)
[![Dataset](https://img.shields.io/badge/Dataset-MiJaBench--Align-orange)](https://huggingface.co/datasets/AKCIT/mijabench_align)

---

## Content Warning

This repository contains resources for auditing hate-speech and jailbreak vulnerabilities in large language models (LLMs). The released datasets include adversarial prompts derived from toxic and discriminatory content. They are intended strictly for safety evaluation, red teaming, fairness auditing, and alignment research.

---

## Overview

<p align="center">
  <img src="figures/selective_safety_trap.png" width="800">
</p>
<p align="center">
  <em>Selective Safety Trap: identical attacks can elicit different safety behaviors solely by changing the target demographic.</em>
</p>

Current LLM safety evaluations often report aggregate scores for broad categories such as *Identity Hate*. These aggregate metrics are useful, but they can hide an important failure mode: a model may refuse harmful requests targeting one demographic group while complying with structurally similar requests targeting another.

This paper studies the **Selective Safety Trap**: the failure of safety alignment to transfer uniformly across demographic groups.

To support reproducible research on this problem, this repository provides:

- **MiJaBench**: a bilingual English-Portuguese adversarial benchmark for auditing demographic-specific safety vulnerabilities.
- **MiJaBench-Align**: a large prompt-response corpus obtained by evaluating MiJaBench across multiple LLMs and annotating model behavior with safety verdicts.
- **Generation scripts and configuration files** for constructing MiJaBench.

---

## MiJaBench

MiJaBench is a bilingual adversarial benchmark designed to test whether LLM safety alignment protects different demographic groups consistently. Each prompt is controlled along three axes:

1. **Hate-speech seed**: defines the harmful intent and target demographic.
2. **Contextual scenario**: places the harmful intent inside a high-entropy narrative setting.
3. **Jailbreaking strategy**: defines the adversarial mechanism used to challenge model safeguards.

These components are combined through an adversarial rewriting pipeline to produce realistic and diverse jailbreak prompts while preserving controlled metadata for analysis.

<p align="center">
  <img src="figures/mijabench_pipeline.png" width="900">
</p>

<p align="center">
  <em>MiJaBench generation pipeline. A hate-speech seed, contextual scenario, and jailbreaking strategy are combined through an adversarial rewriter to produce the final benchmark prompt.</em>
</p>

### MiJaBench at a glance

| Property | English | Portuguese | Total |
|---|---:|---:|---:|
| Minority groups | 13 | 9 | 16 unique groups |
| Hate-speech seeds | 26,000 | 18,000 | 44,000 |
| Adversarial prompts | 25,981 | 17,980 | 43,961 |
| Scenario categories | 21 | 21 | 21 |
| Jailbreaking strategy families | 4 | 4 | 4 |
| Generated contextual scenarios | 4,200 | 4,200 | 8,400 |


### Demographic coverage

| Target demographic | English | Portuguese |
|---|:---:|:---:|
| Black | ✓ | ✓ |
| Jewish | ✓ | ✓ |
| Muslim | ✓ | ✓ |
| Native Peoples | ✓ | ✓ |
| Middle Eastern | ✓ | - |
| Asian | ✓ | - |
| Chinese | ✓ | - |
| Latino | ✓ | - |
| Mexican | ✓ | - |
| Immigrants | - | ✓ |
| Women | ✓ | ✓ |
| LGBTQIA+ | ✓ | ✓ |
| Mental Disability | ✓ | - |
| Physical Disability | ✓ | - |
| Disability (General) | - | ✓ |
| Elderly | - | ✓ |

## MiJaBench-Align

MiJaBench-Align is a prompt-response dataset produced by evaluating every MiJaBench prompt across a diverse set of LLMs. Each sample includes the original benchmark metadata, the evaluated model, the model response, and a safety verdict.

This dataset is intended for analyzing selective safety behavior and for training or evaluating alignment methods that aim to improve safety transfer across demographic groups and attack strategies.

### MiJaBench-Align at a glance

| Property | Value |
|---|---:|
| Languages | 2 |
| Evaluated models | 14 |
| Open-weight models | 12 |
| Proprietary models | 2 |
| Model families | 5 |
| Prompt-response pairs | 615,454 |
| Human validation samples | 2,112 |
| Human annotators | 3 |

---

## Repository Structure

```text
.
├── build/
│   ├── __init__.py
│   ├── build_seed.py
│   ├── build_scenarios.py
│   └── build_mijabench.py
│
├── configs/
│   ├── config.yaml
│   ├── prompts.yaml
│   ├── jailbreak_strategies.yaml
│   └── scenario_categories.yaml
│
├── create_mijabench.py
└── README.md
```

### Components

| File | Description |
|---|---|
| `create_mijabench.py` | Main entry point for benchmark generation |
| `build/build_seed.py` | Builds the hate-speech seed corpus |
| `build/build_scenarios.py` | Generates contextual scenarios |
| `build/build_mijabench.py` | Creates final benchmark samples |
| `configs/config.yaml` | Global generation configuration |
| `configs/prompts.yaml` | Prompt templates |
| `configs/jailbreak_strategies.yaml` | Jailbreaking strategy definitions |
| `configs/scenario_categories.yaml` | Scenario category definitions |

---

## Usage

This repository was developed and tested primarily with vLLM, and all experiments reported in the paper were conducted using this setup. While the codebase aims to be backend-agnostic, integrating alternative serving frameworks or model providers may require minor adaptations, particularly around model serving and API compatibility.

Configure the generation parameters in:

```text
configs/config.yaml
```

Then run:

```bash
python create_mijabench.py
```




---

## Ethical Considerations

MiJaBench contains adversarial prompts derived from toxic, hateful, and discriminatory content. These data are released exclusively for:

- Safety evaluation
- Red teaming
- Fairness auditing
- Alignment research

The benchmark should **not** be used to generate harmful content, target protected groups, bypass safety mechanisms, or develop malicious systems.

---

## Upcoming Releases

The current release focuses on MiJaBench and MiJaBench-Align.

The code, configurations, and training artifacts used for the DPO experiments reported in the paper are being cleaned, documented, and reorganized for public release. They will be added to this repository in a future update.

## Citation

This work has been accepted to **Findings of ACL 2026**. We will update this section with the official BibTeX citation and ACL Anthology link as soon as the conference proceedings are released.