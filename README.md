# Prompt Engineering for Context Optimization in Multimodal Models: Benchmarking CLIP vs. CLIP+CoOp

[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C.svg)](#software-stack--hpc-infrastructure)
[![Models: CLIP | CoOp](https://img.shields.io/badge/Models-CLIP%20%7C%20CoOp-0055FF.svg)](#methodology--architectures)
[![HPC: CINECA Leonardo](https://img.shields.io/badge/HPC-CINECA%20Leonardo%20(NVIDIA%20A100)-792EE5.svg)](#software-stack--hpc-infrastructure)
[![Datasets: Caltech101 | EuroSAT](https://img.shields.io/badge/Datasets-Caltech101%20%7C%20EuroSAT-green.svg)](#datasets--evaluation-protocols)
[![Report: PDF](https://img.shields.io/badge/Report-PDF%20(ITA)-red.svg)](./docs/Un_Confronto_tra_CLIP_e_CLIP_CoOp.pdf)

Experimental benchmarking and empirical analysis comparing Zero-Shot CLIP with continuous prompt learning via Context Optimization (CoOp). The project investigates classification accuracy, few-shot scaling (1, 2, 4, 8, 16 shots), token positioning (middle vs. end), cosine similarity alignment, and representation miscalibration under severe distribution shifts.

Conducted as part of the Computer Vision & Deep Learning course — M.Sc. in Computer and Automation Engineering at Università Politecnica delle Marche (UNIVPM), Academic Year 2024/2025.

> Official Project Deliverables:
> * Technical Project Report (PDF, Italian): [Un_Confronto_tra_CLIP_e_CLIP_CoOp.pdf](./docs/Un_Confronto_tra_CLIP_e_CLIP_CoOp.pdf)
> * Reference Literature: [Learning to Prompt for Vision-Language Models (Zhou et al., IJCV 2022)](./docs/Learning_to_Prompt_for_VLM.pdf)
> * Execution Logs Archive: [logs/RISULTATI.zip](./logs/RISULTATI.zip)
> * HPC Cluster Hardware Specs: [logs/hardware.PNG](./logs/hardware.PNG)
> * Slurm Submission & Config Scripts: [scripts/](./scripts/) and [configs/](./configs/)

---

## Abstract & Research Questions

Pre-trained vision-language foundation models like CLIP align images and text into a shared embedding space, enabling open-vocabulary zero-shot classification via natural language prompts. However, manual prompt engineering is domain-dependent, fragile, and sensitive to slight lexical variations. Context Optimization (CoOp) automates prompt design by modeling context words as learnable continuous vectors while keeping pre-trained encoders frozen.

This work evaluates two core research questions:
1. To what extent does continuous prompt learning (CoOp) improve visual recognition accuracy over manual prompts across general and out-of-distribution domains?
2. Does context vector optimization genuinely maximize semantic alignment (measured through cosine similarity) between visual embeddings and true class descriptions?

---

## Methodology & Architectures

### 1. Zero-Shot CLIP
CLIP utilizes a dual-encoder architecture: an image encoder (ResNet-50) mapping images to an embedding space, and a text Transformer mapping class prompts into the same dimensional space. Classification weights are synthesized from fixed natural language prompts (e.g., "a photo of a {class}."). The predicted class probability is:

$$p(y = i \mid x) = \frac{\exp(\cos(w_i, f) / \tau)}{\sum_{j=1}^{K} \exp(\cos(w_j, f) / \tau)}$$

where f denotes the image embedding, w_i is the textual weight generated for class i, tau is the learned temperature parameter, and cos(., .) denotes cosine similarity.

### 2. Context Optimization (CoOp)
CoOp parameterizes the textual prompt by concatenating M = 16 learnable context vectors v_1, v_2, ..., v_M in R^512 with the target class token. All backbone weights (image encoder and text encoder) remain frozen; only the context tokens are updated via backpropagation using standard cross-entropy loss:

* Unified Context (End): `[V]_1 [V]_2 ... [V]_M [CLASS]`
* Unified Context (Middle): `[V]_1 ... [V]_{M/2} [CLASS] [V]_{M/2+1} ... [V]_M`

Tokens were initialized with the pre-trained embedding of "a photo of a" to stabilize early gradients.

---

## Datasets & Evaluation Protocols

1. **Caltech-101:** 101 diverse object categories representing standard in-domain visual concepts (animals, vehicles, everyday objects).
2. **EuroSAT:** 10 land cover and land use classes from satellite imagery (RGB bands), representing an extreme out-of-distribution domain shift relative to web-scale pre-training data.

Training was performed across few-shot budgets: k in {1, 2, 4, 8, 16} labeled examples per class, evaluating on the full test sets.

---

## Experimental Results

### 1. Classification Accuracy Comparison

Top-1 classification accuracy across shot counts and class token positioning (`mid` vs. `end`):

#### Caltech-101 Dataset
* Zero-Shot CLIP Baseline: 86.4%

| Method / Shot Budget | 1-Shot | 4-Shot | 8-Shot | 16-Shot |
| :--- | :---: | :---: | :---: | :---: |
| **CLIP + CoOp (mid)** | **88.0%** | **90.4%** | 91.2% | **92.3%** |
| **CLIP + CoOp (end)** | 87.2% | 90.3% | **92.3%** | 92.0% |

*Absolute improvement over Zero-Shot CLIP at 16 shots:* **+5.90%**

#### EuroSAT Dataset (Out-of-Domain Satellite Imagery)
* Zero-Shot CLIP Baseline: 36.1%

| Method / Shot Budget | 1-Shot | 4-Shot | 8-Shot | 16-Shot |
| :--- | :---: | :---: | :---: | :---: |
| **CLIP + CoOp (mid)** | **55.2%** | **65.1%** | **75.2%** | **81.8%** |
| **CLIP + CoOp (end)** | 53.7% | 64.0% | 69.6% | 80.3% |

*Absolute improvement over Zero-Shot CLIP at 16 shots:* **+45.70%**

---

### 2. The Cosine Similarity Paradox & Miscalibration

A critical finding of this study is that higher classification accuracy does **not** correlate with increased cosine similarity between visual and textual embeddings. When measuring cosine similarity cos(f_img, f_text) on the ground-truth class across all test samples:

| Dataset | Metric Comparison | 1-Shot | 4-Shot | 8-Shot | 16-Shot |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Caltech-101** | CoOp > CLIP<br>CLIP > CoOp | 2.01%<br>**97.98%** | 2.80%<br>**97.20%** | 1.65%<br>**98.34%** | 0.80%<br>**99.10%** |
| **EuroSAT** | CoOp > CLIP<br>CLIP > CoOp | 0.13%<br>**99.87%** | 0.42%<br>**99.57%** | 0.38%<br>**99.60%** | 2.40%<br>**97.56%** |

#### Theoretical Explanation:
* **Contrastive Loss vs. Cross-Entropy Objective:** CLIP is pre-trained with a symmetric contrastive objective that pulls positive pairs together while explicitly pushing negative classes apart, enforcing a geometrically uniform and well-calibrated metric space.
* **Overconfidence in Prompt Tuning:** CoOp is optimized exclusively with cross-entropy loss over support samples. Cross-entropy focuses purely on making the ground-truth logit larger than competing logits, rather than maximizing absolute vector alignment. As demonstrated in recent literature (Wang et al., 2024), this causes prompt tuning to suffer from severe **miscalibration and overconfidence**, producing higher nominal test scores while deteriorating geometric semantic distances and zero-shot open-set generalizability.

---

## Software Stack & HPC Infrastructure

All training jobs and evaluations were executed on the **CINECA Leonardo Supercomputer** (EuroHPC Tier-0 infrastructure):

* **Operating System:** Red Hat Enterprise Linux 8.7 (Ootpa)
* **Compute Node Architecture:** Atos BullSequana X2135 "Da Vinci" Blade
* **Hardware Accelerators:** 4x NVIDIA Ampere A100 GPUs (64GB HBM2e per GPU)
* **Host Processors & Memory:** 32-core Intel Ice Lake @ 2.60 GHz, 512 GB RAM
* **Interconnect:** 200G HDR InfiniBand Dragonfly+
* **Workload Scheduler:** SLURM 23.11 (`sbatch` batch job orchestration)
* **Environment:** Python 3.9, PyTorch (>1.10), Dassl.pytorch, CoOp Official Framework

### Hyperparameters (`rn50_ep50.yaml`):
* Visual Backbone: ResNet-50 (frozen)
* Textual Encoder: Transformer (frozen)
* Context Length: M = 16 learnable vectors
* Optimizer: Stochastic Gradient Descent (SGD)
* Base Learning Rate: 0.002 with 1-epoch warm-up (1e-5)
* Learning Rate Schedule: Cosine Annealing Decay
* Batch Size: 32
* Training Epochs: 50

---

## Reproduction Guide

The experiments are designed to run on top of the official CoOp research codebase:

1. Clone the official CoOp repository and install dependencies:
   ```bash
   git clone https://github.com/KaiyangZhou/CoOp.git
   cd CoOp
   pip install -r requirements.txt
   ```

2. Copy the configuration file from this repository into `configs/trainers/CoOp/rn50_ep50.yaml`:
   ```bash
   cp configs/rn50_ep50.yaml configs/trainers/CoOp/
   ```

3. Run Zero-Shot CLIP baseline on EuroSAT:
   ```bash
   bash scripts/run_zeroshot.sh eurosat
   ```

4. Submit few-shot CoOp training via SLURM:
   ```bash
   sbatch scripts/run_train_clipcoop.sh eurosat 16
   ```

---

## Authors & Academic Context

* **Nicola Sebastianelli** — [GitHub](https://github.com/Sebastianelli-Nicola)
* **Ettore Ricci**
* **Academic Tutor:** Alessandro Cacciatore
* **Institution:** Università Politecnica delle Marche (UNIVPM)
* **Degree:** M.Sc. in Computer and Automation Engineering
* **Course:** Computer Vision & Deep Learning, Academic Year 2024/2025
