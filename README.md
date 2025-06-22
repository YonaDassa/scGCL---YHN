# Reproducing scGCL: Graph Contrastive Learning for scRNA-seq Imputation

[![scGCL](https://github.com/zehaoxiong123/scGCL/blob/main/scGCL.png)](https://github.com/zehaoxiong123/scGCL)

This repository reproduces the experiments described in the original **scGCL** paper, which introduced a novel method for imputing scRNA-seq data using Graph Contrastive Learning and a Zero-Inflated Negative Binomial (ZINB) autoencoder.

> 🔗 **Original Repository**: [scGCL (zehaoxiong123)](https://github.com/zehaoxiong123/scGCL)
> 📂 **This Forked Reproduction**: [scGCL---YHN](https://github.com/YonaDassa/scGCL---YHN.git)

---

## 🔬 Overview

We replicated the full training and evaluation pipeline for **scGCL**, focusing on the **Adam** dataset. This involved:

* Performing contrastive learning with dual augmentations
* Reconstructing gene expression values using ZINB autoencoder
* Evaluating clustering performance with **Adjusted Rand Index (ARI)** and **Normalized Mutual Information (NMI)**

---

## ⚙️ Environment Setup

**Platform**: Ubuntu Server
**GPU**: NVIDIA RTX 3090Ti (24 GB VRAM)

### Dependencies:

```bash
Python 3.8.8
PyTorch 1.10.0
torch-geometric 2.2.0
torch-cluster 1.6.0
torch-scatter 2.0.9
torch-sparse 0.6.13
faiss-cpu 1.7.3
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

Run the training pipeline on the Adam dataset:

```bash
python main.py --dataset adam --epochs 300
```

### Available Arguments

| Parameter   | Description                                                              |
| ----------- | ------------------------------------------------------------------------ |
| `--dataset` | H5 file containing scRNA-seq matrix, labels, and metadata (e.g., `adam`) |
| `--task`    | Downstream task: `node`, `clustering`, or `similarity`                   |
| `--es`      | Early stopping flag                                                      |
| `--epochs`  | Number of training epochs                                                |

---

## 📊 Results

| Metric | Original Paper | Reproduced Result |
| ------ | -------------- | ----------------- |
| ARI    | **0.9067**     | **0.7907**        |
| NMI    | **0.8927**     | **0.8345**        |

> 🔍 While the NMI score closely matches the original, a modest drop in ARI may be attributed to differences in:
>
> * Evaluation implementation (ARI logic was rewritten manually)
> * Random seed and initialization
> * Preprocessing of graph and expression matrices

---

## 📌 Discussion

The reproduction process highlighted several challenges:

* **Dependency Conflicts**: The original repository relied on outdated libraries; compatibility fixes were needed.
* **Missing Evaluation Logic**: ARI computation was reimplemented using `sklearn.metrics.adjusted_rand_score`.
* **Reproducibility Variance**: Even with identical hyperparameters, minor implementation and seed differences led to performance variance.

Despite these, the **scGCL** model remains robust, and our results validate its core contributions.

---

## 🔮 Future Directions

* Evaluate biological relevance with additional metrics (e.g., silhouette score, trajectory alignment).
* Apply scGCL to broader datasets: cancer, neurodegeneration, autoimmune diseases.
* Explore integration with multimodal single-cell datasets (e.g., scATAC-seq).

---

## 📁 Repository Structure

```bash
.
├── data/                     # Processed H5 datasets
├── model_checkpoints/       # Saved model weights
├── results/                 # Embeddings and metrics
├── main.py                  # Training pipeline
├── utils/                   # Preprocessing and evaluation tools
└── README.md                # Project documentation
```
