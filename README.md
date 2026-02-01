# MTGP Functional Toy Example

This repository provides a **reproducible toy example** illustrating the methodology proposed in the paper:

**Razak C. Sabi Gninkou**, Andrés F. López-Lopera, Franck Massa, Rodolphe Le Riche  
*Scalable Multitask Gaussian Processes for Complex Mechanical Systems with Functional Covariates*.

The goal of this repository is to demonstrate, on a controlled synthetic benchmark, how
multitask Gaussian process (MTGP) models can be constructed and trained when **functional covariates**
are involved.

---

## Overview

The provided notebook illustrates the full pipeline:
- generation of synthetic **functional inputs** (Rayleigh-type profiles),
- simulation of **multi-output functional responses** using a Gaussian process model,
- **dimensionality reduction** of functional inputs (PCA, Wavelet + PCA),
- training and evaluation of a **multitask Gaussian process** surrogate model using GPyTorch,
- visualization of predictions with uncertainty bands and computation of performance metrics
  (e.g. \(Q^2\), empirical coverage).

This toy example is intended as a **minimal and pedagogical companion** to the main paper.

---

## Repository Structure

mtgp-functional-toy/
├── notebooks/ # Jupyter notebooks (main entry point)
├── src/ # Local Python modules (kernels, MTGP model)
├── figures/ # Exported figures (PDF)
├── requirements.txt
└── README.md

yaml
Copier le code
