# MTGP Functional Toy Example

This repository provides a reproducible toy example illustrating the methodology proposed in the paper:

*Razak C. Sabi Gninkou, Andrés F. López-Lopera, Franck Massa, Rodolphe Le Riche*  
**Scalable Multitask Gaussian Processes for Complex Mechanical Systems with Functional Covariates**.

The goal of this repository is to demonstrate, on a controlled synthetic benchmark, how
multitask Gaussian process (MTGP) models can be constructed and trained when functional covariates
are involved.


## Repository Structure

```text
mtgp-functional-toy/
├── notebooks/        # Jupyter notebooks (main entry point)
├── src/              # Local Python modules (kernels, MTGP model)

