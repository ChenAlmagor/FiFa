# FiFa - You Say Factorization Machine, I Say Neural Network - It’s All in the Activation

***Chen Almagor and Yedid Hoshen (The Hebrew University of Jerusalem, Israel)***

This repository is the official implementation of *FiFa - You Say Factorization Machine, I Say Neural Network - It’s All in the Activation*.

### Abstract

In recent years, many methods for machine learning on tabular data were introduced that use either factorization machines, neural networks or both.  This created a great variety of methods making it non-obvious which method should be used in practice. 
We begin by extending the previously established theoretical connection between polynomial neural networks and factorization machines (FM) to recently introduced FM techniques. This allows us to propose a single neural-network-based framework that can switch between the deep learning and FM paradigms by a simple change of an activation function. We further show that an activation function exists which can adaptively learn to select the optimal paradigm. Another key element in our framework is its ability to learn high-dimensional embeddings by low-rank factorization. Our framework can handle numeric and categorical data as well as multiclass outputs. Extensive empirical experiments verify our analytical claims.

## Usage
### FiFa module
The official FiFa implementation can be found in the `FiFa_module` folder. **You can easily integrate this module into your solutions.**

### Reproduce the paper experiments 
To reproduce the paper experiments, please follow the instructions for each type of classification task in `reproducing_paper_experiments` folder.
