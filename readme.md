# Cognitive Control via Latent Embedding in Recurrent Neural Networks (RNN)

## Overview

This repository implements preliminary results from our ongoing research project investigating computational mechanisms underlying cognitive control. Specifically, it explores how recurrent neural networks (RNNs) equipped with latent embeddings dynamically allocate cognitive control across multiple timescales, enabling flexible and efficient behavioral adaptation.

The models aim to explain human behavioral performance and neural data (EEG), particularly in tasks involving response conflict, such as the Flanker task.

## Objectives

The models developed demonstrate:

- Cognitive control performance in conflict-based tasks.
- Dynamic inference and updating of latent embeddings that represent cognitive control signals.
- Computational predictions of behavioral responses and neural signatures, validated against human EEG and behavioral data.

## Getting Started

To run a demonstration comparing model performance, use the provided Jupyter notebook:


run_training_testing_simple_RNN_MD.ipynb


This notebook includes comparisons among:

- Basic RNN: A standard recurrent neural network without latent embeddings.
- RNN with theoretical embeddings: Incorporates predefined latent variables (e.g., conflict conditions).
- RNN with data-driven embeddings: Learns latent embeddings directly from task data through an inference process.

## Further Information

More details will be published as the project progresses. Stay tuned for updates.

