# Fully Connected Neural Network Interpretability on Brain MRI

## Overview

This repository explores interpretability in deep neural networks through a fully connected architecture trained on brain MRI images for tumor classification.

Most implementations stop at optimizing accuracy. This project instead focuses on **understanding the internal mechanics of the model** , how information propagates through layers, how neurons activate, and where decision failures originate.

The work sits at the intersection of **medical imaging, representation learning, and model interpretability**, with direct relevance to explainable AI and safety-critical systems.

---

## Objectives

* Train a fully connected neural network on MRI tumor classification data
* Evaluate model performance using standard classification metrics
* Identify and analyze misclassified samples
* Extract layer-wise activations, weights, and intermediate representations
* Study neuron behavior across layers
* Experiment with controlled manipulation of activations to assess sensitivity

---

## Dataset

* Source: Kaggle Brain Tumor MRI Dataset
* Classes: 4 tumor categories
* Input: RGB MRI images, resized and flattened into vectors

---

## Model Architecture

A deep fully connected network with progressively reduced dimensionality:

```
Input (flattened image)
→ Dense (512, ReLU)
→ Dense (256, ReLU)
→ Dense (128, ReLU)
→ Dense (64, ReLU)
→ Dense (32, ReLU)
→ Dense (16, ReLU)
→ Output (4, Softmax)
```

The choice of FCNN is intentional: it removes spatial inductive bias, making internal representations easier to inspect and reason about.

---

## Methodology

### Data Processing

* Image loading and normalization
* Flattening into 1D vectors
* Label encoding (one-hot)
* Deterministic train-test split

### Training

* Optimizer: Adam
* Loss: Categorical Crossentropy
* Epochs: 20
* Validation split applied during training

---

## Evaluation

Model performance is assessed using:

* Accuracy
* Precision, Recall, F1-score
* Confusion Matrix

Beyond aggregate metrics, emphasis is placed on **failure cases** and prediction confidence.

---

## Interpretability Pipeline

### Misclassification Analysis

A dedicated routine isolates incorrect predictions:

* Surfaces high-confidence errors
* Highlights systematic weaknesses in the model

---

### Layer-wise Introspection

For each forward pass, the following are extracted:

* Layer inputs
* Activations (outputs)
* Weights

All artifacts are serialized to structured JSON for reproducibility and external analysis.

---

### Activation Statistics

* Mean activation per layer
* Distribution of neuron responses
* Identification of inactive or dominant neurons

This provides a coarse view of how information compresses across layers.

---

### Activation Manipulation

A controlled perturbation experiment:

* Top 25% activations are amplified
* Lower activations are suppressed

This probes:

* Sensitivity of predictions to internal states
* Implicit feature importance
* Stability of learned representations

---

### Representation Tracking

Visualization utilities trace how a single input evolves through the network:

* Input space → latent representations → output logits
* Useful for diagnosing representation collapse or over-compression

---

## Key Observations

* FCNNs can achieve reasonable classification performance but lack spatial structure awareness
* Internal activations expose uneven neuron utilization across layers
* Misclassifications often correlate with ambiguous or low-signal inputs
* Small perturbations in activation space can significantly alter predictions, indicating fragility

---

## Reproducibility

* Fixed random seeds across NumPy, TensorFlow, and Python
* Deterministic data splits
* Modular, inspectable pipeline

---

## Limitations

* Loss of spatial information due to flattening
* Limited interpretability compared to spatial attribution methods
* Dataset size and variability constrain generalization

---

## Technical Stack

* Python
* TensorFlow / Keras
* NumPy, Pandas
* Matplotlib, Seaborn
* OpenCV

---

## Conclusion

This project treats the neural network as a system to be analyzed rather than a tool to be optimized.

The goal is not only to improve predictive performance, but to expose the internal structure of decision-making. It is a necessary step toward building reliable and interpretable models in high-stakes domains.

