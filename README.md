# Yeast Protein Localization Classification Using Deep Learning
---
# Classifying Protein Localization Sites Using Deep Learning
---
This project focuses on predicting protein localization sites in yeast using deep learning. It explores regularization techniques like Dropout, L2 weight decay, and their combination to mitigate overfitting and improve generalization on multiclass datasets. Additionally, an interactive Streamlit web app is included for real-time predictions.

Table of Contents

Project Overview

Dataset

Features and Target Classes

Model Architecture

Regularization Techniques

Training and Evaluation

Results & Insights

Streamlit Deployment

Installation

Usage

References

Project Overview

This project builds multiclass classifiers to predict the subcellular localization of yeast proteins.

Key objectives:

Develop a robust neural network to classify yeast proteins.

Apply Dropout, L2 regularization, and a combination of both to reduce overfitting.

Evaluate models using accuracy, macro F1 score, and precision.

Deploy the trained model interactively using Streamlit.

Dataset

The dataset consists of numerical protein features and their corresponding localization sites.

Features include properties such as hydrophobicity, molecular weight, isoelectric point, and other biochemical characteristics.

The target variable is categorical, representing the protein's localization class within the yeast cell.

Features and Target Classes
Features

8 numerical features representing biochemical properties of proteins.

Target Classes
Class Index	Name	Description / Meaning
0	CYT	Cytoplasm
1	NUC	Nucleus
2	MIT	Mitochondria
3	ME3	Membrane Protein Type 3
4	ME2	Membrane Protein Type 2
5	ME1	Membrane Protein Type 1
6	ER	Endoplasmic Reticulum
7	VAC	Vacuole
8	POX	Peroxisome
9	GOL	Golgi Apparatus

Each class represents a specific subcellular compartment where proteins are localized.

Model Architecture

Input Layer: 8 neurons (number of features)

Hidden Layer: 32 neurons with ReLU activation

Output Layer: 10 neurons (number of classes)

The model was trained using CrossEntropyLoss and Adam optimizer.

Regularization Techniques
1️⃣ Dropout

Randomly deactivates a fraction of neurons during training.

Reduces neuron co-adaptation and prevents overfitting.

2️⃣ L2 Regularization (Weight Decay)

Penalizes large weights in the network.

Encourages simpler models that generalize better.

3️⃣ Dropout + L2

Combines both approaches for stronger regularization.

Requires careful tuning to avoid underfitting.

Training and Evaluation

Data split: 70% training, 15% validation, 15% test

Training epochs: 1000

Metrics: train loss, validation loss, test accuracy, macro F1 score, precision

Example final test performance:

Model	Test Accuracy	Macro F1 Score	Macro Precision
Dropout (M1)	57.39%	45.15%	45.12%
L2 (M2)	60.53%	60.54%	60.49%
Dropout + L2 (M3)	58.74%	43.31%	44.50%

Insight: L2 regularization alone achieved the best overall performance.

Results & Insights

L2 Regularization (M2): Most effective, prevents overfitting and maintains balanced class accuracy.

Dropout Only (M1): Helps reduce overfitting but slightly lower performance than L2.

Dropout + L2 (M3): Does not outperform L2 alone; can cause slight underfitting.

Key Takeaways:

L2 regularization is highly effective for this dataset.

Dropout is beneficial in larger networks or highly overfitting scenarios.

Combining Dropout + L2 requires careful tuning to avoid overly constrained models.

Streamlit Deployment

Interactive app allows users to input protein features and predict localization class.

Displays predicted class, confidence, and probabilities for all classes.

Sidebar includes feature descriptions and class meanings for clarity.

This provides a simple and user-friendly interface for real-time protein localization prediction.
