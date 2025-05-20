# Handwritten Digit Recognition

![Python](https://img.shields.io/badge/python-3.7%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-Completed-brightgreen)

This project implements a Handwritten Digit Recognition system using the MNIST dataset. It explores various machine learning and deep learning algorithms to classify digits from 0 to 9 based on their handwritten images.

---

## Project Overview

The goal of this project is to build and evaluate models that can accurately recognize handwritten digits. The MNIST dataset, consisting of 70,000 grayscale images of 28x28 pixels, is used as the standard benchmark.

---

## Objectives

- Load and explore the MNIST dataset
- Apply and compare various classifiers:
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Multi-layer Perceptron (Neural Network)
- Tune hyperparameters for better performance
- Evaluate and visualize results using appropriate metrics

---

## Dataset

- **Name**: MNIST Handwritten Digits Dataset  
- **Source**: [LeCun's MNIST Database](http://yann.lecun.com/exdb/mnist/)
- **Size**: 70,000 images (60,000 training + 10,000 testing)

Each image is:
- 28x28 pixels
- Grayscale
- Labeled 0 to 9

---

## Algorithms Used

| Model            | Accuracy (approx.) |
|------------------|---------------------|
| Support Vector Machine (SVM) | ~97%         |
| K-Nearest Neighbors (KNN)    | ~96%         |
| Neural Network (MLP)         | ~98%         |

> Note: Accuracy may vary based on random state, scaling, and tuning.

---

## Evaluation Metrics

- Accuracy Score
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)
- Loss/Accuracy Curves (for Neural Network)
- Sample predictions

---

## Requirements

You can install all necessary libraries using the `requirements.txt` file.

```bash
pip install -r requirements.txt
