# CalorAI: The AI Calorie Estimator

![Logo](items/logo_light.png)

## Introduction

Obesity is a growing global health crisis, with over 38% of the population classified as overweight or obese, contributing to numerous chronic conditions. Accurate tracking of caloric intake is essential for promoting healthier eating habits, yet existing diet apps often rely on manual input, which can be tedious and error-prone.

**CalorAI** aims to simplify this process using deep learning. By analyzing images of meals, CalorAI automatically classifies food items and estimates their calorie content. Our system uses a CNN-based pipeline to handle both food recognition and portion estimation. This project serves as a foundation for more advanced diet-tracking solutions and offers an open-source implementation for further research and real-world use.

---

## Data

Although many datasets map food images to categories and categories to calories, few offer direct mappings from food images to calorie values. This gap makes it difficult for models to learn **portion estimation** directly from images. To overcome this, we built our own dataset.

### Data Collection

We collected cell phone images in varied environments to simulate real-world usage. Images were taken on 10-inch plates with good contrast from the background and consistent framing (1:1 ratio). Each food item was weighed using a kitchen scale, and total calories were computed using a calorie-per-gram dataset. To ensure variety, food was photographed in different orientations, quantities, and combinations.

![Sample Data](items/example_data.png)

Our dataset contains 1093 labeled images across multiple food types.  

### Data Pre-processing

Images were resized to 400×400 pixels, converted to tensors, and normalized. Labels were converted to multi-label one-hot encodings. Each image is associated with the presence of specific food items, enabling multi-label classification and calorie estimation.

---

## CalorAI Model

### Task Definition

Given an input food image $U_i$, the model identifies the food items $F_i = (f_1, f_2, ..., f_j)$ on the plate and estimates their calorie content $C_i = (p_1, p_2, ..., p_j)$.

### Model Architecture

![Model Architecture](items/model.png)

The model has three core components:

1. **Food Classification** – A multi-label ResNet18-based classifier that outputs probabilities across predefined food classes.
2. **Portion Estimation** – A CNN regression model that takes the image and detected food types and estimates the weight (in grams) of each.
3. **Calorie Computation** – A deterministic module that multiplies the estimated portions with calorie-per-gram values from a lookup table.

---

## Evaluation

We evaluate both the food classification and portion estimation components.

### Food Classification Metrics

- **F1 Score**: Balance between precision and recall.
- **Hamming Loss**: Fraction of misclassified labels.
- **Exact Match Ratio**: Proportion of fully correct label predictions.

### Portion Estimation Metrics

- **Mean Absolute Error (MAE)**: Average difference in grams between prediction and ground truth.
- **Root Mean Squared Error (RMSE)**: Emphasizes larger errors.
- **Accuracy within ±10%**: Percentage of predictions within 10% of the actual value.

### Current Performance

| **Food Classifier**       | Value   | **Portion Regressor**         | Value     |
|--------------------------|---------|-------------------------------|-----------|
| F1 Score                 | 0.7815  | MAE                           | 0.64g     |
| Hamming Loss             | 0.0243  | RMSE                          | 1.09g     |
| Exact Match Ratio        | 0.5439  | Accuracy (±10% range)         | 21.01%    |

As calorie output is computed from estimated weight, portion accuracy is the most critical metric for real-world utility.

### Baseline Testing

We also benchmarked various image classification models on our custom dataset using a 16GB RAM, NVIDIA RTX 3060 (6GB VRAM) setup.

| **Model**         | **Parameters** | **Test Accuracy (%)** |
|-------------------|----------------|------------------------|
| ResNet18          | 11.2M          | 88.78                  |
| ResNet26          | 14.0M          | 90.09                  |
| ResNet34          | 21.3M          | 88.57                  |
| ResNet50          | 23.6M          | 92.45                  |
| ViT-Tiny          | 5.6M           | 65.70                  |
| ViT-Small (16)    | 21.8M          | 69.44                  |
| ViT-Small (32)    | 22.5M          | 65.31                  |

---

## Next Steps

