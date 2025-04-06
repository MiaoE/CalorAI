# CalorAI: The AI Calorie Estimator

![Logo](items/logo_light.png)

## Introduction

Obesity is a growing global health crisis, with over 38% of the population classified as overweight or obese, contributing to numerous chronic conditions. Accurate tracking of caloric intake is essential for promoting healthier eating habits, yet existing diet apps often rely on manual input, which can be tedious and error-prone.

**CalorAI** aims to simplify this process using deep learning. By analyzing images of meals, CalorAI automatically classifies food items and estimates their calorie content. Our system uses a CNN-based pipeline to handle both food recognition and portion estimation. This project serves as a foundation for more advanced diet-tracking solutions and offers an open-source implementation for further research and real-world use. Such a concept could be a great starting point for people who are not sure how to build healthier eating habits. 

---

## Data

Although many datasets map food images to categories and categories to calories, few offer direct mappings from food images to calorie values. This gap makes it difficult for models to learn **portion estimation** directly from images. To overcome this, we built our own dataset.

### Data Collection

We collected cell phone images in varied environments to simulate real-world usage. Images were taken on 10-inch plates with good contrast from the background and consistent framing (1:1 ratio). Each food item was weighed using a kitchen scale, and total calories were computed using a calorie-per-gram dataset. To ensure variety, food was photographed in different orientations, quantities, and combinations.

![Sample Data](items/example_data.png)

Our dataset contains 1093 labeled images across multiple food types.  

### Data Pre-processing

Images were resized to 400×400 pixels, converted to tensors, and normalized. Labels were converted to multi-label one-hot encodings. Each image is associated with the presence of specific food items, enabling multi-label classification and calorie estimation.

### Dataset Split by Food Category

| Food Category    | Total | Train       | Validation  | Test       |
|------------------|-------|-------------|-------------|------------|
| Pineapple        | 188   | 151 (80%)   | 18 (10%)    | 19 (10%)   |
| Blueberries      | 182   | 146 (80%)   | 20 (11%)    | 16 (9%)    |
| Strawberries     | 166   | 135 (81%)   | 16 (10%)    | 15 (9%)    |
| Chicken Breast   | 159   | 129 (81%)   | 15 (9%)     | 15 (9%)    |
| Cantaloupe       | 156   | 126 (81%)   | 16 (10%)    | 14 (9%)    |
| Egg              | 102   | 84 (82%)    | 9 (9%)      | 9 (9%)     |
| Bread            | 98    | 77 (79%)    | 10 (10%)    | 11 (11%)   |
| Grapes           | 93    | 74 (80%)    | 10 (11%)    | 9 (10%)    |
| Cherry Tomato    | 91    | 75 (82%)    | 8 (9%)      | 8 (9%)     |
| Mushrooms        | 90    | 71 (79%)    | 11 (12%)    | 8 (9%)     |
| Jujube           | 86    | 65 (76%)    | 10 (12%)    | 11 (13%)   |
| Broccoli         | 83    | 70 (84%)    | 8 (10%)     | 5 (6%)     |
| Honeydew         | 81    | 64 (79%)    | 9 (11%)     | 8 (10%)    |
| Cauliflower      | 80    | 65 (81%)    | 10 (13%)    | 5 (6%)     |
| Raisins          | 75    | 61 (81%)    | 6 (8%)      | 8 (11%)    |
| Sweet Potato     | 65    | 52 (80%)    | 7 (11%)     | 6 (9%)     |
| Garlic           | 64    | 51 (80%)    | 5 (8%)      | 8 (13%)    |
| Apple            | 51    | 39 (76%)    | 7 (14%)     | 5 (10%)    |
| Carrot           | 50    | 40 (80%)    | 5 (10%)     | 5 (10%)    |
| Clementine       | 41    | 31 (76%)    | 5 (12%)     | 5 (12%)    |
| Pear             | 33    | 27 (82%)    | 2 (6%)      | 4 (12%)    |
| Chives           | 30    | 23 (77%)    | 4 (13%)     | 3 (10%)    |
| Orange           | 23    | 20 (87%)    | 2 (9%)      | 1 (4%)     |
| Banana           | 21    | 16 (76%)    | 3 (14%)     | 2 (10%)    |
| Potato           | 20    | 18 (90%)    | 0 (0%)      | 2 (10%)    |
| Onion            | 14    | 11 (79%)    | 2 (14%)     | 1 (7%)     |
| **Total**        | 2432  | 1949 (80%)  | 270 (11%)   | 213 (9%)   |


### Dataset Split by Number of Food Types

| Number of Food Types | Total | Train       | Validation  | Test       |
|----------------------|-------|-------------|-------------|------------|
| 1                    | 587   | 467 (80%)   | 56 (10%)    | 64 (11%)   |
| 2                    | 212   | 172 (81%)   | 21 (10%)    | 19 (9%)    |
| 3                    | 147   | 120 (82%)   | 16 (11%)    | 11 (7%)    |
| 4                    | 82    | 63 (77%)    | 10 (12%)    | 9 (11%)    |
| 5+                   | 65    | 53 (82%)    | 6 (9%)      | 6 (9%)     |
| **Total**            | 1093  | 895 (82%)   | 109 (10%)   | 109 (10%)  |

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

---

## Guide to Codebase

