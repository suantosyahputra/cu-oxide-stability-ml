# Machine Learning-Based Stability Screening of Cu-Containing Oxides

## Overview

This project investigates whether composition-derived descriptors can be used to predict the thermodynamic stability of Cu-containing oxides using machine learning.

Data were retrieved from the Materials Project database and featurized using Magpie composition descriptors from `matminer`.

Two tasks were explored:

* Regression of energy above hull
* Classification of materials as stable vs unstable

The final approach focuses on classification, which proved significantly more effective.

---

## Motivation

In materials discovery, **energy above hull (Eₕᵤₗₗ)** is a key metric for thermodynamic stability. However, predicting it accurately is challenging because it depends on crystal structure and competing phases.

This project explores how far one can go using only **composition-based features**, which are fast to compute and useful for large-scale screening.

---

## Data

The dataset was collected using the Materials Project API and includes:

* Binary systems: Cu–O
* Ternary systems: Cu–O–X

For each material:

* Composition (from structure)
* Energy above hull
* Chemical system label

Total dataset size:

* ~1000 materials
* Stable: 427
* Unstable: 570

---

## Feature Engineering

Composition features were generated using Magpie descriptors:

* Atomic properties (atomic number, weight, radius)
* Electronic properties (valence, unfilled orbitals)
* Thermodynamic proxies (melting temperature)
* Statistical aggregations (mean, range, std, etc.)

---

## Methodology

### Target Definition

A material is labeled as **stable** if:

```python
energy_above_hull < 0.05
```

---

### Models Compared

* Logistic Regression
* Gaussian Naive Bayes
* K-Nearest Neighbors
* Support Vector Machine
* Neural Network (MLP)
* Random Forest
* Gradient Boosting
* XGBoost

---

## Results

### Model Performance

| Model              | Accuracy  | F1-score  | ROC-AUC   |
| ------------------ | --------- | --------- | --------- |
| GradientBoosting   | **0.785** | 0.733     | **0.861** |
| XGBoost            | 0.775     | **0.737** | 0.853     |
| RandomForest       | 0.770     | 0.733     | 0.856     |
| NeuralNetwork      | 0.715     | 0.628     | 0.772     |
| KNN                | 0.680     | 0.632     | 0.748     |
| LogisticRegression | 0.685     | 0.583     | 0.760     |
| GaussianNB         | 0.600     | 0.583     | 0.676     |
| SVM                | 0.655     | 0.511     | 0.740     |

---

## Key Findings

* Tree-based ensemble models (Random Forest, Gradient Boosting, XGBoost) significantly outperform all other model families.
* Neural networks do not provide additional benefit for this dataset due to its limited size and tabular structure.
* Composition-based features are sufficient for **stability classification**, but not for precise prediction of energy above hull.
* Nonlinear interactions between elemental properties are critical for stability prediction.

---

## Feature Importance Insights

Important predictors include:

* Electronegativity
* Covalent radius
* Atomic weight
* Melting temperature
* Valence electron configuration

These features reflect key physical factors:

* Bonding strength
* Atomic size compatibility
* Electronic structure

---

## Outputs

The project generates:

* `classification_metrics.csv` → model comparison
* `roc_curve_comparison.png` → ROC curves
* Confusion matrices for all models
* Feature importance for tree-based models
* Clean dataset CSV

---

## Conclusion

This project demonstrates that:

> Composition-based machine learning models can effectively screen stable materials, even without structural information.

However, precise thermodynamic prediction requires additional structural descriptors.

---

## Tech Stack

* Python
* pandas
* scikit-learn
* xgboost
* matminer
* pymatgen
* Materials Project API

---

## Future Work

* Incorporate structural features (e.g., coordination, crystal graphs)
* Apply graph neural networks (CGCNN, MEGNet)
* Extend to other material systems (Ni–O, Fe–O, multi-component oxides)

---

## Author

Suanto Syahputra
PhD in Electrochemistry → transitioning into Data Science & AI for Materials
