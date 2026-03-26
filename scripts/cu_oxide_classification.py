import os
import pandas as pd
import matplotlib.pyplot as plt

from mp_api.client import MPRester
from matminer.featurizers.composition import ElementProperty
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

systems_to_query = [
    (["Cu", "O"], 2),
    (["Cu", "O"], 3),
]

all_docs = []

with MPRester("Insert API KEY") as mpr:
    for elements, num_elements in systems_to_query:
        query_docs = mpr.summary.search(
            elements=elements,
            num_elements=num_elements,
            fields=["material_id", "formula_pretty", "structure", "energy_above_hull"]
        )
        all_docs.extend(query_docs)

# Remove duplicates
unique_docs = {}
for doc in all_docs:
    unique_docs[str(doc.material_id)] = doc

docs = [doc for doc in unique_docs.values() if doc.energy_above_hull is not None]

# Build dataset
compositions = [doc.structure.composition for doc in docs]
targets = [doc.energy_above_hull for doc in docs]
formulas = [doc.formula_pretty for doc in docs]
material_ids = [str(doc.material_id) for doc in docs]

chemical_systems = []
for comp in compositions:
    elements = sorted([el.symbol for el in comp.elements])
    chemical_systems.append("-".join(elements))

featurizer = ElementProperty.from_preset("magpie")
X = featurizer.featurize_many(compositions)
feature_names = featurizer.feature_labels()

df = pd.DataFrame(X, columns=feature_names)
df["material_id"] = material_ids
df["formula"] = formulas
df["system"] = chemical_systems
df["energy_above_hull"] = targets

df = df.dropna()

# Classification target
df["stable"] = (df["energy_above_hull"] < 0.05).astype(int)

print("Stable class distribution:")
print(df["stable"].value_counts())

X_data = df[feature_names]
y_data = df["stable"]

X_train, X_test, y_train, y_test = train_test_split(
    X_data,
    y_data,
    test_size=0.2,
    random_state=42,
    stratify=y_data
)

logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=5000))
])

knn_model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", KNeighborsClassifier(n_neighbors=5))
])

svm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(probability=True, random_state=42))
])

mlp_model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=2000,
        early_stopping=True,
        random_state=42
    ))
])


models = {
    "LogisticRegression": logistic_model,
    "GaussianNB": GaussianNB(),
    "KNN": knn_model,
    "SVM": svm_model,
    "NeuralNetwork": mlp_model,
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        eval_metric="logloss"
    )
}

metrics_list = []
trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    metrics_list.append({
        "model": name,
        "accuracy": acc,
        "f1": f1,
        "roc_auc": auc
    })

    print(f"\n{name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))
  

metrics_df = pd.DataFrame(metrics_list)
metrics_df = metrics_df.sort_values("f1", ascending=False)
metrics_df = metrics_df.round(4)
metrics_df.to_csv("outputs/classification_metrics.csv", index=False)
df.to_csv("data/cu_oxide_classification_dataset.csv", index=False)

print("\nSaved classification metrics to outputs/classification_metrics.csv")
print("Saved dataset to data/cu_oxide_classification_dataset.csv")
print(metrics_df)

# ROC curve comparison
plt.figure(figsize=(7, 6))

for name, model in trained_models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/roc_curve_comparison.png", dpi=200)
plt.close()

print("Saved ROC curve comparison to outputs/roc_curve_comparison.png")
# Confusion matrix plot for each model
for name, model in trained_models.items():
    y_pred = model.predict(X_test)

    plt.figure(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, cmap="Blues", colorbar=False
    )
    plt.title(f"{name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"outputs/{name}_confusion_matrix.png", dpi=200)
    plt.close()

print("Saved confusion matrix plots.")


# Feature importance for supported models
for name, model in trained_models.items():
    clf = model
    if isinstance(model, Pipeline):
        clf = model.named_steps.get("clf", model)

    if hasattr(clf, "feature_importances_"):
        importances = pd.DataFrame({
            "feature": feature_names,
            "importance": clf.feature_importances_
        }).sort_values("importance", ascending=False)

        importances.to_csv(f"outputs/{name}_feature_importance.csv", index=False)
        print(f"Saved feature importance to outputs/{name}_feature_importance.csv")

        print(f"\nTop 10 features for {name}:")
        print(importances.head(10))
