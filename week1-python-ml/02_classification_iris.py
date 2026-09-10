"""
Week 1 - Task 2: Classification on the Iris dataset.

Compares:
  * Logistic Regression (linear model)
  * Random Forest      (bagged decision-tree ensemble)

Metrics: accuracy + macro F1 on a held-out test split, plus a
5-fold cross-validation score for a more stable comparison.

Run:
    python 02_classification_iris.py
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
    data = load_iris()
    X, y = data.data, data.target
    print(f"Iris: {X.shape[0]} samples, {X.shape[1]} features, "
          f"{len(data.target_names)} classes {list(map(str, data.target_names))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    models = {
        # LogReg benefits from feature scaling -> wrap in a pipeline.
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)
        ),
        # Random Forest is scale-invariant, so no scaler needed.
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42
        ),
    }

    print("\n" + "=" * 60)
    print(f"{'Model':<22}{'Test Acc':>10}{'Macro F1':>10}{'CV Acc':>12}")
    print("=" * 60)

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        cv = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        results[name] = (acc, f1, cv.mean())
        print(f"{name:<22}{acc:>10.3f}{f1:>10.3f}"
              f"{cv.mean():>10.3f}±{cv.std():.2f}")

    best = max(results, key=lambda k: results[k][2])
    print("=" * 60)
    print(f"Best by CV accuracy: {best}")

    # Feature importances from the Random Forest are a nice interpretability win.
    rf = models["Random Forest"]
    order = np.argsort(rf.feature_importances_)[::-1]
    print("\nRandom Forest feature importances:")
    for i in order:
        print(f"  {data.feature_names[i]:<20} {rf.feature_importances_[i]:.3f}")

    print("\nDetailed report for best model on the test split:")
    print(classification_report(
        y_test, models[best].predict(X_test),
        target_names=data.target_names
    ))


if __name__ == "__main__":
    main()
