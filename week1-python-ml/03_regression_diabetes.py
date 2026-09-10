"""
Week 1 - Task 3: Regression on the Diabetes dataset.

Compares:
  * Linear Regression        (simple linear baseline)
  * Gradient Boosting        (boosted decision-tree ensemble)

Metrics: R^2, RMSE and MAE on a held-out test split, plus 5-fold
cross-validated R^2.

Run:
    python 03_regression_diabetes.py
"""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split


def main() -> None:
    data = load_diabetes()
    X, y = data.data, data.target
    print(f"Diabetes: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target range: {y.min():.0f} .. {y.max():.0f} "
          f"(disease progression score)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42
        ),
    }

    print("\n" + "=" * 62)
    print(f"{'Model':<22}{'R2':>8}{'RMSE':>10}{'MAE':>10}{'CV R2':>12}")
    print("=" * 62)

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        cv = cross_val_score(model, X, y, cv=5, scoring="r2")
        results[name] = (r2, rmse, mae, cv.mean())
        print(f"{name:<22}{r2:>8.3f}{rmse:>10.1f}{mae:>10.1f}"
              f"{cv.mean():>10.3f}±{cv.std():.2f}")

    best = max(results, key=lambda k: results[k][3])
    print("=" * 62)
    print(f"Best by CV R2: {best}")

    # Linear-regression coefficients are directly interpretable.
    lr = models["Linear Regression"]
    order = np.argsort(np.abs(lr.coef_))[::-1]
    print("\nLinear Regression coefficients (by |magnitude|):")
    for i in order[:5]:
        print(f"  {data.feature_names[i]:<8} {lr.coef_[i]:>10.2f}")


if __name__ == "__main__":
    main()
