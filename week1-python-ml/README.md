# Week 1 — Python, NumPy/Pandas & Classical ML

Foundations week: array/data manipulation with NumPy and pandas, then classical
supervised learning (classification and regression) with scikit-learn, always
comparing a simple linear model against a tree ensemble.

## What was done

### 1. NumPy & Pandas — `01_numpy_pandas.py`
- **NumPy array ops**: creation, reshaping, vectorised element-wise math, boolean
  masking, axis-wise aggregation, and a matrix–vector product.
- **Synthetic dataset**: a 20-row "employees" table built with a seeded random
  generator (department, age, salary, experience).
- **Data cleaning**: injected missing values, detected them with `isna().sum()`,
  and imputed — `salary` with the **median** (robust to outliers), `age` with the
  **mean**.
- **groupby analysis**: per-department headcount, average age/salary, and max
  experience, sorted by average salary.

### 2. Classification — `02_classification_iris.py`
- Dataset: **Iris** (150 samples, 4 features, 3 classes).
- Models: **Logistic Regression** (scaled via a pipeline) vs **Random Forest**.
- Evaluation: held-out test accuracy + macro-F1, and 5-fold cross-validation.
- Random Forest feature importances show `petal length`/`petal width` dominate.

### 3. Regression — `03_regression_diabetes.py`
- Dataset: **Diabetes** (442 samples, 10 features).
- Models: **Linear Regression** vs **Gradient Boosting**.
- Evaluation: R², RMSE, MAE on a test split + 5-fold cross-validated R².
- Linear Regression's coefficients give a directly interpretable readout.

## Representative results

| Task | Models | Outcome |
|------|--------|---------|
| Iris classification | LogReg vs RandomForest | Both ~0.91 test acc; RF best by CV (0.967) |
| Diabetes regression | Linear vs GradientBoosting | Close on the test split (R²≈0.45–0.46); Linear best by CV R² (0.482) |

The regression result is a good lesson: a fancier model (gradient boosting) does
**not** automatically win — on this small, mostly-linear dataset, plain linear
regression generalises better under cross-validation.

## Tools used
`Python 3.11`, `numpy`, `pandas`, `scikit-learn`.

## Key learnings
- Vectorised NumPy operations replace Python loops and are both faster and clearer.
- Missing-value strategy matters: median vs mean imputation is a deliberate choice
  driven by the column's distribution.
- `groupby(...).agg(...)` is the workhorse for split-apply-combine analysis.
- Always compare a simple baseline to a complex model, and judge with
  cross-validation — the higher-capacity model is not guaranteed to generalise
  better, especially on small datasets.
- Pipelines (`make_pipeline(StandardScaler(), ...)`) keep preprocessing bound to
  the model so scaling is learned on train folds only and there's no leakage.

## How to run
```bash
pip install numpy pandas scikit-learn
python 01_numpy_pandas.py
python 02_classification_iris.py
python 03_regression_diabetes.py
```

---
AI assistance (Claude) was used for debugging, explaining concepts, and code review while completing this week's tasks.
