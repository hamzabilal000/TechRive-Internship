"""
Week 2 - Task 1: ANN (Multi-Layer Perceptron) on an image dataset.

Uses scikit-learn's `MLPClassifier` (a fully-connected feed-forward neural
network trained with backpropagation) on the 8x8 handwritten-digits image
dataset. PyTorch/TensorFlow are not installed in this environment, so we use
the sklearn MLP -- the task explicitly allows this fallback.

Run:
    python 01_ann_mlp_digits.py
"""

from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def main() -> None:
    digits = load_digits()
    X, y = digits.data, digits.target  # X: (1797, 64) flattened 8x8 images
    print(f"Digits: {X.shape[0]} images, each 8x8={X.shape[1]} pixels, "
          f"{len(set(y))} classes (0-9)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Scale pixel intensities so the optimiser converges well.
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Two hidden layers (64 -> 32) with ReLU activations, Adam optimiser.
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        max_iter=400,
        random_state=42,
    )
    mlp.fit(X_train_s, y_train)

    preds = mlp.predict(X_test_s)
    acc = accuracy_score(y_test, preds)
    print(f"\nNetwork architecture: 64 -> {mlp.hidden_layer_sizes} -> 10")
    print(f"Training iterations run: {mlp.n_iter_}")
    print(f"Final training loss     : {mlp.loss_:.4f}")
    print(f"Test accuracy           : {acc:.3f}")
    print("\nPer-class report:")
    print(classification_report(y_test, preds, digits=3))


if __name__ == "__main__":
    main()
