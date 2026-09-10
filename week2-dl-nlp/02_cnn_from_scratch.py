"""
Week 2 - Task 2: A basic Convolutional Neural Network from scratch in NumPy.

No deep-learning framework is installed, so this implements a small CNN
end-to-end with pure NumPy -- forward pass AND backpropagation -- and trains
it to classify the 8x8 handwritten-digits images.

Architecture:
    input 8x8x1
      -> Conv2D (8 filters, 3x3, valid)   -> 6x6x8
      -> ReLU
      -> MaxPool 2x2                       -> 3x3x8
      -> Flatten                           -> 72
      -> Dense (72 -> 10) + Softmax
    trained with cross-entropy loss and mini-batch SGD.

Run:
    python 02_cnn_from_scratch.py
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


# --------------------------------------------------------------------------- #
# Layer primitives
# --------------------------------------------------------------------------- #
def conv_forward(x, W, b):
    """Valid 3x3 convolution. x:(N,H,W_) W:(F,kh,kw) b:(F,) -> (N,F,oh,ow)."""
    n, h, w = x.shape
    f, kh, kw = W.shape
    oh, ow = h - kh + 1, w - kw + 1
    out = np.zeros((n, f, oh, ow))
    for i in range(oh):
        for j in range(ow):
            patch = x[:, i:i + kh, j:j + kw]           # (N, kh, kw)
            # einsum over the batch and each filter.
            out[:, :, i, j] = np.einsum("nij,fij->nf", patch, W) + b
    return out


def conv_backward(dout, x, W):
    """Gradients for the conv layer. Returns dW, db (dx not needed: first layer)."""
    n, h, w = x.shape
    f, kh, kw = W.shape
    _, _, oh, ow = dout.shape
    dW = np.zeros_like(W)
    db = dout.sum(axis=(0, 2, 3))
    for i in range(oh):
        for j in range(ow):
            patch = x[:, i:i + kh, j:j + kw]           # (N, kh, kw)
            dW += np.einsum("nf,nij->fij", dout[:, :, i, j], patch)
    return dW, db


def relu_forward(x):
    return np.maximum(0, x)


def relu_backward(dout, x):
    return dout * (x > 0)


def maxpool_forward(x, size=2):
    """2x2 max pooling. x:(N,F,H,W) -> (N,F,H//2,W//2). Returns out + argmax mask."""
    n, f, h, w = x.shape
    oh, ow = h // size, w // size
    out = np.zeros((n, f, oh, ow))
    mask = np.zeros_like(x)
    for i in range(oh):
        for j in range(ow):
            region = x[:, :, i * size:i * size + size, j * size:j * size + size]
            flat = region.reshape(n, f, -1)
            amax = flat.argmax(axis=2)
            out[:, :, i, j] = flat.max(axis=2)
            # Build the routing mask for the backward pass (record which
            # position held the max in each pooling window).
            ii = i * size + (amax // size)
            jj = j * size + (amax % size)
            for ni in range(n):
                for fi in range(f):
                    mask[ni, fi, ii[ni, fi], jj[ni, fi]] = 1
    return out, mask


def maxpool_backward(dout, mask, size=2):
    """Route gradients back to the positions that held the max."""
    n, f, oh, ow = dout.shape
    dx = np.zeros_like(mask)
    for i in range(oh):
        for j in range(ow):
            region = dx[:, :, i * size:i * size + size, j * size:j * size + size]
            m = mask[:, :, i * size:i * size + size, j * size:j * size + size]
            region += m * dout[:, :, i, j][:, :, None, None]
    return dx


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class TinyCNN:
    def __init__(self, n_filters=8, n_classes=10, seed=0):
        rng = np.random.default_rng(seed)
        # He-style initialisation.
        self.W1 = rng.normal(0, np.sqrt(2 / 9), size=(n_filters, 3, 3))
        self.b1 = np.zeros(n_filters)
        pooled = 3 * 3 * n_filters  # 6x6 -> pool -> 3x3, times filters
        self.W2 = rng.normal(0, np.sqrt(2 / pooled), size=(pooled, n_classes))
        self.b2 = np.zeros(n_classes)

    def forward(self, x):
        self.x = x
        self.c = conv_forward(x, self.W1, self.b1)
        self.r = relu_forward(self.c)
        self.p, self.mask = maxpool_forward(self.r)
        self.flat = self.p.reshape(x.shape[0], -1)
        self.scores = self.flat @ self.W2 + self.b2
        self.probs = softmax(self.scores)
        return self.probs

    def backward(self, y_onehot, lr):
        n = y_onehot.shape[0]
        dscores = (self.probs - y_onehot) / n            # softmax+CE gradient
        dW2 = self.flat.T @ dscores
        db2 = dscores.sum(axis=0)
        dflat = dscores @ self.W2.T
        dp = dflat.reshape(self.p.shape)
        dr = maxpool_backward(dp, self.mask)
        dc = relu_backward(dr, self.c)
        dW1, db1 = conv_backward(dc, self.x, self.W1)
        # SGD update.
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def predict(self, x):
        return self.forward(x).argmax(axis=1)


def main() -> None:
    digits = load_digits()
    X = digits.images / 16.0          # (1797, 8, 8) scaled to [0,1]
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"Train on {len(X_train)} images, test on {len(X_test)} (8x8 digits)")

    net = TinyCNN(n_filters=8, n_classes=10, seed=0)
    n_classes = 10
    onehot = np.eye(n_classes)

    epochs, batch = 15, 32
    lr = 0.08
    rng = np.random.default_rng(1)
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(X_train))
        for start in range(0, len(idx), batch):
            b = idx[start:start + batch]
            xb, yb = X_train[b], y_train[b]
            probs = net.forward(xb)
            net.backward(onehot[yb], lr)
        # Epoch metrics on the training set.
        tr_pred = net.predict(X_train)
        tr_acc = (tr_pred == y_train).mean()
        eps = 1e-9
        loss = -np.log(net.forward(X_train)[np.arange(len(y_train)), y_train]
                       + eps).mean()
        print(f"epoch {ep:2d} | train loss {loss:.3f} | train acc {tr_acc:.3f}")

    test_acc = (net.predict(X_test) == y_test).mean()
    print(f"\nFinal TEST accuracy: {test_acc:.3f}")


if __name__ == "__main__":
    main()
