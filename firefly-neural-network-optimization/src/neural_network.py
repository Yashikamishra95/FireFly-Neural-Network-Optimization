import numpy as np


class NeuralNetwork:
    """
    64 -> 64 (ReLU) -> 32 (ReLU) -> 10 (Softmax)
    Total weights: 64*64+64 + 64*32+32 + 32*10+10 = 6,570
    He initialization + L2 regularization.
    """

    INPUT   = 64
    HIDDEN1 = 64
    HIDDEN2 = 32
    OUTPUT  = 10

    def __init__(self, l2: float = 1e-4):
        self.l2 = l2
        self.W1 = np.random.randn(self.INPUT,   self.HIDDEN1) * np.sqrt(2.0 / self.INPUT)
        self.b1 = np.zeros((1, self.HIDDEN1))
        self.W2 = np.random.randn(self.HIDDEN1, self.HIDDEN2) * np.sqrt(2.0 / self.HIDDEN1)
        self.b2 = np.zeros((1, self.HIDDEN2))
        self.W3 = np.random.randn(self.HIDDEN2, self.OUTPUT)  * np.sqrt(2.0 / self.HIDDEN2)
        self.b3 = np.zeros((1, self.OUTPUT))

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0.0, self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = np.maximum(0.0, self.z2)
        self.z3 = self.a2 @ self.W3 + self.b3
        e       = np.exp(self.z3 - self.z3.max(axis=1, keepdims=True))
        self.a3 = e / e.sum(axis=1, keepdims=True)
        return self.a3

    def compute_loss(self, y_true, y_pred):
        eps = 1e-10
        p   = np.clip(y_pred, eps, 1 - eps)
        if y_true.ndim == 1:
            ce = -np.log(p[range(len(y_true)), y_true]).mean()
        else:
            ce = -(y_true * np.log(p)).sum(axis=1).mean()
        l2 = self.l2 * (np.sum(self.W1**2) + np.sum(self.W2**2) + np.sum(self.W3**2))
        return ce + l2

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def get_weights(self) -> np.ndarray:
        return np.concatenate([
            self.W1.ravel(), self.b1.ravel(),
            self.W2.ravel(), self.b2.ravel(),
            self.W3.ravel(), self.b3.ravel(),
        ])

    def set_weights(self, w: np.ndarray):
        i = 0
        def _take(rows, cols):
            nonlocal i
            n   = rows * cols
            out = w[i:i+n].reshape(rows, cols); i += n
            return out
        def _take_bias(n):
            nonlocal i
            out = w[i:i+n].reshape(1, n); i += n
            return out
        self.W1 = _take(self.INPUT,   self.HIDDEN1)
        self.b1 = _take_bias(self.HIDDEN1)
        self.W2 = _take(self.HIDDEN1, self.HIDDEN2)
        self.b2 = _take_bias(self.HIDDEN2)
        self.W3 = _take(self.HIDDEN2, self.OUTPUT)
        self.b3 = _take_bias(self.OUTPUT)

    @property
    def n_weights(self):
        return (self.INPUT * self.HIDDEN1 + self.HIDDEN1 +
                self.HIDDEN1 * self.HIDDEN2 + self.HIDDEN2 +
                self.HIDDEN2 * self.OUTPUT + self.OUTPUT)
