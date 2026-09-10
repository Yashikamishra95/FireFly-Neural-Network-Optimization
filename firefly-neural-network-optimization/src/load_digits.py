import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load UCI Digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Normalize features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Split dataset (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y, test_size=0.2, random_state=42
)

# Print information
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")
print(f"Number of classes: {len(np.unique(y))}")
