"""
data_loader.py
UCI Optical Recognition of Handwritten Digits
----------------------------------------------
Normalization  : StandardScaler
Augmentation   : rotation, translation, noise, thick-stroke simulation
Class balancing: digits 1 and 5 oversampled to match majority class count
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import rotate, gaussian_filter, shift


def _thicken(img_8x8: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """
    Simulate a thick drawn stroke by applying a dilation-like blur.
    Randomly choose sigma in [0.8, 1.4] to vary thickness.
    This teaches the model to recognise digits 1 and 5 even when
    drawn with a thick brush on the canvas.
    """
    sigma = rng.uniform(0.8, 1.4)
    thick = gaussian_filter(img_8x8, sigma=sigma)
    # Re-scale to original max so pixel range stays 0-16
    if thick.max() > 0:
        thick = thick / thick.max() * img_8x8.max()
    return thick


def augment_data(X_raw: np.ndarray, y: np.ndarray,
                 copies: int = 3, random_state: int = 0) -> tuple:
    """
    Augment raw (0-16) digit images.

    Per copy, each image gets:
    - Random rotation      : -15 to +15 degrees
    - Random translation   : -1 to +1 pixel (shift)
    - Thick-stroke blur    : 50% chance (simulates canvas drawing)
    - Gaussian noise       : std = 0.4

    Returns original + augmented concatenated.
    """
    rng   = np.random.RandomState(random_state)
    imgs  = X_raw.reshape(-1, 8, 8)
    X_out = [X_raw]
    y_out = [y]

    for _ in range(copies):
        batch = []
        for img in imgs:
            # Rotation
            aug = rotate(img, rng.uniform(-15, 15),
                         reshape=False, mode="nearest")
            # Translation (shift by up to 1 pixel in each axis)
            dr = rng.uniform(-1.0, 1.0)
            dc = rng.uniform(-1.0, 1.0)
            aug = shift(aug, [dr, dc], mode="nearest")
            # Thick-stroke simulation
            if rng.rand() > 0.5:
                aug = _thicken(aug, rng)
            # Noise
            aug = aug + rng.randn(8, 8) * 0.4
            aug = np.clip(aug, 0, 16)
            batch.append(aug.ravel())
        X_out.append(np.array(batch))
        y_out.append(y)

    return np.vstack(X_out), np.concatenate(y_out)


def balance_classes(X_raw: np.ndarray, y: np.ndarray,
                    focus_digits: tuple = (1, 5),
                    random_state: int = 0) -> tuple:
    """
    Oversample focus_digits so their count matches the majority class.
    Applied BEFORE augmentation so augmented copies are also balanced.
    """
    rng        = np.random.RandomState(random_state)
    counts     = np.bincount(y)
    target_n   = counts.max()
    X_list     = [X_raw]
    y_list     = [y]

    for d in focus_digits:
        idx     = np.where(y == d)[0]
        current = len(idx)
        if current < target_n:
            extra = target_n - current
            chosen = rng.choice(idx, size=extra, replace=True)
            X_list.append(X_raw[chosen])
            y_list.append(np.full(extra, d))

    return np.vstack(X_list), np.concatenate(y_list)


def load_uci_digits(test_size: float = 0.2, random_state: int = 42,
                    augment: bool = False, aug_copies: int = 3,
                    balance: bool = False):
    """
    Load, optionally balance + augment, scale, and split UCI Digits.

    Returns
    -------
    X_train  : (n_train, 64)  StandardScaler-transformed
    X_test   : (n_test,  64)  StandardScaler-transformed
    y_train  : (n_train,)     int
    y_test   : (n_test,)      int
    scaler   : fitted StandardScaler
    X_raw    : (1797, 64)     original 0-16 (for display)
    y_raw    : (1797,)        int
    images   : (1797, 8, 8)   0-16 (for display)
    """
    digits = load_digits()
    X_raw  = digits.data.astype(np.float64)
    y_raw  = digits.target
    images = digits.images

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y_raw,
        test_size=test_size,
        random_state=random_state,
        stratify=y_raw,
    )

    if balance:
        X_train_raw, y_train = balance_classes(
            X_train_raw, y_train, focus_digits=(1, 5),
            random_state=random_state
        )

    if augment:
        X_train_raw, y_train = augment_data(
            X_train_raw, y_train,
            copies=aug_copies,
            random_state=random_state,
        )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)

    return X_train, X_test, y_train, y_test, scaler, X_raw, y_raw, images


def get_class_samples(images, y_raw, n_per_class: int = 5):
    samples = {}
    for digit in range(10):
        idx = np.where(y_raw == digit)[0][:n_per_class]
        samples[digit] = [images[i] for i in idx]
    return samples
