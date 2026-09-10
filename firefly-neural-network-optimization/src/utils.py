"""
utils.py
Preprocessing + all matplotlib chart builders for the Streamlit app.

Normalization contract (must match data_loader.py):
    Training : StandardScaler fitted on X_train (zero-mean, unit-variance)
    Inference: same fitted scaler applied to canvas 8x8 vector

UCI digit polarity (verified):
    Background = 0 (dark), digit strokes = 1-16 (bright)
    Canvas draws white strokes on black bg  ->  same polarity, NO inversion.

Key fix for 1/5 misclassification:
    Canvas strokes are ~20px wide on 280x280.  After crop+resize to 8x8
    they become proportionally much thicker than UCI training images
    (~2-3 pixels wide).  We apply morphological erosion after resize to
    thin the stroke back to UCI proportions before feeding the model.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageOps, ImageFilter
from sklearn.metrics import confusion_matrix
from scipy.ndimage import binary_erosion, label as nd_label

# ── Dark theme palette ────────────────────────────────────────────────────────
BG   = "#0E1117"
CARD = "#1E2130"


def _dark(fig, *axes):
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(CARD)
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")


# ── Stroke thinning ───────────────────────────────────────────────────────────

def _thin_stroke(arr_8x8: np.ndarray, threshold: float = 6.0) -> np.ndarray:
    """
    Thin over-thick strokes to match UCI training image proportions.

    UCI digits have strokes ~2-3 pixels wide on an 8x8 grid.
    Canvas drawings, after resize, often produce 4-6 pixel wide strokes.
    One round of binary erosion on the thresholded mask removes the outer
    pixel ring, reducing stroke width without destroying the digit shape.

    Only applied when the stroke is detectably thicker than UCI average
    (more than 35 non-zero pixels out of 64).
    """
    binary = arr_8x8 > threshold
    nonzero = binary.sum()

    # UCI average nonzero pixels: ~30 for digit 1, ~34 for digit 5
    # If drawn digit has >38 nonzero pixels, it's too thick -> erode once
    if nonzero > 38:
        eroded = binary_erosion(binary, structure=np.ones((2, 2)))
        # Only keep erosion if it didn't destroy the digit (min 10 pixels)
        if eroded.sum() >= 10:
            # Blend: keep original intensity where eroded mask is True
            result = arr_8x8 * eroded.astype(np.float64)
            return result
    return arr_8x8


# ── Aspect-ratio-preserving crop ──────────────────────────────────────────────

def _crop_preserve_aspect(arr: np.ndarray, threshold: float = 10.0) -> np.ndarray:
    """
    Crop to bounding box with padding, then pad to square before resize.

    Without this, a tall narrow digit (like 1) gets horizontally stretched
    when resized to 8x8, making it look wider and more like a 4.
    """
    mask = arr > threshold
    if not mask.any():
        return arr

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    r0, r1 = rows[0], rows[-1]
    c0, c1 = cols[0], cols[-1]

    h_box = r1 - r0 + 1
    w_box = c1 - c0 + 1

    # 15% padding
    pad = max(1, int(max(h_box, w_box) * 0.15))
    H, W = arr.shape
    r0 = max(0, r0 - pad);  r1 = min(H - 1, r1 + pad)
    c0 = max(0, c0 - pad);  c1 = min(W - 1, c1 + pad)

    cropped = arr[r0:r1 + 1, c0:c1 + 1]
    ch, cw  = cropped.shape

    # Pad shorter dimension to make it square (preserves aspect ratio)
    if ch != cw:
        side = max(ch, cw)
        square = np.zeros((side, side), dtype=np.float64)
        r_off  = (side - ch) // 2
        c_off  = (side - cw) // 2
        square[r_off:r_off + ch, c_off:c_off + cw] = cropped
        return square
    return cropped


# ── Canvas -> UCI vector ──────────────────────────────────────────────────────

def canvas_to_uci_vector(img_data: np.ndarray, scaler) -> tuple:
    """
    Convert RGBA canvas (H x W x 4) -> StandardScaler-normalised 64-d vector.

    Pipeline
    --------
    1. RGBA -> greyscale
    2. Remove isolated noise pixels (connected-component filter)
    3. Aspect-ratio-preserving crop + square padding
    4. Gaussian blur radius=1  (smooth jagged strokes)
    5. Resize to 8x8 LANCZOS
    6. Scale to 0-16  (UCI native range)
    7. Stroke thinning via erosion  (match UCI stroke width)
    8. scaler.transform()  (same StandardScaler used in training)

    Returns
    -------
    steps      : dict  'original' | 'cropped' | 'thinned' | 'final_8x8'
    raw_8x8    : np.ndarray (8, 8)   0-16 range, for display
    scaled_vec : np.ndarray (1, 64)  scaler-transformed, for the model
    """
    # Step 1: RGBA -> greyscale
    img      = Image.fromarray(img_data.astype("uint8"), "RGBA").convert("L")
    arr_orig = np.array(img, dtype=np.float64)

    # Step 2: Remove isolated noise (keep only the largest connected component)
    binary_mask = arr_orig > 10
    if binary_mask.any():
        labeled, n_comp = nd_label(binary_mask)
        if n_comp > 1:
            sizes = [(labeled == i).sum() for i in range(1, n_comp + 1)]
            largest = np.argmax(sizes) + 1
            arr_orig = arr_orig * (labeled == largest)

    arr_orig_display = (
        arr_orig / arr_orig.max() * 16.0 if arr_orig.max() > 0 else arr_orig
    )

    # Step 3: Aspect-ratio-preserving crop + square padding
    cropped = _crop_preserve_aspect(arr_orig, threshold=10.0)
    arr_cropped_display = (
        cropped / cropped.max() * 16.0 if cropped.max() > 0 else cropped
    )

    # Step 4: Gaussian blur
    img_crop = Image.fromarray(cropped.astype(np.uint8))
    img_crop = img_crop.filter(ImageFilter.GaussianBlur(radius=1))

    # Step 5: Resize to 8x8
    img_8x8 = img_crop.resize((8, 8), Image.LANCZOS)
    arr_8x8 = np.array(img_8x8, dtype=np.float64)

    # Step 6: Scale to 0-16
    if arr_8x8.max() > 0:
        arr_8x8 = arr_8x8 / arr_8x8.max() * 16.0

    # Step 7: Stroke thinning
    thinned = _thin_stroke(arr_8x8)
    # Re-scale after thinning so max is still 16
    if thinned.max() > 0:
        thinned = thinned / thinned.max() * 16.0
    raw_8x8 = thinned.copy()

    # Step 8: StandardScaler
    scaled_vec = scaler.transform(raw_8x8.flatten().reshape(1, -1))

    steps = {
        "original":  arr_orig_display,
        "cropped":   arr_cropped_display,
        "thinned":   thinned,
        "final_8x8": raw_8x8,
    }
    return steps, raw_8x8, scaled_vec


def one_hot(y: np.ndarray, n: int = 10) -> np.ndarray:
    oh = np.zeros((len(y), n))
    oh[np.arange(len(y)), y] = 1
    return oh


# ── Chart: confidence bar ─────────────────────────────────────────────────────

def make_confidence_bar(probs: np.ndarray):
    p    = probs.flatten()
    best = int(np.argmax(p))
    colors = ["#FF4B4B" if i == best else "#4B8BFF" for i in range(10)]

    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.bar(range(10), p, color=colors, edgecolor="#333", linewidth=0.5)
    ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2, color="white")
    ax.set_xticks(range(10))
    ax.set_xticklabels([str(i) for i in range(10)])
    ax.set_xlabel("Digit Class")
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Confidence Distribution")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.2)
    _dark(fig, ax)
    plt.tight_layout()
    return fig


# ── Chart: 8x8 digit preview ─────────────────────────────────────────────────

def make_preprocessing_steps_fig(steps: dict):
    """
    Show four preprocessing stages: original | cropped | thinned | final 8x8
    """
    keys   = ["original", "cropped", "thinned", "final_8x8"]
    labels = ["1. Original", "2. Cropped", "3. Thinned", "4. Final 8x8"]
    # Gracefully handle old 3-key dicts
    keys   = [k for k in keys if k in steps]
    labels = labels[:len(keys)]

    fig, axes = plt.subplots(1, len(keys), figsize=(2.5 * len(keys), 2.5))
    if len(keys) == 1:
        axes = [axes]
    fig.suptitle("Preprocessing Steps", fontsize=10, color="white")
    fig.patch.set_facecolor(BG)
    for ax, key, label in zip(axes, keys, labels):
        ax.imshow(steps[key], cmap="gray", vmin=0, vmax=16, interpolation="nearest")
        ax.set_title(label, fontsize=8, color="white")
        ax.axis("off")
        ax.set_facecolor(CARD)
    plt.tight_layout()
    return fig


def make_digit_preview(raw_8x8: np.ndarray):
    # UCI polarity: bright stroke on dark bg → use cmap='gray' (not gray_r)
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.imshow(raw_8x8, cmap="gray", vmin=0, vmax=16,
              interpolation="nearest")
    ax.set_title("Model Input (8x8)", fontsize=9)
    ax.axis("off")
    _dark(fig, ax)
    plt.tight_layout()
    return fig


# ── Chart: convergence curve ──────────────────────────────────────────────────

def make_convergence_fig(best_curve: list, avg_curve: list):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(best_curve, linewidth=2,   color="#FF4B4B", label="Best Fitness")
    if avg_curve:
        ax.plot(avg_curve, linewidth=1.5, color="#FFD700",
                linestyle="--", label="Avg Fitness")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fitness (Accuracy)")
    ax.set_title("Firefly Algorithm Convergence")
    ax.legend(facecolor=CARD, labelcolor="white", framealpha=0.8)
    ax.grid(True, alpha=0.2)
    _dark(fig, ax)
    plt.tight_layout()
    return fig


# ── Chart: confusion matrix ───────────────────────────────────────────────────

def make_confusion_fig(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        linewidths=0.4, linecolor="#333",
        annot_kws={"size": 8},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Test Set")
    _dark(fig, ax)
    plt.tight_layout()
    return fig


# ── Chart: sample test predictions grid ──────────────────────────────────────

def make_sample_predictions_fig(nn, X_test, y_test, scaler=None, n: int = 10):
    """
    Show n test images with predicted vs true label.
    If scaler is provided, inverse-transforms X_test for display.
    """
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle("Sample Test Predictions", fontsize=13, color="white")
    fig.patch.set_facecolor(BG)
    for i in range(n):
        ax = axes[i // 5][i % 5]
        if scaler is not None:
            display = scaler.inverse_transform(X_test[i:i+1])[0].reshape(8, 8)
        else:
            display = X_test[i].reshape(8, 8) * 16
        ax.imshow(display, cmap="gray", vmin=0, vmax=16, interpolation="nearest")
        pred  = nn.predict(X_test[i:i+1])[0]
        color = "#00FF88" if pred == y_test[i] else "#FF4B4B"
        ax.set_title(f"P:{pred}  T:{y_test[i]}", color=color, fontsize=9,
                     fontweight="bold")
        ax.axis("off")
        ax.set_facecolor(CARD)
    plt.tight_layout()
    return fig


# ── Chart: UCI dataset sample grid ───────────────────────────────────────────

def make_dataset_samples_fig(samples: dict):
    """
    Display 5 sample images per digit class (10 rows x 5 cols).
    `samples` = {digit: [list of (8,8) arrays in 0-16 range]}
    """
    fig, axes = plt.subplots(10, 5, figsize=(7, 14))
    fig.suptitle("UCI Digits Dataset — 5 Samples per Class",
                 fontsize=12, color="white", y=1.01)
    fig.patch.set_facecolor(BG)
    for digit in range(10):
        for col, img in enumerate(samples[digit]):
            ax = axes[digit][col]
            ax.imshow(img, cmap="gray_r", vmin=0, vmax=16,
                      interpolation="nearest")
            ax.axis("off")
            ax.set_facecolor(CARD)
            if col == 0:
                ax.set_ylabel(str(digit), color="white", fontsize=10,
                              rotation=0, labelpad=12, va="center")
    plt.tight_layout()
    return fig


# ── Chart: class distribution bar ────────────────────────────────────────────

def make_class_dist_fig(y_raw: np.ndarray):
    counts = np.bincount(y_raw)
    fig, ax = plt.subplots(figsize=(6, 3))
    # Highlight digits 1 and 5 in red
    colors = ["#FF4B4B" if i in (1, 5) else "#4B8BFF" for i in range(10)]
    ax.bar(range(10), counts, color=colors, edgecolor="#333")
    ax.bar_label(ax.containers[0], fontsize=8, color="white", padding=2)
    ax.set_xticks(range(10))
    ax.set_xlabel("Digit Class")
    ax.set_ylabel("Sample Count")
    ax.set_title("Class Distribution (red = focus digits 1 & 5)")
    ax.set_ylim(0, max(counts) * 1.15)
    ax.grid(axis="y", alpha=0.2)
    _dark(fig, ax)
    plt.tight_layout()
    return fig


def make_per_class_accuracy_fig(y_true: np.ndarray, y_pred: np.ndarray):
    """Bar chart of per-class accuracy, highlighting digits 1 and 5."""
    accs = []
    for d in range(10):
        mask = y_true == d
        accs.append(np.mean(y_pred[mask] == d) * 100 if mask.sum() > 0 else 0.0)

    colors = ["#FF4B4B" if i in (1, 5) else "#4B8BFF" for i in range(10)]
    fig, ax = plt.subplots(figsize=(7, 3))
    bars = ax.bar(range(10), accs, color=colors, edgecolor="#333")
    ax.bar_label(bars, fmt="%.0f%%", fontsize=7, padding=2, color="white")
    ax.set_xticks(range(10))
    ax.set_xlabel("Digit Class")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-Class Accuracy (red = digits 1 & 5)")
    ax.set_ylim(0, 115)
    ax.axhline(np.mean(accs), color="#FFD700", linestyle="--",
               linewidth=1.2, label=f"Mean {np.mean(accs):.1f}%")
    ax.legend(facecolor=CARD, labelcolor="white", fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    _dark(fig, ax)
    plt.tight_layout()
    return fig
