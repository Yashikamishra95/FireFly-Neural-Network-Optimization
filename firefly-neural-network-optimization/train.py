"""
train.py  —  CLI entry point for Firefly Neural Network training.
Uses the SAME pipeline as streamlit_app.py via src/train_model.py.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.train_model import train_model
from src.model_io    import save_model

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Config (mirrors Streamlit sidebar defaults) ───────────────────────────────
config = {
    "population_size": 40,
    "max_iterations":  100,
    "alpha":           0.5,
    "beta0":           1.0,
    "gamma":           0.1,
    "use_augment":     True,
    "use_balance":     True,
    "random_state":    42,
}

# ── Progress callback ─────────────────────────────────────────────────────────
def on_progress(it, total, best, avg=None):
    if it % 10 == 0 or it == total:
        avg_str = f"  Avg={avg*100:.1f}%" if avg is not None else ""
        print(f"  Iter {it:>4}/{total}  Best={best:.4f} ({best*100:.1f}%){avg_str}")

# ── Run ───────────────────────────────────────────────────────────────────────
print("=" * 55)
print("Firefly Neural Network — UCI Digits")
print("Architecture : 64 -> 64 (ReLU) -> 32 (ReLU) -> 10 (Softmax)")
print("Pipeline     : MLP warm-start (64,32) -> Firefly refinement")
print("Fitness      : 0.7 * accuracy + 0.3 * (1 - norm_loss)")
print("Normalization: StandardScaler")
print("=" * 55)

print("\nStep 1/2: MLP warm-start (64,32) via backpropagation...")
print("Step 2/2: Firefly optimization running...\n")

results = train_model(config, progress_callback=on_progress)

nn     = results["nn"]
scaler = results["scaler"]
X_test = results["X_test"]
y_test = results["y_test"]

print(f"\n{'='*55}")
print(f"MLP  — Train: {results['mlp_train_acc']:.2f}%  Test: {results['mlp_test_acc']:.2f}%")
print(f"Firefly — Train: {results['train_acc']:.2f}%  Test: {results['test_acc']:.2f}%")
print(f"{'='*55}")

# Debug: verify predictions
y_test_pred = nn.predict(X_test)
print("Sample predictions:", y_test_pred[:10])
print("Sample true labels:", y_test[:10])
print("Unique predictions:", np.unique(y_test_pred))
print("Unique true labels:", np.unique(y_test))

# ── Save model ────────────────────────────────────────────────────────────────
save_model(nn, scaler, results["meta"])

# ── Append to results.txt ─────────────────────────────────────────────────────
with open(os.path.join(RESULTS_DIR, "results.txt"), "a", encoding="utf-8") as f:
    f.write("Dataset: UCI Optical Recognition of Handwritten Digits\n")
    f.write(f"Arch: 64-64-32-10 | Aug: {config['use_augment']} | Balance: {config['use_balance']}\n")
    f.write(f"MLP(64,32) warm-start: Train {results['mlp_train_acc']:.1f}%  Test {results['mlp_test_acc']:.1f}%\n")
    f.write(f"Pop={config['population_size']} Iter={config['max_iterations']} "
            f"a={config['alpha']} b0={config['beta0']} g={config['gamma']}\n")
    f.write(f"Firefly: Train {results['train_acc']:.2f}%  Test {results['test_acc']:.2f}%\n")
    f.write("-" * 55 + "\n")

# ── Confusion matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix — Test Set")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=120)
plt.show()

# ── Convergence curve ─────────────────────────────────────────────────────────
best_curve = results["convergence_best"]
avg_curve  = results["convergence_avg"]
plt.figure(figsize=(10, 5))
plt.plot(best_curve, linewidth=2,   color="#FF4B4B", label="Best Fitness")
plt.plot(avg_curve,  linewidth=1.5, color="#FFD700", linestyle="--", label="Avg Fitness")
plt.xlabel("Iteration")
plt.ylabel("Fitness")
plt.title("Firefly Algorithm Convergence")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "convergence_curve.png"), dpi=120)
plt.show()
print(f"\nResults saved to {RESULTS_DIR}/")

# ── Sample predictions grid ───────────────────────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle("Sample Test Predictions", fontsize=13)
for i in range(10):
    ax = axes[i // 5][i % 5]
    display = scaler.inverse_transform(X_test[i:i+1])[0].reshape(8, 8)
    ax.imshow(display, cmap="gray", vmin=0, vmax=16, interpolation="nearest")
    pred  = nn.predict(X_test[i:i+1])[0]
    color = "green" if pred == y_test[i] else "red"
    ax.set_title(f"P:{pred}  T:{y_test[i]}", color=color, fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "digit_predictions.png"), dpi=120)
plt.show()
