"""
streamlit_app.py
Firefly Neural Network Optimizer — Interactive Dashboard
UCI Optical Recognition of Handwritten Digits Dataset
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from streamlit_drawable_canvas import st_canvas

from src.data_loader   import load_uci_digits, get_class_samples, augment_data, balance_classes
from src.model         import NeuralNetwork
from src.firefly       import FireflyAlgorithm
from src.model_io      import save_model, load_model
from src.utils import (
    canvas_to_uci_vector,
    one_hot,
    make_confidence_bar,
    make_digit_preview,
    make_preprocessing_steps_fig,
    make_convergence_fig,
    make_confusion_fig,
    make_sample_predictions_fig,
    make_dataset_samples_fig,
    make_class_dist_fig,
    make_per_class_accuracy_fig,
)

# ── Constants ─────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
from src.neural_network import NeuralNetwork as _NN
N_WEIGHTS   = _NN().n_weights  # auto-derived from NeuralNetwork class

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Firefly NN Optimizer",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Auto-load saved model on cold start ───────────────────────────────────────
if "nn" not in st.session_state:
    nn_loaded, scaler_loaded, meta_loaded = load_model()
    if nn_loaded is not None:
        st.session_state.update({
            "nn":               nn_loaded,
            "scaler":           scaler_loaded,
            "meta":             meta_loaded,
            "model_loaded":     True,
            "convergence_best": meta_loaded.get("convergence_best", []),
            "convergence_avg":  meta_loaded.get("convergence_avg",  []),
            "train_acc":        meta_loaded.get("train_acc",  None),
            "test_acc":         meta_loaded.get("test_acc",   None),
        })
    else:
        st.session_state["model_loaded"] = False

# ── Dataset (cached — loaded once per session) ────────────────────────────────
@st.cache_data(show_spinner=False)
def get_data():
    # v4: 64-64-32-10 network, thick-stroke augmentation, class balancing
    return load_uci_digits()

@st.cache_data(show_spinner=False)
def get_samples():
    # v4: unpack 8 values
    _, _, _, _, _sc, X_raw, y_raw, images = load_uci_digits()
    return get_class_samples(images, y_raw, n_per_class=5), y_raw

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Firefly Parameters")

pop_size   = st.sidebar.slider("Population Size",  10, 100, 40,   5)
iterations = st.sidebar.slider("Iterations",        10, 300, 100, 10)
alpha      = st.sidebar.slider("Alpha  (a)",        0.1, 2.0, 0.5, 0.05)
beta0      = st.sidebar.slider("Beta0  (b0)",       0.1, 2.0, 1.0, 0.05)
gamma      = st.sidebar.slider("Gamma  (g)",        0.01, 1.0, 0.1, 0.01)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**a** — exploration randomness  \n"
    "**b0** — base attractiveness  \n"
    "**g** — light absorption coefficient"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset**")
st.sidebar.info(
    "UCI Optical Recognition  \n"
    "of Handwritten Digits  \n"
    "1,797 samples · 64 features  \n"
    "10 classes (0–9)  \n"
    "Normalization: StandardScaler"
)

if st.session_state.get("model_loaded"):
    m = st.session_state.get("meta", {})
    st.sidebar.markdown("---")
    st.sidebar.success("Trained model loaded")
    if m.get("train_acc"):
        st.sidebar.metric("Train Acc", f"{m['train_acc']:.1f}%")
        st.sidebar.metric("Test Acc",  f"{m['test_acc']:.1f}%")
else:
    st.sidebar.warning("No saved model — train first")

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🔥 Firefly Neural Network Optimizer")
st.caption(
    "UCI Optical Recognition of Handwritten Digits  ·  "
    "64 features (8x8 pixels / 16)  ·  "
    "Firefly Algorithm weight optimization"
)

tab_data, tab_train, tab_draw, tab_results = st.tabs([
    "📂 Dataset Explorer",
    "🚀 Train Model",
    "✏️ Draw & Predict",
    "📊 Results & Logs",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATASET EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.subheader("📂 UCI Optical Recognition of Handwritten Digits")

    X_train, X_test, y_train, y_test, scaler, X_raw, y_raw, images = get_data()
    samples, y_raw_cached = get_samples()

    # ── Dataset stats ─────────────────────────────────────────────────────────
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Total Samples",   "1,797")
    col_b.metric("Features",        "64  (8x8 pixels)")
    col_c.metric("Classes",         "10  (digits 0–9)")
    col_d.metric("Normalization",   "StandardScaler")

    st.markdown("---")

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("**Class Distribution**")
        st.pyplot(make_class_dist_fig(y_raw_cached))

        st.markdown("**Train / Test Split**")
        split_data = {
            "Split":   ["Train (80%)", "Test (20%)"],
            "Samples": [len(X_train),   len(X_test)],
        }
        for label, count in zip(split_data["Split"], split_data["Samples"]):
            st.markdown(f"- {label}: **{count}** samples")

        st.markdown("**Preprocessing Pipeline**")
        st.code(
            "from sklearn.datasets import load_digits\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "digits = load_digits()\n"
            "X = digits.data          # shape (1797, 64), range 0-16\n"
            "y = digits.target\n"
            "# 80/20 stratified split, then scale\n"
            "scaler  = StandardScaler()\n"
            "X_train = scaler.fit_transform(X_train_raw)\n"
            "X_test  = scaler.transform(X_test_raw)",
            language="python",
        )

    with col_right:
        st.markdown("**Sample Images — 5 per Digit Class**")
        st.pyplot(make_dataset_samples_fig(samples))

    # ── Raw feature peek ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Feature Vector Preview** — first training sample")
    sample_vec = X_train[0].reshape(1, -1)
    st.markdown(
        f"Shape: `{sample_vec.shape}` · "
        f"Min: `{sample_vec.min():.3f}` · "
        f"Max: `{sample_vec.max():.3f}` · "
        f"Mean: `{sample_vec.mean():.3f}`"
    )
    fig_prev, ax_prev = plt.subplots(1, 2, figsize=(6, 2.5))
    fig_prev.patch.set_facecolor("#0E1117")
    ax_prev[0].imshow(X_raw[0].reshape(8, 8),
                      cmap="gray", vmin=0, vmax=16, interpolation="nearest")
    ax_prev[0].set_title(f"Label: {y_train[0]}", color="white", fontsize=9)
    ax_prev[0].axis("off")
    ax_prev[0].set_facecolor("#1E2130")
    ax_prev[1].bar(range(64), X_train[0], color="#4B8BFF", linewidth=0)
    ax_prev[1].set_xlabel("Feature Index", color="white")
    ax_prev[1].set_ylabel("Norm. Value", color="white")
    ax_prev[1].set_title("64-d Feature Vector", color="white")
    ax_prev[1].tick_params(colors="white")
    ax_prev[1].set_facecolor("#1E2130")
    ax_prev[1].grid(axis="y", alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig_prev)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TRAIN MODEL
# ═══════════════════════════════════════════════════════════════════════════════
with tab_train:
    st.subheader("Train Neural Network with Firefly Algorithm")
    st.markdown(
        f"**Config:** Population `{pop_size}` | Iterations `{iterations}` | "
        f"a `{alpha}` | b0 `{beta0}` | g `{gamma}`"
    )
    st.markdown(
        "**Architecture:** 64 -> 64 (ReLU) -> 32 (ReLU) -> 10 (Softmax)  \n"
        "**Pipeline:** MLP warm-start (64,32) -> Firefly refinement  \n"
        "**Fitness:** 0.7 x accuracy + 0.3 x (1 - norm_loss) + L2  \n"
        "**Normalization:** StandardScaler"
    )

    col_opt1, col_opt2 = st.columns(2)
    use_augment = col_opt1.checkbox(
        "Data Augmentation",
        value=True,
        help="Rotation, translation, thick-stroke simulation, noise."
    )
    use_balance = col_opt2.checkbox(
        "Balance Classes 1 & 5",
        value=True,
        help="Oversample digits 1 and 5 to match majority class count."
    )

    train_btn = st.button("Start Training", type="primary")

    progress_bar = st.progress(0)
    status_text  = st.empty()
    chart_slot   = st.empty()
    log_slot     = st.empty()
    metrics_slot = st.empty()

    if train_btn:
        from src.train_model import train_model

        status_text.info("Step 1/2: MLP warm-start (64,32) via backpropagation...")
        logs = []

        def on_progress(it, total, best, avg=None):
            progress_bar.progress(it / total)
            avg_str = f" | Avg: {avg*100:.1f}%" if avg is not None else ""
            logs.append(
                f"Iter {it:>4}/{total} | "
                f"Best: {best:.4f} ({best*100:.1f}%){avg_str}"
            )
            if it % 5 == 0 or it == total:
                status_text.info(f"Step 2/2: Firefly running... Iter {it}/{total}  Best={best*100:.1f}%")
                chart_slot.pyplot(make_convergence_fig([], []))
                log_slot.text_area(
                    "Optimization Log (last 20 iterations)",
                    "\n".join(logs[-20:]), height=200,
                )

        results = train_model(
            config={
                "population_size": pop_size,
                "max_iterations":  iterations,
                "alpha":  alpha,
                "beta0":  beta0,
                "gamma":  gamma,
                "use_augment": use_augment,
                "use_balance": use_balance,
                "random_state": 42,
            },
            progress_callback=on_progress,
        )

        nn        = results["nn"]
        scaler    = results["scaler"]
        X_test    = results["X_test"]
        y_test    = results["y_test"]
        train_acc = results["train_acc"]
        test_acc  = results["test_acc"]
        mlp_train_acc = results["mlp_train_acc"]
        mlp_test_acc  = results["mlp_test_acc"]
        best_curve    = results["convergence_best"]
        avg_curve     = results["convergence_avg"]
        meta          = results["meta"]

        y_test_pred = nn.predict(X_test)
        acc1 = float(np.mean(y_test_pred[y_test==1] == 1) * 100) if (y_test==1).sum() > 0 else 0.0
        acc5 = float(np.mean(y_test_pred[y_test==5] == 5) * 100) if (y_test==5).sum() > 0 else 0.0
        meta.update({"acc_digit1": acc1, "acc_digit5": acc5,
                     "augmentation": use_augment, "balancing": use_balance})

        chart_slot.pyplot(make_convergence_fig(best_curve, avg_curve))
        progress_bar.progress(1.0)
        save_model(nn, scaler, meta)

        st.session_state.update({
            "nn": nn, "scaler": scaler, "meta": meta,
            "model_loaded": True,
            "convergence_best": best_curve, "convergence_avg": avg_curve,
            "train_acc": train_acc, "test_acc": test_acc,
            "X_test": X_test, "y_test": y_test,
        })

        with open(os.path.join(RESULTS_DIR, "results.txt"), "a", encoding="utf-8") as f:
            f.write("Dataset: UCI Optical Recognition of Handwritten Digits\n")
            f.write(f"Arch: 64-64-32-10 | Aug: {use_augment} | Balance: {use_balance}\n")
            f.write(f"MLP(64,32) warm-start: Train {mlp_train_acc:.1f}% Test {mlp_test_acc:.1f}%\n")
            f.write(f"Pop={pop_size} Iter={iterations} a={alpha} b0={beta0} g={gamma}\n")
            f.write(f"Firefly: Train {train_acc:.2f}% Test {test_acc:.2f}%\n")
            f.write(f"  Digit-1: {acc1:.0f}%  Digit-5: {acc5:.0f}%\n")
            f.write("-" * 55 + "\n")

        conv_fig = make_convergence_fig(best_curve, avg_curve)
        conv_fig.savefig(os.path.join(RESULTS_DIR, "convergence_curve.png"),
                         dpi=120, bbox_inches="tight")
        plt.close(conv_fig)
        pred_fig = make_sample_predictions_fig(nn, X_test, y_test, scaler)
        pred_fig.savefig(os.path.join(RESULTS_DIR, "digit_predictions.png"),
                         dpi=120, bbox_inches="tight")
        plt.close(pred_fig)

        status_text.success(
            f"Done! Firefly test: {test_acc:.1f}% | Digit-1: {acc1:.0f}% | Digit-5: {acc5:.0f}%"
        )
        with metrics_slot.container():
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("MLP Train",     f"{mlp_train_acc:.1f}%")
            c2.metric("MLP Test",      f"{mlp_test_acc:.1f}%")
            c3.metric("Firefly Train", f"{train_acc:.1f}%")
            c4.metric("Firefly Test",  f"{test_acc:.1f}%")
            c5.metric("Digit-1 Acc",   f"{acc1:.0f}%")
            c6.metric("Digit-5 Acc",   f"{acc5:.0f}%")
            st.pyplot(make_per_class_accuracy_fig(y_test, y_test_pred))
            st.pyplot(make_confusion_fig(y_test, y_test_pred))
            st.pyplot(make_sample_predictions_fig(nn, X_test, y_test, scaler))

    elif st.session_state.get("model_loaded"):
        ta = st.session_state.get("train_acc")
        te = st.session_state.get("test_acc")
        if ta is not None:
            c1, c2 = st.columns(2)
            c1.metric("Training Accuracy", f"{ta:.2f}%")
            c2.metric("Testing Accuracy",  f"{te:.2f}%")

        best_c = st.session_state.get("convergence_best", [])
        avg_c  = st.session_state.get("convergence_avg",  [])
        if best_c:
            st.pyplot(make_convergence_fig(best_c, avg_c))

        if "X_test" in st.session_state:
            nn_s     = st.session_state["nn"]
            scaler_s = st.session_state.get("scaler")
            X_test   = st.session_state["X_test"]
            y_test   = st.session_state["y_test"]
            st.pyplot(make_confusion_fig(y_test, nn_s.predict(X_test)))
            st.pyplot(make_sample_predictions_fig(nn_s, X_test, y_test, scaler_s))

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DRAW & PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_draw:
    # canvas_key is bumped by Clear so st_canvas renders a blank slate
    if "canvas_key" not in st.session_state:
        st.session_state["canvas_key"] = 0

    col_left, col_right = st.columns([1, 2], gap="large")

    with col_left:
        st.subheader("✏️ Draw a Digit (0–9)")
        st.caption(
            "Draw on the black canvas below.  \n"
            "The stroke is resized to 8×8 and divided by 16  \n"
            "— identical to the UCI training format."
        )

        # ── Drawing canvas ────────────────────────────────────────────────────
        # key changes on Clear so the component remounts with a blank canvas
        canvas_result = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=20,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key=f"canvas_{st.session_state['canvas_key']}",
        )

        # Persist image_data the moment the canvas renders so button
        # clicks (which trigger a rerun) don't wipe it
        if (
            canvas_result.image_data is not None
            and canvas_result.image_data.sum() > 0
        ):
            st.session_state["canvas_image"] = canvas_result.image_data

        # ── Buttons row ───────────────────────────────────────────────────────
        btn_col1, btn_col2 = st.columns(2)
        predict_btn = btn_col1.button(
            "🔍 Predict Digit",
            disabled=not st.session_state.get("model_loaded"),
            type="primary",
        )
        if btn_col2.button("🗑️ Clear Canvas"):
            st.session_state["canvas_key"]  += 1   # remount canvas blank
            st.session_state["canvas_image"] = None
            st.session_state.pop("last_pred", None)
            st.rerun()

        # ── 8×8 live preview (shown as soon as there is a stroke) ─────────────
        stored_img = st.session_state.get("canvas_image")
        if stored_img is not None and st.session_state.get("scaler") is not None:
            _steps_prev, raw_preview, _ = canvas_to_uci_vector(
                stored_img, st.session_state["scaler"]
            )
            st.markdown("**Processed 8x8 input:**")
            st.pyplot(make_digit_preview(raw_preview))

    with col_right:
        st.subheader("🔮 Prediction Result")

        if not st.session_state.get("model_loaded"):
            st.info(
                "No trained model found.  \n"
                "Go to the **Train Model** tab, train the network, "
                "then come back here to predict."
            )
        else:
            st.success("Model ready — draw a digit and click Predict")

        # ── Run prediction ────────────────────────────────────────────────────
        if predict_btn:
            img_data = st.session_state.get("canvas_image")
            if img_data is None:
                st.error("Canvas is empty — please draw a digit first.")
            elif not st.session_state.get("scaler"):
                st.error("No scaler found — please retrain the model.")
            else:
                nn_pred    = st.session_state["nn"]
                scaler_inf = st.session_state["scaler"]
                steps, raw_8x8, scaled_vec = canvas_to_uci_vector(img_data, scaler_inf)
                probs = nn_pred.forward(scaled_vec)
                pred  = int(np.argmax(probs))
                conf  = float(probs[0, pred]) * 100
                st.session_state["last_pred"] = {
                    "pred": pred, "conf": conf,
                    "probs": probs, "raw_8x8": raw_8x8,
                    "scaled_vec": scaled_vec, "steps": steps,
                }

        # ── Display cached result (survives reruns) ───────────────────────────
        result = st.session_state.get("last_pred")
        # Guard: discard stale cached results that predate the 'steps' key
        if result and "steps" not in result:
            st.session_state.pop("last_pred", None)
            result = None
        if result:
            pred       = result["pred"]
            conf       = result["conf"]
            probs      = result["probs"]
            raw_8x8    = result["raw_8x8"]
            scaled_vec = result.get("scaled_vec", result.get("normed_vec"))
            steps      = result["steps"]

            st.markdown(
                f"<h1 style='text-align:center;color:#FF4B4B;font-size:3rem;'>"
                f"Predicted: {pred}</h1>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<p style='text-align:center;font-size:1.2rem;'>"
                f"Confidence: <b>{conf:.1f}%</b></p>",
                unsafe_allow_html=True,
            )

            c1, c2 = st.columns([3, 1])
            with c1:
                st.pyplot(make_confidence_bar(probs))
            with c2:
                st.pyplot(make_digit_preview(raw_8x8))

            st.pyplot(make_preprocessing_steps_fig(steps))

            st.markdown("**Top-3 predictions:**")
            top3 = np.argsort(probs[0])[::-1][:3]
            for rank, idx in enumerate(top3, 1):
                bar_pct = int(probs[0, idx] * 100)
                st.markdown(
                    f"{rank}. Digit **{idx}** — "
                    f"`{'█' * (bar_pct // 5)}{' ' * (20 - bar_pct // 5)}` "
                    f"**{probs[0, idx]*100:.2f}%**"
                )

            with st.expander("How the canvas is preprocessed"):
                st.markdown(
                    "1. RGBA canvas → greyscale  \n"
                    "2. Crop to bounding box + 15% padding  \n"
                    "3. Gaussian blur (radius=1)  \n"
                    "4. Resize to **8×8** (UCI resolution)  \n"
                    "5. Scale to **0–16** (UCI native range)  \n"
                    "6. **StandardScaler.transform()** (same scaler as training)"
                )
                if scaled_vec is not None:
                    st.markdown(
                        f"Input vector shape: `(1, 64)`  \n"
                        f"Min: `{scaled_vec.min():.3f}` · "
                        f"Max: `{scaled_vec.max():.3f}` · "
                        f"Mean: `{scaled_vec.mean():.3f}`"
                    )

            best_c = st.session_state.get("convergence_best", [])
            avg_c  = st.session_state.get("convergence_avg",  [])
            if best_c:
                st.markdown("**Last training convergence:**")
                st.pyplot(make_convergence_fig(best_c, avg_c))

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — RESULTS & LOGS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_results:
    st.subheader("📊 Saved Results & Visualizations")

    # results.txt
    results_file = os.path.join(RESULTS_DIR, "results.txt")
    if os.path.exists(results_file):
        with open(results_file, encoding="utf-8") as f:
            content = f.read()
        st.text_area("Experiment Log (results.txt)", content, height=240)
    else:
        st.info("No results.txt yet — run training first.")

    st.markdown("---")

    # Saved PNGs
    saved_imgs = [
        ("convergence_curve.png", "Convergence Curve"),
        ("digit_predictions.png", "Sample Predictions"),
        ("Figure_1.png",          "Figure 1"),
    ]
    img_cols = st.columns(len(saved_imgs))
    for col, (fname, caption) in zip(img_cols, saved_imgs):
        fpath = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(fpath):
            col.image(fpath, caption=caption)

    # Live sample grid
    if st.session_state.get("model_loaded") and "X_test" in st.session_state:
        st.markdown("---")
        st.subheader("Live Sample Predictions (current session)")
        nn_r     = st.session_state["nn"]
        scaler_r = st.session_state.get("scaler")
        X_test   = st.session_state["X_test"]
        y_test   = st.session_state["y_test"]
        st.pyplot(make_sample_predictions_fig(nn_r, X_test, y_test, scaler_r))

    # Model metadata
    if st.session_state.get("model_loaded"):
        st.markdown("---")
        st.subheader("Saved Model Metadata")
        meta = st.session_state.get("meta", {})
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"- **Dataset:** {meta.get('dataset', 'UCI Digits')}")
            st.markdown(f"- **Normalization:** {meta.get('normalization', 'divide_by_16')}")
            st.markdown(f"- **Architecture:** {meta.get('architecture', '64-32-10')}")
            st.markdown(f"- **Population Size:** {meta.get('pop_size', '—')}")
            st.markdown(f"- **Iterations:** {meta.get('iterations', '—')}")
        with col2:
            st.markdown(f"- **Alpha:** {meta.get('alpha', '—')}")
            st.markdown(f"- **Beta0:** {meta.get('beta0', '—')}")
            st.markdown(f"- **Gamma:** {meta.get('gamma', '—')}")
            st.markdown(f"- **Train Acc:** {meta.get('train_acc', 0):.2f}%")
            st.markdown(f"- **Test Acc:** {meta.get('test_acc', 0):.2f}%")
