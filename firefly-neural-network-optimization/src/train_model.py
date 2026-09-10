"""
train_model.py
Shared training pipeline for both train.py (CLI) and streamlit_app.py (UI).

Pipeline
--------
1. Load + optionally balance/augment UCI Digits via data_loader
2. MLP warm-start  (64, 32) hidden layers via sklearn backprop
3. Firefly refinement of the warm-started weights
4. Evaluate and return results

Both entry points call train_model(config) and get back the same
NeuralNetwork, scaler, metrics, and convergence curves.
"""
import numpy as np
from sklearn.neural_network import MLPClassifier

from src.data_loader      import load_uci_digits, augment_data, balance_classes
from src.neural_network   import NeuralNetwork
from src.firefly_algorithm import FireflyAlgorithm
from src.utils            import one_hot

from sklearn.model_selection import train_test_split


def train_model(config: dict, progress_callback=None) -> dict:
    """
    Parameters
    ----------
    config : dict with keys:
        population_size  int   (default 40)
        max_iterations   int   (default 100)
        alpha            float (default 0.5)
        beta0            float (default 1.0)
        gamma            float (default 0.1)
        use_augment      bool  (default True)
        use_balance      bool  (default True)
        random_state     int   (default 42)

    progress_callback : callable(iteration, total, best_fitness) | None
        Called after each Firefly iteration so callers can update a UI.

    Returns
    -------
    dict with keys:
        nn              NeuralNetwork  (weights set to best found)
        scaler          fitted StandardScaler
        X_test          np.ndarray
        y_test          np.ndarray
        train_acc       float  (%)
        test_acc        float  (%)
        mlp_train_acc   float  (%)
        mlp_test_acc    float  (%)
        convergence_best list[float]
        convergence_avg  list[float]
        meta            dict   (all config + results, for model_io)
    """
    np.random.seed(config.get("random_state", 42))

    pop_size   = config.get("population_size", 40)
    iterations = config.get("max_iterations",  100)
    alpha      = config.get("alpha",  0.5)
    beta0      = config.get("beta0",  1.0)
    gamma      = config.get("gamma",  0.1)
    use_aug    = config.get("use_augment", True)
    use_bal    = config.get("use_balance", True)
    rng        = config.get("random_state", 42)

    # ── 1. Load base split (no aug/balance yet — done on raw data) ────────────
    _, X_test, _, y_test, scaler, X_raw_all, y_raw_all, _ = load_uci_digits(
        random_state=rng
    )

    X_tr_raw, _, y_train, _ = train_test_split(
        X_raw_all, y_raw_all, test_size=0.2, random_state=rng, stratify=y_raw_all
    )

    if use_bal:
        X_tr_raw, y_train = balance_classes(
            X_tr_raw, y_train, focus_digits=(1, 5), random_state=rng
        )

    if use_aug:
        X_tr_raw, y_train = augment_data(
            X_tr_raw, y_train, copies=3, random_state=rng
        )

    X_train     = scaler.transform(X_tr_raw)
    y_train_oh  = one_hot(y_train)

    # ── 2. MLP warm-start ─────────────────────────────────────────────────────
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=800,
        random_state=rng,
        learning_rate_init=0.001,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )
    mlp.fit(X_train, y_train)
    mlp_train_acc = mlp.score(X_train, y_train) * 100
    mlp_test_acc  = mlp.score(X_test,  y_test)  * 100

    # ── 3. Firefly refinement ─────────────────────────────────────────────────
    nn = NeuralNetwork()
    fa = FireflyAlgorithm(
        population_size=pop_size,
        dimension=nn.n_weights,
        alpha=alpha,
        beta0=beta0,
        gamma=gamma,
        max_iterations=iterations,
        neural_network=nn,
        X_train=X_train,
        y_train=y_train_oh,
    )
    fa.initialize_population()
    fa.warm_start_from_mlp(mlp)

    best_curve, avg_curve = [], []
    prev_best = -np.inf

    for it in range(iterations):
        for i in range(pop_size):
            fa.fitness_values[i] = fa.fitness(fa.fireflies[i])

        best_idx = int(np.argmax(fa.fitness_values))
        if fa.fitness_values[best_idx] > fa.best_fitness:
            fa.best_fitness = fa.fitness_values[best_idx]
            fa.best_firefly = fa.fireflies[best_idx].copy()

        if fa.best_fitness <= prev_best:
            fa._stagnation_count += 1
        else:
            fa._stagnation_count = 0
        prev_best = fa.best_fitness

        if fa._stagnation_count >= fa.stagnation_limit:
            fa._escape_stagnation()

        avg_fit = float(np.mean(fa.fitness_values))
        best_curve.append(float(fa.best_fitness))
        avg_curve.append(avg_fit)
        fa.convergence_curve = best_curve

        if it == 0:
            print(f"  [Iter 1] Fitness spread — Min={min(fa.fitness_values):.4f}  "
                  f"Max={max(fa.fitness_values):.4f}  "
                  f"Avg={avg_fit:.4f}  Std={np.std(fa.fitness_values):.4f}")

        fa.move_fireflies()
        fa.alpha *= 0.98

        if progress_callback:
            progress_callback(it + 1, iterations, fa.best_fitness, avg_fit)

    fa._apply(fa.best_firefly)

    # ── 4. Evaluate ───────────────────────────────────────────────────────────
    train_acc = float(np.mean(nn.predict(X_train) == y_train) * 100)
    test_acc  = float(np.mean(nn.predict(X_test)  == y_test)  * 100)

    meta = {
        "population_size": pop_size,
        "max_iterations":  iterations,
        "alpha": alpha, "beta0": beta0, "gamma": gamma,
        "use_augment": use_aug, "use_balance": use_bal,
        "random_state": rng,
        "train_acc":     train_acc,
        "test_acc":      test_acc,
        "mlp_train_acc": mlp_train_acc,
        "mlp_test_acc":  mlp_test_acc,
        "convergence_best": best_curve,
        "convergence_avg":  avg_curve,
        "normalization":  "StandardScaler",
        "architecture":   "64-64-32-10",
        "dataset":        "UCI Optical Recognition of Handwritten Digits",
    }

    return {
        "nn":              nn,
        "scaler":          scaler,
        "X_test":          X_test,
        "y_test":          y_test,
        "train_acc":       train_acc,
        "test_acc":        test_acc,
        "mlp_train_acc":   mlp_train_acc,
        "mlp_test_acc":    mlp_test_acc,
        "convergence_best": best_curve,
        "convergence_avg":  avg_curve,
        "meta":            meta,
    }
