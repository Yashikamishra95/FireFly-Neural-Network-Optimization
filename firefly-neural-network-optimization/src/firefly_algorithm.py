"""
firefly_algorithm.py

Key design decisions
--------------------
warm_start_from_mlp()
    Seeds firefly[0] with MLP weights exactly, then spreads the rest
    with sigma=0.1 (tight) so the population starts near the good basin
    but has enough diversity to explore.  sigma=0.5 was too wide —
    fireflies scattered away from the MLP solution and fitness dropped.

Fitness  0.5 * accuracy + 0.5 * (1 - norm_loss)
    Equal weighting gives the loss term more influence.  Since accuracy
    is a step function, the smooth loss landscape drives visible
    improvement on the convergence curve even when integer accuracy
    hasn't changed yet.

gamma scaling
    gamma is divided by dimension so the attraction term beta =
    exp(-gamma * r^2) stays meaningful in high-dimensional spaces.
    With raw gamma=0.1 and r~sqrt(6570)*noise, beta collapses to ~0
    and no movement happens.

Noise injection on stagnation
    If best fitness hasn't improved for stagnation_limit iterations,
    Gaussian noise (std=0.05) is added to ALL fireflies so the whole
    swarm escapes the plateau, not just the bottom half.

Alpha decay  0.98 per iteration
    Starts at 0.5 (exploration), decays toward exploitation.
"""
import numpy as np
from src.neural_network import NeuralNetwork


class FireflyAlgorithm:

    WEIGHT_BOUND = 3.0 # Bound for the weights to prevent extreme values that cause NaN in loss

    def __init__(self, population_size, dimension, alpha, beta0, gamma,
                 max_iterations, neural_network: NeuralNetwork,
                 X_train, y_train, stagnation_limit: int = 10):
        self.population_size  = population_size
        self.dimension        = dimension
        self.alpha            = alpha
        self.beta0            = beta0
        # Scale gamma by dimension so attraction doesn't collapse in high-dim
        self.gamma            = gamma / dimension
        self.max_iterations   = max_iterations
        self.neural_network   = neural_network
        self.X_train          = X_train
        self.y_train          = y_train
        self.stagnation_limit = stagnation_limit

        self.convergence_curve = []
        self.fireflies         = None
        self.fitness_values    = None
        self.best_firefly      = None
        self.best_fitness      = -np.inf
        self._stagnation_count = 0

    # ── Population init ───────────────────────────────────────────────────────

    def initialize_population(self):
        self.fireflies = np.random.uniform(
            -self.WEIGHT_BOUND, self.WEIGHT_BOUND,
            (self.population_size, self.dimension),
        )
        self.fitness_values = np.zeros(self.population_size)

    def warm_start_from_mlp(self, mlp):
        """
        Layered diversity around the MLP seed:
          - 1/3 tight  (sigma=0.1) — exploit the good basin
          - 1/3 medium (sigma=0.5) — explore nearby
          - 1/3 wide   (sigma=1.5) — explore broadly
        This guarantees fitness variation across the population from
        iteration 1, so fireflies always have brighter neighbours to
        move toward.
        """
        parts = []
        for coef, intercept in zip(mlp.coefs_, mlp.intercepts_):
            parts.extend([coef.ravel(), intercept.ravel()])
        seed = np.concatenate(parts)
        seed = np.clip(seed, -self.WEIGHT_BOUND, self.WEIGHT_BOUND)

        self.fireflies[0] = seed
        n  = self.population_size - 1
        t  = n // 3
        m  = n // 3
        w  = n - t - m
        idx = 1
        for sigma, count in [(0.1, t), (0.5, m), (1.5, w)]:
            noise = np.random.normal(0, sigma, (count, self.dimension))
            self.fireflies[idx:idx+count] = np.clip(
                seed + noise, -self.WEIGHT_BOUND, self.WEIGHT_BOUND
            )
            idx += count

    # ── Weight interface ──────────────────────────────────────────────────────

    def _apply(self, w):
        self.neural_network.set_weights(w)

    # ── Fitness ───────────────────────────────────────────────────────────────

    def fitness(self, w):
        self._apply(w)
        nn     = self.neural_network
        y_pred = nn.forward(self.X_train)

        y_true = (np.argmax(self.y_train, axis=1)
                  if self.y_train.ndim > 1 else self.y_train)

        accuracy  = np.mean(np.argmax(y_pred, axis=1) == y_true)
        loss      = nn.compute_loss(y_true, y_pred)
        norm_loss = min(loss / np.log(10), 1.0)

        # Equal weighting: loss term provides smooth gradient on the curve
        return 0.5 * accuracy + 0.5 * (1.0 - norm_loss)

    # ── Movement ──────────────────────────────────────────────────────────────

    def move_fireflies(self):
        for i in range(self.population_size):
            for j in range(self.population_size):
                if self.fitness_values[j] > self.fitness_values[i]:
                    r    = np.linalg.norm(self.fireflies[i] - self.fireflies[j])
                    beta = self.beta0 * np.exp(-self.gamma * r * r)
                    step = self.alpha * (np.random.rand(self.dimension) - 0.5)
                    self.fireflies[i] += beta * (self.fireflies[j] - self.fireflies[i]) + step
                    self.fireflies[i]  = np.clip(self.fireflies[i],
                                                  -self.WEIGHT_BOUND, self.WEIGHT_BOUND)

    def _escape_stagnation(self):
        """Inject meaningful noise into all fireflies except the best."""
        self.fireflies[1:] = np.clip(
            self.fireflies[1:] + np.random.normal(0, 0.3, (self.population_size - 1, self.dimension)),
            -self.WEIGHT_BOUND, self.WEIGHT_BOUND,
        )
        self.fireflies[0] = self.best_firefly.copy()
        self._stagnation_count = 0

    # ── Main loop ─────────────────────────────────────────────────────────────

    def optimize(self):
        if self.fireflies is None:
            self.initialize_population()
        prev_best = -np.inf

        for it in range(self.max_iterations):
            for i in range(self.population_size):
                self.fitness_values[i] = self.fitness(self.fireflies[i])

            best_idx = int(np.argmax(self.fitness_values))
            if self.fitness_values[best_idx] > self.best_fitness:
                self.best_fitness = self.fitness_values[best_idx]
                self.best_firefly = self.fireflies[best_idx].copy()

            if self.best_fitness <= prev_best:
                self._stagnation_count += 1
            else:
                self._stagnation_count = 0
            prev_best = self.best_fitness

            if self._stagnation_count >= self.stagnation_limit:
                self._escape_stagnation()

            avg_fitness = float(np.mean(self.fitness_values))
            self.convergence_curve.append(float(self.best_fitness))

            self.move_fireflies()
            self.alpha *= 0.98  # decay: exploration → exploitation

            if (it + 1) % 10 == 0:
                print(f"Iter {it+1}/{self.max_iterations} | "
                      f"Best={self.best_fitness:.4f} ({self.best_fitness*100:.1f}%) | "
                      f"Avg={avg_fitness:.4f} ({avg_fitness*100:.1f}%)")

        self._apply(self.best_firefly)
        return self.best_firefly
