"""
model_io.py  —  Save / load NeuralNetwork weights + StandardScaler + metadata.
"""
import pickle, os
from src.neural_network import NeuralNetwork

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "model.pkl")


def save_model(nn: NeuralNetwork, scaler, meta: dict):
    os.makedirs(os.path.dirname(os.path.abspath(MODEL_PATH)), exist_ok=True)
    payload = {
        "weights": nn.get_weights(),   # single flat vector — architecture-agnostic
        "scaler":  scaler,
        "meta":    meta,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)


def load_model():
    """Returns (NeuralNetwork, scaler, meta) or (None, None, None)."""
    if not os.path.exists(MODEL_PATH):
        return None, None, None
    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)

    nn = NeuralNetwork()

    # Support both old per-matrix format and new flat-vector format
    if "weights" in payload:
        nn.set_weights(payload["weights"])
    else:
        # Legacy: old saves stored W1/b1/W2/b2 separately
        nn.W1, nn.b1 = payload["W1"], payload["b1"]
        nn.W2, nn.b2 = payload["W2"], payload["b2"]

    return nn, payload.get("scaler"), payload.get("meta", {})
