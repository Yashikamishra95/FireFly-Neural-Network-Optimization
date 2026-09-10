"""
model.py
Public interface for the Neural Network model.
Architecture: 64 → 32 (ReLU) → 10 (Softmax)
"""
from src.neural_network import NeuralNetwork

__all__ = ["NeuralNetwork"]
