"""
src/policies/ann_policy.py - Feedforward Artificial Neural Network policy.

Maps a normalized state vector to an action vector via a two-hidden-layer
feedforward network with ReLU activations and sigmoid output scaling.
"""

import numpy as np
from . import PolicyBase


class ANNPolicy(PolicyBase):
    """Feedforward ANN policy mapping state -> action.

    Architecture:
      - Input layer: n_inputs neurons
      - Hidden layer 1: h1 neurons with ReLU activation
      - Hidden layer 2: h2 neurons with ReLU activation
      - Output layer: n_outputs neurons with sigmoid activation,
        linearly scaled to [output_min, output_max]

    Parameter layout (flat vector):
      [W1 (n_inputs × h1), b1 (h1),
       W2 (h1 × h2),       b2 (h2),
       W3 (h2 × n_outputs), b3 (n_outputs)]

    Total params = n_inputs*h1 + h1 + h1*h2 + h2 + h2*n_outputs + n_outputs

    For n_inputs=9, n_outputs=1, h1=8, h2=8:
      9*8 + 8 + 8*8 + 8 + 8*1 + 1 = 72+8+64+8+8+1 = 161

    For n_inputs=15, n_outputs=1, h1=8, h2=8:
      15*8 + 8 + 8*8 + 8 + 8*1 + 1 = 120+8+64+8+8+1 = 209

    Bounds:
      - All weights and biases: [-3, 3]
        Wide enough for diverse function expression; consistent with
        EMODPS ANN literature (e.g., Giuliani et al. 2016).
    """

    _PARAM_LB = -3.0
    _PARAM_UB =  3.0

    def __init__(self, n_inputs, n_outputs, h1=8, h2=8,
                 output_max=None, output_min=0.0):
        """
        Args:
            n_inputs:   State vector dimension.
            n_outputs:  Action vector dimension.
            h1:         Number of neurons in hidden layer 1.
            h2:         Number of neurons in hidden layer 2.
            output_max: Upper bound on each output (scalar or array).
            output_min: Lower bound on each output (scalar or array).
        """
        self._n_inputs = n_inputs
        self._n_outputs = n_outputs
        self._h1 = h1
        self._h2 = h2

        out_max = np.ones(n_outputs) if output_max is None else np.full(n_outputs, output_max)
        self._output_max = np.asarray(out_max, dtype=np.float64)
        self._output_min = np.full(n_outputs, output_min, dtype=np.float64)

        # Parameter arrays (initialized to zeros; set via set_params)
        self._W1 = np.zeros((n_inputs, h1))   # [n_inputs, h1]
        self._b1 = np.zeros(h1)               # [h1]
        self._W2 = np.zeros((h1, h2))         # [h1, h2]
        self._b2 = np.zeros(h2)               # [h2]
        self._W3 = np.zeros((h2, n_outputs))  # [h2, n_outputs]
        self._b3 = np.zeros(n_outputs)        # [n_outputs]

    # ------------------------------------------------------------------
    # PolicyBase interface
    # ------------------------------------------------------------------

    @property
    def n_params(self) -> int:
        ni, no = self._n_inputs, self._n_outputs
        h1, h2 = self._h1, self._h2
        return ni * h1 + h1 + h1 * h2 + h2 + h2 * no + no

    @property
    def n_inputs(self) -> int:
        return self._n_inputs

    @property
    def n_outputs(self) -> int:
        return self._n_outputs

    @property
    def name(self) -> str:
        return (f"ANN(h1={self._h1}, h2={self._h2}, "
                f"n_inputs={self._n_inputs}, n_outputs={self._n_outputs})")

    def set_params(self, flat_vector: np.ndarray) -> None:
        """Unpack flat DV vector into layer weights and biases."""
        v = np.asarray(flat_vector, dtype=np.float64)
        if v.size != self.n_params:
            raise ValueError(
                f"Expected {self.n_params} parameters, got {v.size}."
            )
        ni, no = self._n_inputs, self._n_outputs
        h1, h2 = self._h1, self._h2

        i = 0
        self._W1 = v[i:i + ni * h1].reshape(ni, h1);  i += ni * h1
        self._b1 = v[i:i + h1];                        i += h1
        self._W2 = v[i:i + h1 * h2].reshape(h1, h2);  i += h1 * h2
        self._b2 = v[i:i + h2];                        i += h2
        self._W3 = v[i:i + h2 * no].reshape(h2, no);  i += h2 * no
        self._b3 = v[i:i + no]

    def get_bounds(self) -> tuple:
        """Return (lower_bounds, upper_bounds) as 1-D arrays of length n_params."""
        lb = np.full(self.n_params, self._PARAM_LB)
        ub = np.full(self.n_params, self._PARAM_UB)
        return lb, ub

    def __call__(self, state: np.ndarray) -> np.ndarray:
        """Evaluate policy.

        Args:
            state: 1-D array of length n_inputs (normalized to [0, 1]).

        Returns:
            action: 1-D array of length n_outputs in [output_min, output_max].
        """
        x = np.asarray(state, dtype=np.float64)
        x = np.maximum(0, x @ self._W1 + self._b1)                   # ReLU
        x = np.maximum(0, x @ self._W2 + self._b2)                   # ReLU
        x = x @ self._W3 + self._b3                                   # linear
        sig = 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))             # sigmoid
        return self._output_min + sig * (self._output_max - self._output_min)
