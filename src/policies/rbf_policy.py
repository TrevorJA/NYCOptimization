"""
src/policies/rbf_policy.py - Radial Basis Function policy.

Maps a normalized state vector to an action vector via a weighted sum of
Gaussian RBF activations, bounded by sigmoid output scaling.
"""

import numpy as np
from . import PolicyBase


class RBFPolicy(PolicyBase):
    """Radial Basis Function policy mapping state -> action.

    Architecture:
      - n_rbf Gaussian centers in input space
      - Each center has: coordinates (n_inputs), width (1), weights (n_outputs)
      - Output = normalized weighted sum of activations, passed through sigmoid
        and linearly scaled to [output_min, output_max]

    Parameters per RBF (total = n_rbf * (n_inputs + 1 + n_outputs)):
      - centers:  n_inputs values in [0, 1]
      - width:    1 value in [0.01, 2.0]
      - weights:  n_outputs values in [-1, 1]
    """

    # Center bounds [-1, 1]: per Giuliani et al. 2016 and Zatarain Salazar et al.
    # 2024, all state inputs are normalized to [0, 1] before the RBF, and centers
    # span [-1, 1] to allow placement near and slightly beyond input boundaries.
    _CENTER_LB = -1.0
    _CENTER_UB = 1.0
    _WIDTH_LB = 0.01
    _WIDTH_UB = 2.0
    _WEIGHT_LB = -1.0
    _WEIGHT_UB = 1.0
    _EPSILON = 1e-6

    def __init__(self, n_inputs, n_outputs, n_rbf=6,
                 output_max=None, output_min=0.0):
        """
        Args:
            n_inputs:   State vector dimension.
            n_outputs:  Action vector dimension.
            n_rbf:      Number of radial basis functions.
            output_max: Upper bound on each output (scalar or array).
            output_min: Lower bound on each output (scalar or array).
        """
        self._n_inputs = n_inputs
        self._n_outputs = n_outputs
        self._n_rbf = n_rbf

        out_max = np.ones(n_outputs) if output_max is None else np.full(n_outputs, output_max)
        self._output_max = np.asarray(out_max, dtype=np.float64)
        self._output_min = np.full(n_outputs, output_min, dtype=np.float64)

        # Parameter arrays (initialized to zero; set via set_params)
        self._centers = np.zeros((n_rbf, n_inputs))   # [n_rbf, n_inputs]
        self._widths  = np.ones(n_rbf) * 0.5           # [n_rbf]
        self._weights = np.zeros((n_rbf, n_outputs))   # [n_rbf, n_outputs]

    # ------------------------------------------------------------------
    # PolicyBase interface
    # ------------------------------------------------------------------

    @property
    def n_params(self) -> int:
        return self._n_rbf * (self._n_inputs + 1 + self._n_outputs)

    @property
    def n_inputs(self) -> int:
        return self._n_inputs

    @property
    def n_outputs(self) -> int:
        return self._n_outputs

    @property
    def name(self) -> str:
        return f"RBF(n_rbf={self._n_rbf}, n_inputs={self._n_inputs}, n_outputs={self._n_outputs})"

    def set_params(self, flat_vector: np.ndarray) -> None:
        """Unpack flat DV vector into centers, widths, weights."""
        v = np.asarray(flat_vector, dtype=np.float64)
        if v.size != self.n_params:
            raise ValueError(
                f"Expected {self.n_params} parameters, got {v.size}."
            )
        ni, no, nr = self._n_inputs, self._n_outputs, self._n_rbf
        center_end = nr * ni
        width_end  = center_end + nr
        self._centers = v[:center_end].reshape(nr, ni)
        self._widths  = v[center_end:width_end]
        self._weights = v[width_end:].reshape(nr, no)

    def get_bounds(self) -> tuple:
        """Return (lower_bounds, upper_bounds) as 1-D arrays of length n_params."""
        ni, no, nr = self._n_inputs, self._n_outputs, self._n_rbf
        lb = np.concatenate([
            np.full(nr * ni, self._CENTER_LB),          # center lower bounds
            np.full(nr, self._WIDTH_LB),                # width lower bounds
            np.full(nr * no, self._WEIGHT_LB),          # weight lower bounds
        ])
        ub = np.concatenate([
            np.full(nr * ni, self._CENTER_UB),          # center upper bounds
            np.full(nr, self._WIDTH_UB),                # width upper bounds
            np.full(nr * no, self._WEIGHT_UB),          # weight upper bounds
        ])
        return lb, ub

    def __call__(self, state: np.ndarray) -> np.ndarray:
        """Evaluate policy.

        Args:
            state: 1-D array of length n_inputs (normalized to [0,1]).

        Returns:
            action: 1-D array of length n_outputs in [output_min, output_max].
        """
        s = np.asarray(state, dtype=np.float64)

        # Gaussian activations: [n_rbf]
        diff = s[np.newaxis, :] - self._centers          # [n_rbf, n_inputs]
        sq_dist = np.sum(diff ** 2, axis=1)              # [n_rbf]
        activations = np.exp(-sq_dist / (2.0 * self._widths ** 2))

        # Normalized weighted sum: [n_outputs]
        total_activation = np.maximum(activations.sum(), self._EPSILON)
        raw = (activations @ self._weights) / total_activation  # [n_outputs]

        # Sigmoid scaling to [output_min, output_max]
        sig = 1.0 / (1.0 + np.exp(-raw))
        return self._output_min + sig * (self._output_max - self._output_min)
