"""
src/policies/spline_additive_policy.py - Additive B-spline policy.

A single-layer Kolmogorov-Arnold Network: for each (input, output) pair, a
learnable univariate B-spline maps the normalized state feature to a scalar
contribution. The action is the sigmoid-scaled sum of those contributions
plus a per-output bias.

Mathematically equivalent to a Generalized Additive Model (GAM) with learnable
B-spline bases per feature and output. See docs/notes/kan_policy_review.md for
a justification of the single-layer additive form over a full multi-layer KAN.
"""

import numpy as np
from scipy.interpolate import BSpline

from . import PolicyBase


class SplineAdditivePolicy(PolicyBase):
    """Additive B-spline policy mapping state -> action.

    Architecture:
      - For each (input_i, output_o) pair, a univariate B-spline phi_{i,o}
        on [0, 1] with clamped uniform knot vector, grid size G, order k.
      - Each spline has (G + k) control-point coefficients.
      - Output: y_o = sigmoid( sum_i phi_{i,o}(x_i) + b_o ), linearly
        scaled to [output_min, output_max].

    Parameter layout (flat vector):
      [spline_coefs (n_inputs x n_outputs x n_basis),
       biases (n_outputs)]
      where n_basis = G + k.

    Total params: n_inputs * n_outputs * (G + k) + n_outputs

    For n_inputs=9, n_outputs=1, G=5, k=3:
      9 * 1 * 8 + 1 = 73

    For n_inputs=15, n_outputs=1, G=5, k=3:
      15 * 1 * 8 + 1 = 121

    Bounds:
      - Spline coefficients: [-1, 1]  (matches RBF weight convention).
      - Output biases:       [-3, 3]  (matches ANN bias convention).
    """

    _COEF_LB = -1.0
    _COEF_UB =  1.0
    _BIAS_LB = -3.0
    _BIAS_UB =  3.0

    def __init__(self, n_inputs, n_outputs, grid_size=5, spline_order=3,
                 output_max=None, output_min=0.0):
        """
        Args:
            n_inputs:     State vector dimension.
            n_outputs:    Action vector dimension.
            grid_size:    Number of interior grid intervals on [0, 1] (G).
            spline_order: Spline polynomial order (k). k=3 is cubic.
            output_max:   Upper bound on each output (scalar or array).
            output_min:   Lower bound on each output (scalar or array).
        """
        self._n_inputs = n_inputs
        self._n_outputs = n_outputs
        self._G = int(grid_size)
        self._k = int(spline_order)
        self._n_basis = self._G + self._k

        # Clamped uniform knot vector on [0, 1]:
        #   k copies of 0, then linspace(0, 1, G+1), then k copies of 1.
        # Total knots = G + 1 + 2k, number of basis functions = G + k.
        interior = np.linspace(0.0, 1.0, self._G + 1)
        self._knots = np.concatenate([
            np.zeros(self._k),
            interior,
            np.ones(self._k),
        ])

        out_max = np.ones(n_outputs) if output_max is None else np.full(n_outputs, output_max)
        self._output_max = np.asarray(out_max, dtype=np.float64)
        self._output_min = np.full(n_outputs, output_min, dtype=np.float64)

        # Parameter storage (initialized to zeros; set via set_params)
        self._coefs = np.zeros((n_inputs, n_outputs, self._n_basis))
        self._biases = np.zeros(n_outputs)

    # ------------------------------------------------------------------
    # PolicyBase interface
    # ------------------------------------------------------------------

    @property
    def n_params(self) -> int:
        return self._n_inputs * self._n_outputs * self._n_basis + self._n_outputs

    @property
    def n_inputs(self) -> int:
        return self._n_inputs

    @property
    def n_outputs(self) -> int:
        return self._n_outputs

    @property
    def name(self) -> str:
        return (f"SplineAdditive(n_inputs={self._n_inputs}, G={self._G}, "
                f"k={self._k}, n_outputs={self._n_outputs})")

    def set_params(self, flat_vector: np.ndarray) -> None:
        """Unpack flat DV vector into spline coefficients and biases."""
        v = np.asarray(flat_vector, dtype=np.float64)
        if v.size != self.n_params:
            raise ValueError(
                f"Expected {self.n_params} parameters, got {v.size}."
            )
        ni, no, nb = self._n_inputs, self._n_outputs, self._n_basis
        n_coef = ni * no * nb
        self._coefs = v[:n_coef].reshape(ni, no, nb)
        self._biases = v[n_coef:].copy()

    def get_bounds(self) -> tuple:
        """Return (lower_bounds, upper_bounds) as 1-D arrays of length n_params."""
        ni, no, nb = self._n_inputs, self._n_outputs, self._n_basis
        n_coef = ni * no * nb
        lb = np.concatenate([
            np.full(n_coef, self._COEF_LB),
            np.full(no, self._BIAS_LB),
        ])
        ub = np.concatenate([
            np.full(n_coef, self._COEF_UB),
            np.full(no, self._BIAS_UB),
        ])
        return lb, ub

    def __call__(self, state: np.ndarray) -> np.ndarray:
        """Evaluate policy.

        Args:
            state: 1-D array of length n_inputs (normalized to [0, 1]).

        Returns:
            action: 1-D array of length n_outputs in [output_min, output_max].
        """
        s = np.clip(np.asarray(state, dtype=np.float64), 0.0, 1.0)

        # Sum of univariate spline contributions per output.
        # _coefs[i, :, :] has shape (n_outputs, n_basis); BSpline expects
        # the leading axis of c to be the basis axis, so transpose to
        # (n_basis, n_outputs). BSpline(...)(s[i]) then returns an array
        # of shape (n_outputs,).
        contrib = np.zeros(self._n_outputs)
        for i in range(self._n_inputs):
            c = self._coefs[i].T                             # [n_basis, n_outputs]
            spline = BSpline(self._knots, c, self._k, extrapolate=True)
            contrib += spline(s[i])

        # Sigmoid (clipped for numerical stability) + linear scaling.
        raw = contrib + self._biases
        sig = 1.0 / (1.0 + np.exp(-np.clip(raw, -20, 20)))
        return self._output_min + sig * (self._output_max - self._output_min)
