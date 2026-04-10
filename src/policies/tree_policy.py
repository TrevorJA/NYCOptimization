"""
src/policies/tree_policy.py - Oblique Decision Tree policy.

Maps a state vector to an action vector via a binary tree where each
internal node splits on a linear combination of features (w · x + b < 0).
This is more expressive than axis-aligned trees for the same depth, following
Giuliani et al. 2016 and Quinn et al. 2017 for reservoir EMODPS.
"""

import numpy as np
from . import PolicyBase


class ObliqueTreePolicy(PolicyBase):
    """Oblique Decision Tree policy mapping state -> action.

    Architecture:
      - Full binary tree of configurable depth d
      - n_internal = 2^d - 1 internal nodes, n_leaves = 2^d leaves
      - Each internal node: split weights (n_inputs) + bias (1)
      - Each leaf: output values (n_outputs), passed through sigmoid and
        linearly scaled to [output_min, output_max]

    Parameter layout (flat vector):
      [node0_weights (n_inputs), node0_bias (1),
       node1_weights (n_inputs), node1_bias (1),
       ...,
       node(n_internal-1)_weights, node(n_internal-1)_bias,
       leaf0_values (n_outputs),
       ...,
       leaf(n_leaves-1)_values (n_outputs)]

    Total params = n_internal * (n_inputs + 1) + n_leaves * n_outputs

    Bounds:
      - Split weights: [-1, 1]
      - Split bias:    [-1, 1]
      - Leaf values:   [-3, 3]  (sigmoid maps to ~[0.05, 0.95])
    """

    _SPLIT_LB = -1.0
    _SPLIT_UB =  1.0
    _LEAF_LB  = -3.0
    _LEAF_UB  =  3.0

    def __init__(self, n_inputs, n_outputs, depth=3,
                 output_max=None, output_min=0.0):
        """
        Args:
            n_inputs:   State vector dimension.
            n_outputs:  Action vector dimension.
            depth:      Tree depth (d). Internal nodes = 2^d - 1, leaves = 2^d.
            output_max: Upper bound on each output (scalar or array).
            output_min: Lower bound on each output (scalar or array).
        """
        self._n_inputs = n_inputs
        self._n_outputs = n_outputs
        self._depth = depth
        self._n_internal = 2 ** depth - 1
        self._n_leaves = 2 ** depth

        out_max = np.ones(n_outputs) if output_max is None else np.full(n_outputs, output_max)
        self._output_max = np.asarray(out_max, dtype=np.float64)
        self._output_min = np.full(n_outputs, output_min, dtype=np.float64)

        # Parameter arrays (initialized to zeros; set via set_params)
        self._split_weights = np.zeros((self._n_internal, n_inputs))  # [n_internal, n_inputs]
        self._split_biases  = np.zeros(self._n_internal)              # [n_internal]
        self._leaf_values   = np.zeros((self._n_leaves, n_outputs))   # [n_leaves, n_outputs]

    # ------------------------------------------------------------------
    # PolicyBase interface
    # ------------------------------------------------------------------

    @property
    def n_params(self) -> int:
        return self._n_internal * (self._n_inputs + 1) + self._n_leaves * self._n_outputs

    @property
    def n_inputs(self) -> int:
        return self._n_inputs

    @property
    def n_outputs(self) -> int:
        return self._n_outputs

    @property
    def name(self) -> str:
        return (f"ObliqueTree(depth={self._depth}, "
                f"n_inputs={self._n_inputs}, n_outputs={self._n_outputs})")

    def set_params(self, flat_vector: np.ndarray) -> None:
        """Unpack flat DV vector into split weights, biases, and leaf values."""
        v = np.asarray(flat_vector, dtype=np.float64)
        if v.size != self.n_params:
            raise ValueError(f"Expected {self.n_params} parameters, got {v.size}.")

        ni, no = self._n_inputs, self._n_outputs
        n_int, n_lv = self._n_internal, self._n_leaves

        # Each internal node stores (n_inputs + 1) values: weights then bias
        node_block = n_int * (ni + 1)
        node_data = v[:node_block].reshape(n_int, ni + 1)
        self._split_weights = node_data[:, :ni]
        self._split_biases  = node_data[:, ni]

        self._leaf_values = v[node_block:].reshape(n_lv, no)

    def get_bounds(self) -> tuple:
        """Return (lower_bounds, upper_bounds) as 1-D arrays of length n_params."""
        ni, no = self._n_inputs, self._n_outputs
        n_int, n_lv = self._n_internal, self._n_leaves

        # Build bounds in the same order as set_params: interleaved [weights, bias] per node
        node_lb = np.tile(
            np.concatenate([np.full(ni, self._SPLIT_LB), [self._SPLIT_LB]]),
            n_int
        )
        node_ub = np.tile(
            np.concatenate([np.full(ni, self._SPLIT_UB), [self._SPLIT_UB]]),
            n_int
        )
        leaf_lb = np.full(n_lv * no, self._LEAF_LB)
        leaf_ub = np.full(n_lv * no, self._LEAF_UB)

        lb = np.concatenate([node_lb, leaf_lb])
        ub = np.concatenate([node_ub, leaf_ub])
        return lb, ub

    def __call__(self, state: np.ndarray) -> np.ndarray:
        """Evaluate policy.

        Args:
            state: 1-D array of length n_inputs.

        Returns:
            action: 1-D array of length n_outputs in [output_min, output_max].
        """
        s = np.asarray(state, dtype=np.float64)

        node_idx = 0
        for _ in range(self._depth):
            w = self._split_weights[node_idx]
            b = self._split_biases[node_idx]
            if np.dot(w, s) + b < 0.0:
                node_idx = 2 * node_idx + 1   # left child
            else:
                node_idx = 2 * node_idx + 2   # right child

        leaf_idx = node_idx - self._n_internal
        raw = self._leaf_values[leaf_idx]     # [n_outputs]

        sig = 1.0 / (1.0 + np.exp(-raw))
        return self._output_min + sig * (self._output_max - self._output_min)
