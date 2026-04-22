"""
src/policies/tree_policy.py - Soft Oblique Decision Tree policy.

A binary tree of configurable depth whose internal nodes split on a learned
linear combination of inputs, but where the split is a **soft** probabilistic
gate (not a hard branch). Every leaf contributes to the output weighted by
its root-to-leaf path probability, so the policy is a continuous,
differentiable function of the state — appropriate for reservoir release
control, where physical realism demands a smooth response surface.

References:
  - Suárez & Lutsko (1999), "Globally Optimal Fuzzy Decision Trees."
  - Irsoy, Yıldız, Alpaydın (2012), "Soft Decision Trees," ICPR.
  - Frosst & Hinton (2017), "Distilling a Neural Network Into a Soft
    Decision Tree."

This replaces the earlier hard-branch ObliqueTreePolicy, which was
piecewise-constant and produced physically unrealistic step changes in
release across split boundaries.
"""

import numpy as np
from . import PolicyBase


class SoftTreePolicy(PolicyBase):
    """Soft oblique decision tree mapping state -> action.

    Architecture:
      - Full binary tree of configurable depth d.
      - n_internal = 2^d - 1 internal nodes, n_leaves = 2^d leaves.
      - Each internal node has split weights w (n_inputs) and bias b.
        Probability of taking the RIGHT branch at node i is
            p_right_i = sigmoid(gamma * (w_i · s + b_i)).
      - Each leaf has output values; the final raw output is the
        path-probability-weighted sum of all leaves, then sigmoid and
        linearly scaled to [output_min, output_max].
      - A single scalar temperature gamma is optimized alongside the
        weights/biases/leaves. Higher gamma -> sharper gating (approaching
        the hard tree in the limit); lower gamma -> softer mixing.

    Parameter layout (flat vector):
      [node0_weights (n_inputs), node0_bias (1),
       node1_weights (n_inputs), node1_bias (1),
       ...,
       node(n_internal-1)_weights, node(n_internal-1)_bias,
       leaf0_values (n_outputs),
       ...,
       leaf(n_leaves-1)_values (n_outputs),
       gamma (1)]

    Total params = n_internal * (n_inputs + 1) + n_leaves * n_outputs + 1

    Bounds:
      - Split weights:  [-1, 1]
      - Split biases:   [-1, 1]
      - Leaf values:    [-3, 3]  (sigmoid maps to ~[0.05, 0.95])
      - Gamma:          [1,  20]  (soft -> near-hard gating)
    """

    _SPLIT_LB = -1.0
    _SPLIT_UB =  1.0
    _LEAF_LB  = -3.0
    _LEAF_UB  =  3.0
    _GAMMA_LB =  1.0
    _GAMMA_UB = 20.0

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
        self._split_weights = np.zeros((self._n_internal, n_inputs))
        self._split_biases  = np.zeros(self._n_internal)
        self._leaf_values   = np.zeros((self._n_leaves, n_outputs))
        # Moderate default temperature; overwritten by set_params.
        self._gamma = 5.0

    # ------------------------------------------------------------------
    # PolicyBase interface
    # ------------------------------------------------------------------

    @property
    def n_params(self) -> int:
        return self._n_internal * (self._n_inputs + 1) + self._n_leaves * self._n_outputs + 1

    @property
    def n_inputs(self) -> int:
        return self._n_inputs

    @property
    def n_outputs(self) -> int:
        return self._n_outputs

    @property
    def name(self) -> str:
        return (f"SoftTree(depth={self._depth}, gamma={self._gamma:.2f}, "
                f"n_inputs={self._n_inputs}, n_outputs={self._n_outputs})")

    def set_params(self, flat_vector: np.ndarray) -> None:
        """Unpack flat DV vector into split weights, biases, leaf values, and gamma."""
        v = np.asarray(flat_vector, dtype=np.float64)
        if v.size != self.n_params:
            raise ValueError(f"Expected {self.n_params} parameters, got {v.size}.")

        ni, no = self._n_inputs, self._n_outputs
        n_int, n_lv = self._n_internal, self._n_leaves

        node_block = n_int * (ni + 1)
        leaf_block = n_lv * no

        node_data = v[:node_block].reshape(n_int, ni + 1)
        self._split_weights = node_data[:, :ni]
        self._split_biases  = node_data[:, ni]

        self._leaf_values = v[node_block:node_block + leaf_block].reshape(n_lv, no)
        self._gamma = float(v[node_block + leaf_block])

    def get_bounds(self) -> tuple:
        """Return (lower_bounds, upper_bounds) as 1-D arrays of length n_params."""
        ni, no = self._n_inputs, self._n_outputs
        n_int, n_lv = self._n_internal, self._n_leaves

        # Each internal node stores [weights..., bias] consecutively.
        node_lb = np.tile(
            np.concatenate([np.full(ni, self._SPLIT_LB), [self._SPLIT_LB]]),
            n_int,
        )
        node_ub = np.tile(
            np.concatenate([np.full(ni, self._SPLIT_UB), [self._SPLIT_UB]]),
            n_int,
        )
        leaf_lb = np.full(n_lv * no, self._LEAF_LB)
        leaf_ub = np.full(n_lv * no, self._LEAF_UB)

        lb = np.concatenate([node_lb, leaf_lb, [self._GAMMA_LB]])
        ub = np.concatenate([node_ub, leaf_ub, [self._GAMMA_UB]])
        return lb, ub

    def __call__(self, state: np.ndarray) -> np.ndarray:
        """Evaluate policy via soft tree traversal.

        Output is the path-probability-weighted mixture of all leaf values,
        passed through sigmoid and scaled to [output_min, output_max].
        """
        s = np.asarray(state, dtype=np.float64)

        # p_right[i] = sigmoid(gamma * (w_i · s + b_i)) at each internal node.
        logits = self._gamma * (self._split_weights @ s + self._split_biases)
        p_right = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))

        # Walk each leaf back to the root, multiplying the appropriate gate
        # probability for each step. O(n_leaves * depth) — negligible for d<=4.
        raw = np.zeros(self._n_outputs, dtype=np.float64)
        for j in range(self._n_leaves):
            node_idx = self._n_internal + j
            prob = 1.0
            while node_idx > 0:
                parent = (node_idx - 1) // 2
                if node_idx == 2 * parent + 2:
                    prob *= p_right[parent]
                else:
                    prob *= 1.0 - p_right[parent]
                node_idx = parent
            raw += prob * self._leaf_values[j]

        sig = 1.0 / (1.0 + np.exp(-np.clip(raw, -20.0, 20.0)))
        return self._output_min + sig * (self._output_max - self._output_min)
