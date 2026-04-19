"""
external.py - External policy architecture registry.

Tracks formulations that use learned or parameterized external policy
architectures (RBF, ANN, decision-tree, etc.) rather than the FFMP
rule structure. Built-in architectures (rbf, tree, ann) are registered
at module load time; add-ons can call register_architecture() directly.

Usage:
    from src.formulations.external import register_architecture, is_external_policy
"""

import importlib

# Max total release budget shared by all single-output aggregate policies.
# ~sum of the three reservoir flood-max release limits.
_MAX_TOTAL_RELEASE_MGD = 3000.0

# Registry: name -> architecture descriptor dict
_EXTERNAL_REGISTRY: dict = {}


###############################################################################
# Registration API
###############################################################################

def register_architecture(name: str, architecture_type: str, **kwargs):
    """Register an external policy architecture.

    Args:
        name: Short name used as formulation_name (e.g. "rbf", "ann").
        architecture_type: Architecture family ("rbf", "ann", "tree").
        **kwargs: Architecture-specific parameters stored in the descriptor.
    """
    _EXTERNAL_REGISTRY[name] = {
        "type": architecture_type,
        **kwargs,
    }


def is_external_policy(formulation_name: str) -> bool:
    """Return True if the formulation uses an external policy architecture."""
    return formulation_name in _EXTERNAL_REGISTRY


def get_architecture(formulation_name: str, state_features=None):
    """Return an instantiated (un-parameterized) policy for the named architecture.

    Args:
        formulation_name: Architecture name ("rbf", "tree", "ann").
        state_features: List of feature names / dicts. If None, uses
            config.STATE_FEATURES.

    Returns:
        PolicyBase instance. Call set_params() before evaluating.

    Raises:
        ValueError: If formulation_name is not in the registry.
    """
    if formulation_name not in _EXTERNAL_REGISTRY:
        raise ValueError(
            f"Unknown architecture '{formulation_name}'. "
            f"Available: {list(_EXTERNAL_REGISTRY.keys())}"
        )
    spec = _EXTERNAL_REGISTRY[formulation_name]
    mod = importlib.import_module(spec["policy_module"])
    cls = getattr(mod, spec["policy_class"])

    from src.external_policy import build_state_config, N_STATE_TEMPORAL
    state_config = build_state_config(features=state_features)
    n_inputs = len(state_config) + N_STATE_TEMPORAL

    kwargs = {k: v for k, v in spec.items()
              if k not in ("type", "policy_class", "policy_module", "description")}
    return cls(n_inputs=n_inputs, n_outputs=1, **kwargs)


###############################################################################
# Built-in architecture registrations
###############################################################################

register_architecture(
    "rbf",
    architecture_type="rbf",
    description="Gaussian RBF policy (6 centers)",
    policy_class="RBFPolicy",
    policy_module="src.policies",
    n_rbf=6,
    output_max=_MAX_TOTAL_RELEASE_MGD,
)

register_architecture(
    "tree",
    architecture_type="tree",
    description="Oblique decision tree (depth 3)",
    policy_class="ObliqueTreePolicy",
    policy_module="src.policies",
    depth=3,
    output_max=_MAX_TOTAL_RELEASE_MGD,
)

register_architecture(
    "ann",
    architecture_type="ann",
    description="Feedforward ANN (2x8 hidden layers)",
    policy_class="ANNPolicy",
    policy_module="src.policies",
    h1=8,
    h2=8,
    output_max=_MAX_TOTAL_RELEASE_MGD,
)