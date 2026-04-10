"""
external.py - External policy architecture registry.

Tracks formulations that use learned or parameterized external policy
architectures (RBF, ANN, decision-tree, etc.) rather than the FFMP
rule structure. Currently no such formulations are registered; this
module is the extension point when they are added.

Usage:
    from src.formulations.external import register_architecture, is_external_policy

    # Register a new formulation with an RBF architecture
    register_architecture(
        "rbf_6obj",
        architecture_type="rbf",
        n_inputs=5,
        n_outputs=3,
        n_centers=10,
    )
"""


# Registry: formulation_name -> architecture descriptor dict
_EXTERNAL_REGISTRY: dict = {}


def register_architecture(formulation_name: str, architecture_type: str, **kwargs):
    """Register an external policy architecture for a formulation.

    Args:
        formulation_name: Name of the formulation (must also appear in FORMULATIONS).
        architecture_type: Architecture family string, e.g. "rbf", "ann", "tree".
        **kwargs: Architecture-specific parameters (n_inputs, n_outputs, etc.).
    """
    _EXTERNAL_REGISTRY[formulation_name] = {
        "type": architecture_type,
        **kwargs,
    }


def is_external_policy(formulation_name: str) -> bool:
    """Return True if the formulation uses an external policy architecture.

    FFMP-style formulations that map DVs directly to NYCOperationsConfig
    parameters return False. RBF/ANN/tree formulations return True.

    Args:
        formulation_name: Formulation name to check.

    Returns:
        bool
    """
    return formulation_name in _EXTERNAL_REGISTRY


def get_architecture(formulation_name: str) -> dict | None:
    """Return the architecture descriptor for a formulation, or None.

    Args:
        formulation_name: Formulation name.

    Returns:
        Dict with at least a "type" key, or None if not an external policy.
    """
    return _EXTERNAL_REGISTRY.get(formulation_name)
