"""
src/policies/__init__.py - Policy architecture base class and imports.
"""

from abc import ABC, abstractmethod
import numpy as np


class PolicyBase(ABC):
    """Base class for all policy architectures."""

    @abstractmethod
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """Evaluate policy: state vector -> action vector."""
        ...

    @abstractmethod
    def set_params(self, flat_vector: np.ndarray) -> None:
        """Set parameters from a flat decision variable vector (for Borg)."""
        ...

    @abstractmethod
    def get_bounds(self) -> tuple:
        """Return (lower_bounds, upper_bounds) arrays."""
        ...

    @property
    @abstractmethod
    def n_params(self) -> int:
        """Number of decision variables."""
        ...

    @property
    @abstractmethod
    def n_inputs(self) -> int:
        """Dimension of the state vector this policy expects."""
        ...

    @property
    @abstractmethod
    def n_outputs(self) -> int:
        """Dimension of the action vector."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Architecture identifier string."""
        ...


from .rbf_policy import RBFPolicy
from .tree_policy import ObliqueTreePolicy
from .ann_policy import ANNPolicy

__all__ = ["PolicyBase", "RBFPolicy", "ObliqueTreePolicy", "ANNPolicy"]
