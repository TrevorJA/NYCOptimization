"""
parameterized_salt_front_adjustment.py — DV-controllable salt-front
adjustment parameter.

Subclasses upstream `pywrdrb.parameters.salt_front_location.FlowTargetSaltFrontAdjustmentRatio`
and replaces the hardcoded multiplier dicts (lines 412-422 in upstream) with
attributes loaded from the Pywr model JSON. This lets the FFMP-family DV
registry expose the operational table to the optimizer.

Registered with Pywr under `type="NYCOptParameterizedSaltFrontAdjustmentRatio"`.

Constructor extra kwargs (compared to the parent):

- `multipliers`: dict with keys `"trenton"` and `"montague"`. Each value is
  a list of (months_tuple, [4 multipliers]) pairs ordered same as upstream.
  E.g.::

      {"trenton":  [[[12,1,2,3,4], [1, 1, 0.926, 0.926]],
                    [[5,6,7,8,9,10,11], [1.074, 1, 0.926, 0.926]]],
       "montague": [[[12,1,2,3,4], [1.185, 1, 1, 0.815]],
                    [[5,6,7,8],    [1.031, 1, 1, 0.688]],
                    [[9,10,11],    [1.1, 1, 1, 0.733]]]}

- `rm_band_thresholds`: list of 3 floats `[hi, mid, lo]`, default `[92.5, 87.0, 82.9]`,
  used to bin the salt-front position into the 4 columns of the multiplier table.
  Indexing convention matches upstream: idx 0 => sf > hi; 1 => mid < sf <= hi;
  2 => lo < sf <= mid; 3 => sf <= lo.

- `nyc_drought_emergency_level`: integer drought-level index that triggers the
  rule. Inherited from parent (default `n_drought_levels - 1`); exposed here as
  a DV when `SALT_FRONT_PARAM_MODE` includes the gate.

The activation gate semantics intentionally match the parent: the rule fires
when `int(round(drought_level_agg_nyc_idx)) == self.nyc_drought_emergency_level`.

The default values for `multipliers` and `rm_band_thresholds` match the
upstream FFMP table verbatim, so when this parameter is registered with
defaults the simulation is byte-identical to upstream.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# These imports execute only after the namespace bootstrap has been run by
# src.ts_options::_bootstrap_pywrdrb_ml_namespace, since the upstream parent
# class lives in pywrdrb's parameter module.
from pywr.parameters import load_parameter
from pywrdrb.parameters.salt_front_location import (
    FlowTargetSaltFrontAdjustmentRatio,
)


# --- Defaults match upstream FFMP table exactly --------------------------

DEFAULT_TRENTON_TABLE = [
    [[12, 1, 2, 3, 4],            [1.0, 1.0, 0.925925926, 0.925925926]],
    [[5, 6, 7, 8, 9, 10, 11],     [1.074074074, 1.0, 0.925925926, 0.925925926]],
]

DEFAULT_MONTAGUE_TABLE = [
    [[12, 1, 2, 3, 4],            [1.185185185, 1.0, 1.0, 0.814814815]],
    [[5, 6, 7, 8],                [1.03125, 1.0, 1.0, 0.6875]],
    [[9, 10, 11],                 [1.1, 1.0, 1.0, 0.733333333]],
]

DEFAULT_RM_BAND_THRESHOLDS = [92.5, 87.0, 82.9]  # hi, mid, lo


def _normalize_table(rows):
    """Convert list-of-[months, multipliers] into a tuple-keyed dict.

    Each row is `[[months...], [m0, m1, m2, m3]]`. Returns
    `{tuple(months): list(multipliers)}` for fast `month in key` lookup.
    """
    return {tuple(int(m) for m in months): [float(x) for x in vals]
            for months, vals in rows}


class NYCOptParameterizedSaltFrontAdjustmentRatio(FlowTargetSaltFrontAdjustmentRatio):
    """DV-controllable variant of the salt-front MRF adjustment parameter.

    Reads the multiplier table and RM bin thresholds from instance attributes
    set by the Pywr loader (see `load`). The activation gate matches the
    parent and uses `nyc_drought_emergency_level`.

    Behavior is byte-identical to the parent when constructed with default
    values.
    """

    def __init__(
        self,
        model,
        salinity_model,
        update_salt_front_location,
        ml_model_type,
        drought_level_agg_nyc,
        flow_target,
        nyc_drought_emergency_level: int = 6,
        multipliers: dict | None = None,
        rm_band_thresholds: list | None = None,
        **kwargs,
    ):
        super().__init__(
            model,
            salinity_model,
            update_salt_front_location,
            ml_model_type,
            drought_level_agg_nyc,
            flow_target,
            nyc_drought_emergency_level=nyc_drought_emergency_level,
            **kwargs,
        )

        if multipliers is None:
            trenton_rows = DEFAULT_TRENTON_TABLE
            montague_rows = DEFAULT_MONTAGUE_TABLE
        else:
            trenton_rows = multipliers.get("trenton", DEFAULT_TRENTON_TABLE)
            montague_rows = multipliers.get("montague", DEFAULT_MONTAGUE_TABLE)
        self._trenton_table = _normalize_table(trenton_rows)
        self._montague_table = _normalize_table(montague_rows)

        rm = rm_band_thresholds if rm_band_thresholds is not None else DEFAULT_RM_BAND_THRESHOLDS
        if len(rm) != 3:
            raise ValueError(
                f"rm_band_thresholds must have exactly 3 entries (hi, mid, lo); got {rm}"
            )
        # Sort defensively so ordering (hi > mid > lo) is preserved even if
        # the optimizer samples them in arbitrary order.
        rm_sorted = sorted([float(x) for x in rm], reverse=True)
        self._rm_hi, self._rm_mid, self._rm_lo = rm_sorted

    def value(self, timestep, scenario_index):
        # Mirror the parent's gate. Note: parent writes to ml_model.records;
        # we keep that behavior so debug logging is unchanged.
        drought_level_agg_nyc_idx = self.drought_level_agg_nyc.get_value(scenario_index)
        ml_model = self.salinity_model.ml_model
        ml_model.records["drought_idx"][ml_model.t] = drought_level_agg_nyc_idx

        if int(round(drought_level_agg_nyc_idx)) != self.nyc_drought_emergency_level:
            return 1.0

        if self.ml_model_type == "lstm":
            sf_mu = ml_model.sf_mu
        elif self.ml_model_type == "rf":
            sf_mu = ml_model.saltfront
        else:
            raise ValueError(
                f"Invalid ml_model_type='{self.ml_model_type}'; expected 'lstm' or 'rf'"
            )

        if sf_mu > self._rm_hi:
            idx = 0
        elif sf_mu > self._rm_mid:
            idx = 1
        elif sf_mu > self._rm_lo:
            idx = 2
        else:
            idx = 3

        month = int(timestep.month)
        if self.flow_target == "delTrenton":
            for months, vals in self._trenton_table.items():
                if month in months:
                    ratio = vals[idx]
                    ml_model.records["adj_ratio_Trenton"][ml_model.t] = ratio
                    return ratio
        elif self.flow_target == "delMontague":
            for months, vals in self._montague_table.items():
                if month in months:
                    ratio = vals[idx]
                    ml_model.records["adj_ratio_Montague"][ml_model.t] = ratio
                    return ratio
        else:
            raise ValueError(
                f"Invalid flow_target='{self.flow_target}'; expected 'delTrenton' or 'delMontague'"
            )
        # If we somehow miss every month bucket (should not happen with full
        # coverage), return 1.0 so the parameter doesn't return None.
        return 1.0

    @classmethod
    def load(cls, model, data: dict[str, Any]):
        """Pywr loader. Pops our extra kwargs out of `data` before delegating."""
        # Pop our extra fields first so they don't leak into the parent's load.
        multipliers = data.pop("multipliers", None)
        rm_band_thresholds = data.pop("rm_band_thresholds", None)

        # Required parent fields, mirroring upstream load() at salt_front_location.py:449.
        flow_target = data.pop("flow_target", None)
        ml_model_type = data.pop("ml_model_type", "lstm")
        nyc_drought_emergency_level = data.pop("nyc_drought_emergency_level", 6)

        salinity_model = load_parameter(model, "salinity_model")
        update_salt_front_location = load_parameter(model, "update_salt_front_location")
        drought_level_agg_nyc = load_parameter(model, "drought_level_agg_nyc")

        return cls(
            model,
            salinity_model,
            update_salt_front_location,
            ml_model_type,
            drought_level_agg_nyc,
            flow_target,
            nyc_drought_emergency_level=nyc_drought_emergency_level,
            multipliers=multipliers,
            rm_band_thresholds=rm_band_thresholds,
            **data,
        )


NYCOptParameterizedSaltFrontAdjustmentRatio.register()
