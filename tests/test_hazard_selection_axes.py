"""The campaign hazard-selection axis set (m = 6, nested-P diagnostic) stays coherent.

Guards the config default against drift: every configured selection axis must be
a real candidate hazard metric, and the two axes the nested-P diagnostic dropped
(their per-axis tail enrichment is geometry-limited at any affordable pool size)
must stay out unless the env override is used deliberately.
"""

import config


def test_campaign_selection_axes_are_valid_candidates():
    from scengen.hazard_metrics import CANDIDATE_EVENT_METRICS

    assert len(config.HAZARD_SELECTION_AXES) == 6
    assert set(config.HAZARD_SELECTION_AXES) <= set(CANDIDATE_EVENT_METRICS)
    assert len(set(config.HAZARD_SELECTION_AXES)) == 6  # no duplicates


def test_dropped_axes_stay_out_of_the_default():
    assert "drought_duration" not in config.HAZARD_SELECTION_AXES
    assert "flood_rise_rate" not in config.HAZARD_SELECTION_AXES
