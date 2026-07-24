"""scenario_discovery.py - Success/failure of a policy across the DU forcing factors.

The scenario-discovery step of MORDM (Kasprzyk et al. 2013): having re-evaluated a
policy across the held-out deeply-uncertain ensemble, show WHERE in the uncertain
forcing space it satisfices and where it fails. Here the uncertain space is the
sampled CMIP6 harmonic forcing factors theta = (m, r1, r2) that DEFINE each state
of the world (SOW), so each of the 50 SOWs is one point in theta-space, colored by
whether the policy meets all seven objective thresholds jointly in that SOW (the
same SOW-unit domain criterion used for the robustness scorecard).

This is a direct pass/fail scatter over the factor ranges -- not a boosted-tree or
PRIM box -- so the vulnerable corner of the forcing space is read straight off the
plot.

theta factors (docs/notes/methods/forcing_parameterization.md):
  m  -> log annual-mean change; e^m is the water-year volume multiplier (drier<1<wetter)
  r1 -> annual-harmonic amplitude (winter-wettening / summer-drying)
  r2 -> semiannual-harmonic amplitude (snowmelt-shoulder / bimodal shape)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

#: Interpretable axis labels for the sampled theta factors. ``m`` is shown as the
#: volume multiplier e^m (more legible than the log coordinate).
_THETA_LABEL = {
    "m": "Water-year volume multiplier  e^m\n(drier < 1 < wetter)",
    "r1": "Seasonal amplitude  r1\n(winter-wet / summer-dry)",
    "r2": "Semiannual amplitude  r2\n(snowmelt shoulder)",
}


def _policy_sow_passfail(raw, solution_id: int, within_sow_agg: str = "mean"
                         ) -> np.ndarray:
    """Boolean ``(n_sow,)``: does the policy meet ALL thresholds in each SOW?

    SOWs are in ascending SOW-id order, aligned to ``theta[::realizations_per_sow]``.
    """
    from src.robustness import collapse_within_sow, _satisfy
    cube_sow, sow_labels = collapse_within_sow(raw, within_sow_agg)     # (S, n_sow, M)
    sat = _satisfy(cube_sow, raw.base_names, raw.thresholds, raw.kinds)
    joint = sat.all(axis=2)                                            # (S, n_sow)
    sidx = raw.solution_ids.index(int(solution_id))
    return joint[sidx], list(sow_labels)


def _load_sow_theta(ensemble_dir) -> tuple[np.ndarray, list]:
    """SOW-level theta factors ``(n_sow, d)`` and their names, from the staged ensemble."""
    ensemble_dir = Path(ensemble_dir)
    npz = ensemble_dir / "forcing_profiles.npz"
    with np.load(npz, allow_pickle=True) as z:
        theta = np.asarray(z["theta_params"], dtype=float)
        names = [str(x) for x in z["theta_param_names"]]
        rpp = int(z["realizations_per_profile"])
    return theta[::rpp], names


def plot_scenario_discovery(reeval_dir, ensemble_dir, solution_id, out_file,
                            within_sow_agg: str = "mean",
                            policy_label: str = "most-robust policy",
                            figsize: tuple = (14, 4.8)) -> dict:
    """Pass/fail scatter of one policy across the DU theta-factor ranges.

    Args:
        reeval_dir: Step-08 re-eval dir (the cube).
        ensemble_dir: Staged E_test dir (``outputs/synthetic_ensembles/<tag>``),
            holding ``forcing_profiles.npz`` with the sampled theta.
        solution_id: Policy to diagnose (typically the most-robust acceptable one).
        out_file: PNG path.
        within_sow_agg: Within-SOW risk attitude (matches the scorecard).
        policy_label: Legend/title label for the policy.
        figsize: Figure size.

    Returns:
        Dict with the pass count and fraction.
    """
    from src.robustness import load_raw

    raw = load_raw(Path(reeval_dir))
    passfail, sow_labels = _policy_sow_passfail(raw, solution_id, within_sow_agg)
    theta_sow, names = _load_sow_theta(ensemble_dir)

    if theta_sow.shape[0] != passfail.shape[0]:
        # Align defensively on the shared SOW count (both are ascending SOW order).
        n = min(theta_sow.shape[0], passfail.shape[0])
        theta_sow, passfail = theta_sow[:n], passfail[:n]

    # Column m -> e^m for display; keep r1, r2 as sampled.
    disp = theta_sow.copy()
    if "m" in names:
        disp[:, names.index("m")] = np.exp(theta_sow[:, names.index("m")])

    npass = int(passfail.sum())
    n = passfail.size
    pairs = [(0, 1), (0, 2), (1, 2)] if len(names) >= 3 else [(0, 1)]

    fig, axes = plt.subplots(1, len(pairs), figsize=figsize)
    axes = np.atleast_1d(axes).ravel()
    for ax, (i, j) in zip(axes, pairs):
        ax.scatter(disp[passfail, i], disp[passfail, j], marker="o", s=55,
                   color="#2ca25f", edgecolor="white", label="satisfices (pass)",
                   zorder=3)
        ax.scatter(disp[~passfail, i], disp[~passfail, j], marker="X", s=55,
                   color="#c94a4a", alpha=0.85, label="fails", zorder=3)
        ax.set_xlabel(_THETA_LABEL.get(names[i], names[i]), fontsize=9)
        ax.set_ylabel(_THETA_LABEL.get(names[j], names[j]), fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, loc="best")

    fig.suptitle(
        f"Scenario discovery — {policy_label} (id {int(solution_id)}): "
        f"satisfices all 7 thresholds in {npass}/{n} SOWs ({100*npass/n:.0f}%)\n"
        f"each point is one deeply-uncertain state of the world in sampled "
        f"forcing-factor space",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {"solution_id": int(solution_id), "n_pass": npass, "n_sow": n,
            "pass_fraction": npass / n}
