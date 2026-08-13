# Grouped-epsilon re-assessment on the production ensemble fronts

Substrate: cross-seed raw unions (draw 0 / seed 1, 500k NFE) for historic, fixed_probabilistic, hazard_filling_stationary. `adj110` = retained size x1.1 (measured cross-seed live-search inflation); the target band 1000-1200 applies to the LARGEST design (the re-eval cost binder — front sizes differ ~2.5x across designs). `overcoarse` flags (ADVISORY, per design) mark candidates whose occupied 1-D box count on some axis falls below min(8, the adopted vector's count on that axis) — judge them against the per-axis coverage tables.

| candidate | eps_rel | eps_def | eps_flood | eps_storage | size_historic | adj110_historic | size_fixed_probabilistic | adj110_fixed_probabilistic | size_hazard_filling_stationary | adj110_hazard_filling_stationary | max_adj110 | fineness | in_band_max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| adopted |  |  |  |  | 2604 | 2864 | 6170 | 6787 | 5544 | 6098 | 6787 |  | False |
| grouped_base | 0.02 | 5.0 | 0.3 | 5.0 | 2152 | 2367 | 5384 | 5922 | 5009 | 5510 | 5922 | 1.000 | False |
| rel_x2 | 0.04 | 5.0 | 0.3 | 5.0 | 1059 | 1165 | 2675 | 2943 | 2274 | 2501 | 2943 | 1.189 | False |
| rel_x3 | 0.06 | 5.0 | 0.3 | 5.0 | 627 | 690 | 1755 | 1931 | 1390 | 1529 | 1931 | 1.316 | False |
| rel_x4 | 0.08 | 5.0 | 0.3 | 5.0 | 384 | 422 | 1205 | 1326 | 918 | 1010 | 1326 | 1.414 | False |
| rel_x5 | 0.1 | 5.0 | 0.3 | 5.0 | 334 | 367 | 948 | 1043 | 734 | 807 | 1043 | 1.495 | True |
| def_7.5 | 0.02 | 7.5 | 0.3 | 5.0 | 1212 | 1333 | 3360 | 3696 | 3137 | 3451 | 3696 | 1.107 | False |
| def_10 | 0.02 | 10.0 | 0.3 | 5.0 | 958 | 1054 | 2671 | 2938 | 2462 | 2708 | 2938 | 1.189 | False |
| flood_0.5 | 0.02 | 5.0 | 0.5 | 5.0 | 1736 | 1910 | 3578 | 3936 | 3557 | 3913 | 3936 | 1.136 | False |
| flood_0.6 | 0.02 | 5.0 | 0.6 | 5.0 | 1736 | 1910 | 5384 | 5922 | 3600 | 3960 | 5922 | 1.189 | False |
| stor_7.5 | 0.02 | 5.0 | 0.3 | 7.5 | 1399 | 1539 | 3733 | 4106 | 4015 | 4416 | 4416 | 1.107 | False |
| stor_10 | 0.02 | 5.0 | 0.3 | 10.0 | 1288 | 1417 | 3432 | 3775 | 3787 | 4166 | 4166 | 1.189 | False |
| joint_x1.5 | 0.03 | 7.5 | 0.45 | 7.5 | 524 | 576 | 901 | 991 | 1369 | 1506 | 1506 | 1.500 | False |
| joint_x2 | 0.04 | 10.0 | 0.6 | 10.0 | 224 | 246 | 788 | 867 | 515 | 566 | 867 | 2.000 | False |
| joint_x3 | 0.06 | 15.0 | 1.0 | 15.0 | 68 | 75 | 114 | 125 | 219 | 241 | 241 | 3.080 | False |
| rel_x2.5 | 0.05 | 5.0 | 0.3 | 5.0 | 740 | 814 | 2094 | 2303 | 1736 | 1910 | 2303 | 1.257 | False |
| mixed_r2 | 0.04 | 7.5 | 0.3 | 5.0 | 610 | 671 | 1615 | 1777 | 1367 | 1504 | 1777 | 1.316 | False |
| mixed_r2.5 | 0.05 | 7.5 | 0.5 | 7.5 | 262 | 288 | 689 | 758 | 614 | 675 | 758 | 1.750 | False |
| mixed_r3 | 0.06 | 7.5 | 0.5 | 7.5 | 232 | 255 | 618 | 680 | 450 | 495 | 680 | 1.831 | False |
| mixed_r2f | 0.04 | 7.5 | 0.5 | 5.0 | 487 | 536 | 1173 | 1290 | 980 | 1078 | 1290 | 1.495 | False |
| mixed_r2fs | 0.04 | 7.5 | 0.5 | 7.5 | 342 | 376 | 864 | 950 | 753 | 828 | 950 | 1.655 | False |
| mixed_r2.5f | 0.05 | 7.5 | 0.5 | 5.0 | 352 | 387 | 952 | 1047 | 779 | 857 | 1047 | 1.581 | True |
| keepf_a | 0.05 | 10.0 | 0.3 | 5.0 | 335 | 369 | 991 | 1090 | 784 | 862 | 1090 | 1.495 | True |
| keepf_b | 0.06 | 7.5 | 0.3 | 5.0 | 352 | 387 | 1105 | 1216 | 816 | 898 | 1216 | 1.456 | False |
| keepf_c | 0.05 | 7.5 | 0.3 | 5.0 | 426 | 469 | 1287 | 1416 | 1089 | 1198 | 1416 | 1.392 | False |
| keepf_d | 0.06 | 10.0 | 0.3 | 5.0 | 292 | 321 | 818 | 900 | 632 | 695 | 900 | 1.565 | False |

Selection rule (finest candidate whose largest-design adjusted size is in band): **keepf_a**. Decision deferred — epsilon steers the live search, so adopted-vector archives will run ~10-35% larger than this static re-filter predicts.
