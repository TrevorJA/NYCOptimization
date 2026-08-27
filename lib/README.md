This folder holds the licensed BorgMOEA sources (serial Borg and the multi-master
variant, written in C). They are proprietary and gitignored (`lib/borg/`).

To reproduce the results, request access at [borgmoea.org](http://borgmoea.org/)
and place `borg.c`, `borg.h`, `borgmm.c`, `borgmm.h`, `mt19937ar.c`, and
`borg.py` in `lib/borg/`, then build both shared libraries the launcher loads,
`libborg.so` (serial) and `libborgmm.so` (multi-master). The compile lines are
in the top-level [README](../README.md) §1.3.
