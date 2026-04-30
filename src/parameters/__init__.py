"""
src.parameters — NYCOptimization-side custom Pywr parameters.

These are subclasses or substitutes of upstream Pywr-DRB parameters that
expose additional decision-variable surfaces for the optimization. They
are registered with Pywr at module import time so model JSONs can refer
to them by `type` string.

Modules here are imported lazily (see src.ts_options::_bootstrap_pywrdrb_ml_namespace
for the trigger). Importing this package alone does NOT pull in pywrdrb;
each submodule imports its own pywrdrb dependencies.
"""
