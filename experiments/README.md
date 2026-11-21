# Experiments

Each folder holds one paper experiment, set up so it can be reproduced. Experiments live outside
`src/` and are not part of the `nlb2` package. They use the installed `nlb2` (`pip install -e .` from the
repository root), and each one pins its other dependencies.

Every experiment folder contains:

- a `README.md` with one command per stage and approximate runtimes;
- the code and its frozen protocol;
- a committed record of the results, small enough for git;
- `make verify`, which checks that record without data or a GPU.

Large inputs and intermediates are downloaded or regenerated, never committed.

| Experiment | Paper | Status |
|---|---|---|
| [`figure1_lfads_vs_mtm`](figure1_lfads_vs_mtm) | Figure 1 and its appendix: LFADS vs MtM under linear, MLP and transformer readouts (co-smoothing, wheel speed, whisker, paws, licking) | Locked; test evaluated once (2026-11-21) |
