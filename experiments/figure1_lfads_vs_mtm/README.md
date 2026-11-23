# Figure 1: does the readout decide LFADS vs MtM?

The NLB2 paper's Figure 1 compares a task-specific latent model, LFADS (from NLB2), with a foundation
model, the released BrainWideBench MtM, on 29 held-out IBL sessions from 13 mice, with 3 seeds each.

1. **Stage 1.** Each model learns a representation of the session from neural data only. LFADS is fit per
   session. MtM keeps its pretrained backbone frozen and learns only new-session stitchers. A
   randomly initialised MtM is included as a control.
2. **Stage 2.** The frozen representations are read out by linear, MLP and transformer decoders for:
   - co-smoothing (held-out neurons: co-bps and MAE);
   - wheel speed;
   - four further BrainWideBench TS1 behaviours: whisker motion energy, right and left paw speed, and
     licking rate.

Readouts were chosen on validation data only and locked before a single test evaluation. Every
variant that was fitted is listed in the ledgers.

This folder holds the code, the locked record, and the steps to verify or reproduce it.

- Results: [`v2/final_table.md`](v2/final_table.md) and [`v2/final_table_ext.md`](v2/final_table_ext.md).
- Decision log: [`v2/LEDGER.md`](v2/LEDGER.md).

## Quick start: check the record (no data, no GPU, about 1 minute)

```sh
pip install -e ../..                    # NLB2 itself; provides torch, numpy, scipy, h5py
pip install matplotlib pillow omegaconf
cd experiments/figure1_lfads_vs_mtm
make verify                             # lock integrity + recomputed statistics, see below
make figure                             # figures/figure1_v2.{pdf,png} from the committed record
make paper PAPER_DIR=<paper>/figures/figure1_v2   # figure + appendix tables for the paper
```

`make verify` runs `code/verify.py`, which checks:

- that every locked code file, both protocols and both selections still match the SHA-256 recorded in
  `v2/lockfile.json` and `v2/lockfile_ext.json`;
- that the single test evaluation (`v2/evaluated*.json`) used those lockfiles and produced the
  committed `v2/test_metrics*.csv`;
- that rerunning the statistics from those metrics reproduces `v2/final_stats*.json`, `v2/final_table*.md`
  and `v2/session_values.csv`;
- that the ledgers list every fitted readout.

With the inputs fetched, it also re-checks that none of the 29 sessions is in MtM's pretraining set.

## Full re-run

You need one CUDA GPU, about 8 GB of inputs and about 70 GB of free disk for intermediate files.
Recreate the reference environment (Python 3.14) first. `--no-deps` installs NLB2 on top of the
pinned versions without changing any of them:

```sh
pip install -r environment/requirements-hal.txt --extra-index-url https://download.pytorch.org/whl/cu130
pip install --no-deps -e ../..
```

Then:

```sh
make fetch          # 29 sessions (S3), MtM checkpoint (Hugging Face, SHA-256 checked), ibl-bwb at the pinned commit
make clean-record   # lock/evaluate refuse to overwrite the committed record; `git checkout -- v2 results` restores it
make all            # stage1 -> primary (screen, select, seeds, ledger, lock, evaluate, stats) -> extension -> figure
git diff --stat v2  # compare the new run with the committed record
```

The targets can also be run one at a time; see the `Makefile`. Approximate times on the reference
machine (RTX 4090, i9-14900KF, 32 threads), with stages overlapping as they did in the original run:

| Step | Wall clock |
|---|---|
| Stage 1, seeds 1–2 (174 fits) | ~1.7 h |
| Primary screen, seed 0 (3,480 readouts) | ~2 h |
| Primary seeds 1–2 (1,392 readouts) | ~3.5 h |
| Extension screen + seeds (7,308 readouts) | ~4 h |
| Test evaluation | 9 min (primary), ~1 min (extension) |

Seed 0 of Stage 1 was run earlier on a laptop CPU (`environment/requirements-laptop-stage1-seed0.txt`),
and its time was not recorded. A re-run's neural-network readouts can differ slightly across GPUs and
library versions. The ridge and GLM readouts and the statistics are deterministic.

## How the record was produced (rules)

1. **No pretraining overlap.** None of the 29 sessions or 13 mice is in MtM's 423-session pretraining
   set. The checkpoint's per-session stitchers match that list exactly. `mtm_model.py` asserts this
   at load time.
2. **Validation-only selection.**
   - `protocol_v2.py` and `protocol_ext.py` were frozen before any fit.
   - Architectures are chosen per session by leave-one-mouse-out cross-fitting on seed-0 validation
     scores.
   - Optimizer settings are chosen per fit on validation data.
3. **Lock, then test once.**
   - `select_lock_v2.py lock` hashes the protocol, the code, the selection and every readout's weights.
   - `evaluate_v2.py` is the only code that reads test data. It verifies the lock first and refuses to
     run a second time.
4. **Everything logged.** `v2/ledger.csv.gz` (primary) and `v2/ledger_ext.csv.gz` (extension) list
   every optimizer and penalty setting of every readout fitted. `v2/LEDGER.md` is the narrative log,
   including what was seen before the protocol existed.
5. **Nothing changed after test.** Splits, metrics and held-out units are fixed in the locked code.

The locked files in `code/` must stay byte-identical, or `make verify` fails. Change behaviour only
in unlocked files, such as the plotting scripts and the Makefile. One example:
`ext.py ledger` writes to the primary ledger's path, so the `ext-ledger` target moves files around it
instead of editing `ext.py`.

## Layout

| Path | Contents |
|---|---|
| `code/` | Stage 1 (`stage1_represent.py`, `lfads_model.py`, `mtm_model.py`, `orchestrate.py`); readouts (`protocol_v2.py`, `readouts_v2.py`, `orchestrate_v2.py`, `select_lock_v2.py`, `evaluate_v2.py`, `stats_v2.py`, `dryrun_val_v2.py`); extension (`protocol_ext.py`, `ext.py`); outputs (`plot_figure1_v2.py`, `paper_assets_v2.py`); `verify.py` |
| `v2/` | Committed record: selections, lockfiles, test evaluation receipts, test metrics, statistics, tables, ledgers, `LEDGER.md`, `paper_methods_patch.md` |
| `results/<method>/<eid>/seed<k>/neurons.csv` | Stage 1 native co-smoothing error per held-out neuron (map in panel a) |
| `assets/` | Atlas slice and task illustration used in the figure; see `assets/SOURCES.md` |
| `environment/` | Exact package versions: `requirements-hal.txt` (all readouts, Stage 1 seeds 1–2) and `requirements-laptop-stage1-seed0.txt` |
| `eval_recording_ids.txt`, `data_sizes.txt` | The 29 BrainWideBench evaluation sessions and their file sizes |

Not committed, because they are too large: Stage 1 latents, rates and checkpoints (12 GB), readout
weights (`v2/fits`, 18 GB; `v2/fits_ext`, 25 GB) and test predictions (1.2 GB). Their SHA-256 hashes
are in the lockfiles, so an archived copy can be checked against this record.

## Pinned versions

- NLB2 commit `5012657` for LFADS (Stage 1 used an unmodified snapshot of it).
- `brainbench-org/ibl-bwb` commit `d30ec9c` for the MtM model code and data utilities.
- MtM checkpoint `nerdslab/ibl-bwb-mtm`, SHA-256 `cb107873…7721`.
- Reference machine: Python 3.14.4, torch 2.14.1+cu130, NVIDIA driver 595.91.07.

The `environment/requirements-*.txt` files are `pip freeze` records of the two machines and are
kept exactly as recorded. They do not list NLB2, because Stage 1 imported the source snapshot
rather than an installed package. They also do not list nlb-tools, which Figure 1 does not use.
Their pandas 3.1.1 satisfies NLB2's core requirements. Only NLB2's optional `nlb` extra keeps
pandas below 3, for nlb-tools.
