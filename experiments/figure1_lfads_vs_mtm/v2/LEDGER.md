# Figure 1 v2 ledger

The narrative record of decisions for the v2 readout experiment, in order. The machine-generated
record of every fitted variant is `v2/ledger.csv` (from `code/select_lock_v2.py ledger`).

## 2026-11-20: goal

The question is whether the readout determines the LFADS-vs-MtM result. The experiment compares
co-smoothing (pooled co-bps, MAE) and wheel-speed R² under linear, MLP and transformer readouts,
on all 29 BWB eval sessions × 3 seeds, using paired tests and a representation × readout
interaction test. Rules: readouts are selected on validation data only; the configuration is
locked; test metrics are computed once.

## Pretraining-overlap check (rule 1)

- `eval_recording_ids.txt` (29 sessions) is identical to BWB `ibl_brain_wide_bench_2026/data/eval_eids.txt`.
- None of the 29 sessions appear in the published MtM pretraining list, `vendor/ibl-bwb/src/core/data/pretrain_recording_ids.txt` (423 sessions; identical to `pretrain_eids.txt`).
- None of the 13 eval mice appear among the 126 pretraining mice (checked via `bwm_qc.csv`).
- Checkpoint `checkpoints/mtm.pt` (SHA-256 `cb107873…7721`, verified):
  - It has 423 per-session `in_stitcher`/`out_stitcher` entries. These are exactly the 423 pretraining sessions.
  - `session_emb.vocab` holds those 423 plus an `NA` padding token.
  - None of the 29 eval sessions appear in either.
- `mtm_model.MtMNeural` also asserts that each eval session is absent from the checkpoint vocabulary at load time.

## What had been seen before the v2 protocol existed (disclosure)

- **v1 Stage 2, seed 0.** Test R² for the baseline transformer (`d2_w256_h4`), linear and MLP readouts was aggregated over 7 sessions and displayed in this session before v2 was designed. LFADS: 0.05 / 0.15 / 0.24, MtM: 0.05 / 0.13 / 0.17 (linear / MLP / transformer). v1 also computed test R² for all 8 transformer screen configs on the screen sessions. Those per-config numbers were not displayed.
- **v1 Stage 1 native co-smoothing, seed 0, all 29 sessions.** These results were known from `HANDOFF.md`: LFADS MAE 6.89 Hz vs MtM 7.28 Hz.
- **How v2 handles this.** Every v2 readout is refit from scratch in `v2/fits/`. No v1 readout output is read. Selection uses validation metrics only. The v1 test numbers above played no part in choosing the v2 search spaces, which keep v1's architectures and add the MLP sizes and GLM co-smoothing readout listed below.

## Changes from v1 (all fixed before any v2 fit)

1. **Fresh outputs and a separate test step.** Fits write validation metrics only. `evaluate_v2.py` is the only code that reads test data. It runs once, after `lockfile.json`, and refuses to run again or if any hash changed.
2. **Normalization fix** (HANDOFF §5). The per-channel scale is floored at 10% of the median positive std. v1 only guarded std below 1e-6, which let the smoothed-spikes MLP diverge (R² as low as −35).
3. **Co-smoothing readouts are new.** Readouts map held-in-driven frozen features to held-out counts with a Poisson loss. The linear readout is a Poisson GLM; MLP and transformer use the same families as for wheel. The Stage 1 native readouts (LFADS's own output layer, MtM's `out_stitcher`) are reported as reference rows.
4. **The architecture screen is cross-fitted (rule 5).** v1 screened on `sorted(eids)[:8]`, which are 8 of the 29 reported sessions. That contradicts the methods text ("development sessions that do not appear in any reported result"). v2 fits every architecture at seed 0 on every session (validation only). Each session's architecture is then chosen from the other 12 mice's validation scores, so neither a session nor its mouse chooses its own architecture. The literal text cannot be met while reporting all 29 sessions, because every other BWB session is in MtM's pretraining set (rule 1). The methods sentence needs a one-line edit (see `v2/paper_methods_patch.md`).
5. **MLP architectures are screened the same way** (6 configs; v1 used a single fixed config). The transformer keeps v1's 8 configs.
6. **Ridge α grid:** 17 log-spaced values from 1e-4 to 1e4, matching the paper text. v1 used 11 values from 1e-6 to 1e4.

## Environment

- **Seed 0 Stage 1** (all 29 sessions × 3 models) is from the original laptop run (CPU, torch 2.15.0).
- **Seeds 1–2 Stage 1** run on HAL: MtM and MtM-random on CUDA, LFADS on CPU, torch 2.14.1+cu130. torch 2.15.0 has no cu130 wheel.
- **All v2 readouts** run on HAL (RTX 4090, CUDA). The Poisson GLM uses float32 on GPU. A float64 smoke test gave the same validation co-bps (0.3742 vs 0.3743).

## Known mismatches between the paper text and the code (not changed, per rule 4; text needs edits)

- **Held-out units.** The text says 25%, stratified by region and firing-rate quartile. The code holds out about 20%, stratified by region only (SHA-256 mask, `figure1_data.py`).
- **Readout hyperparameter selection.** The text says 5-fold CV over training trials. The code uses the BWB TS1 causal validation split.
- **Search budgets.** The text says 32 configurations per readout. The code tunes 4 optimizer settings per architecture, plus the cross-fitted architecture screen.

## Log

- 2026-11-20 21:47: HAL setup done; checkpoint hash verified; vendored code re-synced after an rsync exclude bug.
- 2026-11-20 21:50: Stage 1 seeds 1–2 started (`orchestrate.py`, 14 workers).
- 2026-11-20 21:53: GLM switched from float64 to float32 on GPU (96 s to 9.5 s per fit; same validation co-bps). The smoke-test fits were deleted before the screen started.
- 2026-11-20 21:55: Seed-0 screen started (`orchestrate_v2.py screen`, 8 workers). Protocol SHA-256 `c8fa23bc…`.
- 2026-11-20 21:58: `evaluate_v2.py` dry run on the validation split. 69 readouts re-scored; the largest difference from the recorded validation metric is 3.8e-9.
- 2026-11-20 22:10: Stats script checked on synthetic data in a scratch dir (v2/ untouched). It recovers a planted interaction for wheel and correctly finds none for a constant co-bps gap.
- 2026-11-20 22:15: **Extension frozen (rule 6)** as `code/protocol_ext.py`, SHA-256 `8d136fd8…`, before any v2 test metric existed. Tasks: whisker motion energy, right paw speed, left paw speed, licking rate. All 29 sessions pass BWB behavior QC for all four, with no NaNs. Readouts and screen are identical to wheel. Representations: lfads, mtm, mtm_random. It has its own lock and test evaluation (`*_ext` files). Choice, contrast and reward are excluded because they are sequence-level classification targets.
- 2026-11-20 22:18: Extension smoke test on one session, all four tasks (fits deleted). The extension screen is queued to start after the primary screen (`chain_ext_screen.sh`).
- 2026-11-20 22:15: **Performance-only change** to `readouts_v2.fit_net`. Host-device syncs went from two per minibatch to one per epoch: the permutation moves to the GPU once per epoch, the loss accumulates on the GPU, and the finiteness check runs at epoch end. It uses the same random permutation and the same update order. Checked by refitting 3 existing screen fits in a scratch dir: validation R² identical to 6 decimal places, and the selected epoch identical for all 12 optimizer runs. It is 1.2–1.7× faster per fit. Fits launched before 22:15 record the earlier `code_sha256` (`a4c4dfb5…`); later fits record `f1139854…`.
- 2026-11-20 22:15: **Considered and rejected:** shortening early stopping (min 40 / patience 30). Simulating shorter rules on 860 recorded optimizer histories changed the selected epoch in 4–12% of runs. The changes were concentrated in slow-learning runs (lr 3e-4, larger readouts), so a shorter rule would handicap the complex readouts. **Also rejected:** LFADS Stage 1 on GPU, which took 10.2 s/epoch vs 5.0 s/epoch on 2 CPU threads.
- 2026-11-20 22:17: Added a running-elsewhere guard to `orchestrate_v2.py`. The screen orchestrator is queued to restart with 20 workers once Stage 1 frees the CPU (`chain_more_workers.sh`).
- 2026-11-20 22:25: Hand-off to HAL so the laptop can sleep. `chain_primary_seeds.sh` runs select → seeds 1–2 → ledger after the screen; `chain_ext_seeds.sh` does the same for the extension. Both stop before `lock`. Lock, the single test evaluation and stats are run from the Claude session after the validation selection has been reviewed.
- 2026-11-20 22:35: Added `code/plot_figure1_v2.py`. Panel a: native co-smoothing map averaged over 3 seeds. Panel b: co-bps by readout. Panel c: wheel R² by readout. Bars are LFADS and MtM side by side, with Holm p per readout and the interaction p. The layout was checked on synthetic stats in HAL `/tmp`. The real figure will be rendered only from locked v2 outputs. The original `plot_figure1.py` is unchanged.
- 2026-11-20 23:37: Stage 1 seeds 1–2 complete (87/87 each, 0 failures). Screen at 2445/3480 fits with 20 workers, 0 failures.
- 2026-11-20 23:40: **Interim validation-only check** (seed 0; no test data read).
  - Primary contrast: LFADS is ahead of MtM under every readout screened so far. Wheel validation R² gap is 0.04–0.08. Cosmooth validation co-bps gap is about 0.02 (transformers).
  - Control issue: the smoothed-spikes MLP readout for wheel is unstable on a few sessions. Example: session `29a6def1`, `mlp_d1_w128`, validation R² −3.2, best epoch 1. This looks like train→validation drift in raw unit rates under the causal split; the transformer on the same inputs is stable (≈0.29).
  - No change made. `smoothed_spikes` is a secondary control. Its MLP architecture is still chosen by the pre-registered cross-fitted rule. Changing the protocol now would be validation-driven tuning of a control.
- 2026-11-20 23:53: Seed-0 screen finished: 3480/3480 fits, 0 failed. Cross-fitted selection written to `v2/selection.json` (log `v2_select.txt`). Seeds 1–2 started with 12 workers. Extension screen started with 8 workers.
- 2026-11-21 01:40: **Selection review** (validation only).
  - The deepest transformer, `d6_w256_h4`, is chosen for most cells. This is the edge of the pre-registered grid.
  - The top 2–4 transformer configs per cell are within about 0.002–0.005 validation R² or co-bps of each other, which looks like a plateau rather than an unfinished trend.
  - The grid is not extended. Doing so would change the frozen protocol after the screen had been seen.
  - This is a limitation to disclose: deeper readouts were not tried.
- 2026-11-21 03:27: Seeds 1–2 finished: 464 jobs (1392 fits), 0 failed. `v2/ledger.csv` written.
- 2026-11-21 03:28: **Pre-lock checks.**
  - Code on HAL is byte-identical to the laptop copy for all 9 v2/ext files.
  - Every primary seed 1–2 fit has finite validation metrics. The only validation values below −1 are 6 seed-0 smoothed-spikes MLP screen variants, already noted at 23:40.
  - Rule 1 re-checked at lock time: 0/29 eval sessions in the 423 pretraining sessions; 0 eval eids in the checkpoint keys.
- 2026-11-21 03:28: **LOCKED.** `v2/lockfile.json`, SHA-256 `a32ad9141a04…`.
  - Covers 2088 readouts: 2 tasks × 4 representations × 29 sessions × 3 seeds × {linear, selected MLP, selected transformer}.
  - Protocol `c8fa23bc…`, selection `658da532…`.
- 2026-11-21 03:37: **TEST EVALUATED ONCE** (`evaluate_v2.py`; 8 min 51 s on HAL).
  - 2349 rows: 2088 locked readouts plus 261 Stage 1 native co-smoothing rows.
  - `test_metrics.csv` SHA-256 `e337b40c…`, `evaluate_v2.py` SHA-256 `4b15cef1…`.
- 2026-11-21 03:38: `stats_v2.py` produced `final_table.md`, `final_stats.json` and `session_values.csv`. `plot_figure1_v2.py` produced `figures/figure1_v2.{png,pdf}`. Nothing in protocol, splits, metrics, held-out units or readouts was changed after test.
- 2026-11-21 03:56: Extension seeds 1–2 finished: 696 jobs (2088 fits), 0 failed.
  - Extension selection is in `v2_ext_select.txt`. As in the primary experiment, the deepest transformer (`d6_w256_h4`) is often chosen, with a flat top of the ranking.
- 2026-11-21 03:57: **Bookkeeping fix (no effect on any result).**
  - `ext.py ledger` wrote to `v2/ledger.csv` because `select_lock_v2.ledger()` hard-codes that path, which overwrote the primary per-fit ledger on HAL.
  - Fix: the file was moved to `v2/ledger_ext.csv` (7308 fits, 42804 runs). The primary ledger was regenerated with `select_lock_v2.py ledger` (4872 fits, 25752 runs). The regenerated file is byte-identical (SHA-256 `fbbbc4ee…`) to the copy saved to the laptop at 03:40.
  - Neither lockfile hashes the ledger CSVs. Do not re-run `ext.py ledger` without moving the file afterwards.
- 2026-11-21 03:58: Extension pre-lock check: all 7308 extension fits have finite validation R² ≥ −1.
- 2026-11-21 03:58: **Extension LOCKED.** `v2/lockfile_ext.json`, SHA-256 `ee60e5f2…`, covering 3132 readouts (4 tasks × 3 representations × 29 sessions × 3 seeds × 3 families). Combined protocol hash `f2605033…` (`protocol_v2.py` + `protocol_ext.py`).
- 2026-11-21 03:59: **Extension TEST EVALUATED ONCE.** 3132 rows; `test_metrics_ext.csv` SHA-256 `1a96bcf8…`. `ext.py stats` produced `final_table_ext.md` and `final_stats_ext.json`. Nothing was changed after test.
- 2026-11-21 04:05: Copying results to the laptop briefly pulled part of `v2/fits_ext` (because of a wildcard). That partial copy was verified to be a subset of HAL's files and deleted. HAL holds all fits (`v2/fits` 18 GB, `v2/fits_ext`) and test predictions. The laptop holds the locks, metrics, stats, ledgers and figure, and these match HAL by SHA-256.
- 2026-11-21 04:05: **Exploratory observation (not pre-registered; not a confirmatory claim).**
  - Across the 7 outcomes, the MtM − LFADS gap is smaller in magnitude under the transformer than under linear in 6 (co-bps, MAE, whisker, rpaw, lpaw, licks). Wheel is the exception.
  - No per-outcome interaction test is significant after Holm, and the outcomes share sessions and representations, so this pattern is descriptive only.
- 2026-11-21 13:55: **Paper figure and appendix (presentation only; no fits, metrics or tests changed).**
  - `plot_figure1_v2.py` now uses the original Figure 1 design (`plot_figure1.py` layout, 3.25 × 4.25 in) filled with the locked results.
    - Panel a: native co-smoothing map averaged over 3 seeds.
    - Panel b: wheel-speed R² (v2 lock) and licking-rate R² (extension lock) as two side-by-side plots, with Holm p per readout and the interaction p. Bar means are asserted to equal the stats files.
    - The IBL task cartoon was dropped to make room.
  - μ_pop now shows ± s.e. from a hierarchical bootstrap (mice → sessions → units; 10,000 reps, seed 20261120): LFADS 7.32 ± 0.90 Hz, MtM 7.68 ± 1.00 Hz. The naive s.e.m. over 757 units (±0.34 / ±0.36) treats units as independent and was not used.
  - `paper_assets_v2.py` writes `benchmark_paper/figures/figure1_v2/` (figure, caption, 3 appendix tables, `SOURCES.json` with input hashes).
    - The caption is the original Figure 1 caption plus three additions: s.e. on μ_pop, licking rate in (b), and the p-value definitions.
  - `paper.tex` now points Figure 1 at `figures/figure1_v2/`; the October package in `figures/figure1_experiment/` is untouched.
  - New appendix section "Figure 1" (`app:figure1-results`) with draft text and tables. The figure stays on page 1.
- 2026-11-21 14:10: **Figure revisions per Jon (presentation only).**
  - μ_pop uses the s.e.m. across all 757 held-out units: LFADS 7.32 ± 0.33 Hz (0.3347), MtM 7.68 ± 0.36 Hz (0.3630).
    - This supersedes the 13:55 entry. The hierarchical-bootstrap s.e. (0.90 / 1.00 Hz) is not shown; it is noted here for reference.
  - Panel b: wheel speed (top) and licking rate (bottom) are stacked, smaller, on the left. The IBL task cartoon is back on the right, at its original position.
  - Caption: `paper.tex` again inputs the original, unedited `figures/figure1_experiment/publication/figure1_caption.tex`. Jon will write the caption. `paper_assets_v2.py` no longer generates a caption.
  - Figure 1 stays on page 1.
