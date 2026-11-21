# Figure 1: LFADS vs MtM under linear / MLP / transformer readouts. See README.md.
#
# No data needed:   make verify | make figure | make paper PAPER_DIR=<dir>
# Full re-run:      make fetch, then make clean-record, then make all (one CUDA GPU; about 6 h on the reference machine)

PY ?= python
WORKERS ?= 10
STAGE1_WORKERS ?= 14
EXT_WORKERS ?= 8
PAPER_DIR ?= figures/paper

export PYTHONPATH := $(CURDIR)/vendor/ibl-bwb/src:$(CURDIR)/code
export PYTHONWARNINGS := ignore
export FIG1_DEVICE ?= cuda

.PHONY: help verify figure paper fetch stage1 screen select seeds ledger lock evaluate stats \
        ext-screen ext-select ext-seeds ext-ledger ext-lock ext-evaluate ext-stats primary extension all clean-record

help:
	@sed -n '1,5p' Makefile

# ---- from the committed record (no data, no GPU) -------------------------------------------------------
verify:
	$(PY) code/verify.py

figure:
	$(PY) code/plot_figure1_v2.py

paper: figure
	$(PY) code/paper_assets_v2.py --paper-dir $(PAPER_DIR)

# ---- full re-run ------------------------------------------------------------------------------------
fetch:
	sh fetch_inputs.sh

# Stage 1 representations, seeds 0-2. LFADS on CPU (faster than GPU for this model), MtM on the GPU.
stage1:
	FIG1_DEVICE=cpu FIG1_DEVICE_MTM=cuda FIG1_DEVICE_MTM_RANDOM=cuda FIG1_WORKERS=$(STAGE1_WORKERS) \
	  $(PY) code/orchestrate.py lfads,mtm,mtm_random

# Primary readouts (wheel speed, co-smoothing). Validation data only until `evaluate`.
screen:
	FIG1_WORKERS=$(WORKERS) $(PY) code/orchestrate_v2.py screen
select:
	$(PY) code/select_lock_v2.py select
seeds:
	FIG1_WORKERS=$(WORKERS) $(PY) code/orchestrate_v2.py seeds
ledger:
	$(PY) code/select_lock_v2.py ledger
	gzip -9 -n -f -k v2/ledger.csv
lock:
	$(PY) code/select_lock_v2.py lock
evaluate:
	$(PY) code/evaluate_v2.py
stats:
	$(PY) code/stats_v2.py

# Extension readouts (whisker, paws, licking), locked and tested separately.
ext-screen:
	FIG1_WORKERS=$(EXT_WORKERS) $(PY) code/ext.py orchestrate screen
ext-select:
	$(PY) code/ext.py select
ext-seeds:
	FIG1_WORKERS=$(EXT_WORKERS) $(PY) code/ext.py orchestrate seeds
# ext.py reuses select_lock_v2.ledger(), which always writes v2/ledger.csv. ext.py is hash-locked, so the
# primary ledger is moved aside here instead of changing the code.
ext-ledger:
	@if [ -f v2/ledger.csv ]; then mv v2/ledger.csv v2/ledger.primary.csv; fi
	$(PY) code/ext.py ledger
	mv v2/ledger.csv v2/ledger_ext.csv
	@if [ -f v2/ledger.primary.csv ]; then mv v2/ledger.primary.csv v2/ledger.csv; fi
	gzip -9 -n -f -k v2/ledger_ext.csv
ext-lock:
	$(PY) code/ext.py lock
ext-evaluate:
	$(PY) code/ext.py evaluate
ext-stats:
	$(PY) code/ext.py stats

primary: screen select seeds ledger lock evaluate stats
extension: ext-screen ext-select ext-seeds ext-ledger ext-lock ext-evaluate ext-stats
all: stage1 primary extension figure

# Remove the committed locked outputs so a from-scratch run can select, lock and evaluate again
# (lock and evaluate refuse to overwrite). `git checkout -- v2 results` restores the record.
clean-record:
	rm -f v2/selection*.json v2/lockfile*.json v2/evaluated*.json v2/test_metrics*.csv v2/final_stats*.json \
	      v2/final_table*.md v2/session_values.csv v2/ledger*.csv.gz
	find results -name neurons.csv -delete
