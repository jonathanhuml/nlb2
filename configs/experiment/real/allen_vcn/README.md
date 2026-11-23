# Allen VCN NLB-style configs

These configs reproduce the selected HAL runs summarized in
`_manifests/summary_with_all_optional_best_final.csv`.

Final HAL run directory, relative to the repository root:
`runs/allen_vcn_nlb_style_20260805_tuned_core_v2`.

Each method has one config per dataset, named
`<method>_allen_vcn_<dataset>_20ms_nlb_style.yaml`. The manifests list the
original HAL run names. Some of those carry the sweep that selected them (for
example `..._nlb_style_optional_fallback_lfadslib_...`), so match a manifest
row to its config by method and dataset. The hyperparameters themselves are in
the config.

The BO-drifting CASSM table value is an arithmetic-mean ensemble;
its ensemble descriptor is saved in `configs/manifests/allen_vcn/`, and its
source run configs are saved in `cassm/ensemble_sources/`.
