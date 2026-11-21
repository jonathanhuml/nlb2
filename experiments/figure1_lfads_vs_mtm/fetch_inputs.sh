#!/bin/sh
# Download the public inputs (~8 GB) needed to re-run the experiment from scratch. Resumable and verified.
#   data/<eid>.h5        the 29 BrainWideBench evaluation sessions (S3), sizes checked against data_sizes.txt
#   checkpoints/mtm.pt   released BrainWideBench MtM checkpoint (Hugging Face), SHA-256 checked
#   vendor/ibl-bwb       BrainWideBench code (MtM model, data utilities), pinned to the commit used for the paper
# Not needed for `make verify`, `make figure` or `make paper`, which use the committed record only.
set -e
cd "$(dirname "$0")"
IBL_BWB_COMMIT=d30ec9cf6859361ceb68399d8c0ab4184d8588cd
MTM_SHA256=cb1078739366b5246750e1b537e91bb02fb15cdf61c45193893fe35537c27721

sha256() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }

mkdir -p data checkpoints vendor
for eid in $(cat eval_recording_ids.txt); do
  want=$(awk -v e="$eid" '$1==e{print $2}' data_sizes.txt)
  have=$( [ -f "data/$eid.h5" ] && wc -c < "data/$eid.h5" | tr -d ' ' || echo 0)
  if [ "$have" != "$want" ]; then
    echo "downloading $eid"
    curl -fL -C - -o "data/$eid.h5" \
      "https://brain-wide-bench.s3.amazonaws.com/brainsets/selected_units/ibl_brain_wide_bench_2026/eval/$eid.h5"
  fi
done

if [ ! -f checkpoints/mtm.pt ]; then
  curl -fL -o checkpoints/mtm.pt https://huggingface.co/nerdslab/ibl-bwb-mtm/resolve/main/mtm.pt
fi
[ "$(sha256 checkpoints/mtm.pt)" = "$MTM_SHA256" ] || { echo "checkpoints/mtm.pt: SHA-256 mismatch"; exit 1; }

if [ ! -d vendor/ibl-bwb/.git ]; then
  git clone -q https://github.com/brainbench-org/ibl-bwb.git vendor/ibl-bwb
fi
git -C vendor/ibl-bwb fetch -q origin "$IBL_BWB_COMMIT" 2>/dev/null || true
git -C vendor/ibl-bwb checkout -q "$IBL_BWB_COMMIT"
echo "inputs ready (ibl-bwb at $(git -C vendor/ibl-bwb rev-parse --short HEAD))"
