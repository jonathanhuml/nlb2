# Proposed edits to `benchmark_paper/paper.tex` (Appendix: Behavioral Decoding)

`paper.tex` has not been changed. These edits bring the methods text in line with what the v2 code does.

## Transformer screen (current text, Stage 2 bullet)

> Its architecture (depth, width, heads) is screened once, on a set of development sessions that do not appear in any reported result. The selected architecture is then fixed, and only its optimization hyperparameters are tuned per session, with a budget of 32 configurations.

## Proposed

> MLP and transformer architectures (MLP: depth and width; transformer: depth, width and heads) are screened with leave-one-mouse-out cross-fitting. Every architecture is fit on every session, and each session uses the architecture with the best mean validation score across sessions from the other mice. No session's own data, or its mouse's data, chooses its architecture. Only the optimizer settings (learning rate and weight decay) are tuned on each session's own validation split. All readouts are frozen in a hashed lockfile before test metrics are computed, once.

## Other sentences that no longer match the code

1. "Readout hyperparameters are chosen per session by 5-fold cross-validated $R^2$ over the training trials." The code uses the BWB TS1 causal validation split. Suggested replacement: "...are chosen per session on the BWB TS1 validation split (causal train/val/test)."
2. "MLP. Depth $\in \{1, 2\}$; width $\in \{64, 128, 256\}$; ... budget of 32 configurations." The code uses MLP configs {d1w128, d1w256, d2w64, d2w128, d2w256, d3w128} × learning rate {1e-3, 3e-4} × weight decay {0.01, 0.1}.
3. Co-smoothing readouts (new in v2) need a sentence. Suggested: "For co-smoothing, the same three readout families map the frozen, held-in-driven representation at each bin to held-out spike counts under a Poisson loss (linear = Poisson GLM with L2 penalty selected on validation co-bps)."
4. Held-out neurons in the IBL sessions: about 20%, stratified by Beryl region only, using a SHA-256-seeded mask. The text says 25%, stratified by region and firing-rate quartile.
