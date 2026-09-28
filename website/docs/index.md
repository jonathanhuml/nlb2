# NLB2

NLB2 provides PyTorch benchmark scaffolding for
latent variable models of neural dynamics.

## Documentation

- [Model documentation](models/index.md)
- [Model output contract](model_output_contract.md)
- [Optimizer contract](optimizer_contract.md)

## CLI

Run a full NLB2 experiment with:

```bash
nlb2 run -d lorenz -m cassm
```

The public registries include synthetic datasets such as `lorenz` and
`chaotic_rnn`, plus model entries such as `ilqr_vae`, `lfads`, and `ndt`.

This builds the dataset, model, trainer, evaluation metrics, and a self-contained
run folder through the public `nlb2.Experiment` orchestration API.

Model pages are generated from model class docstrings and config defaults:

```bash
python website/tools/generate_model_docs.py
```

Images referenced from model docstrings should live under `website/docs/assets/`.
Use links like `![Diagram](assets/model-diagram.png)` in docstrings; generated
model pages rewrite those links to the correct relative path.

Serve the documentation site locally with:

```bash
mkdocs serve --config-file website/mkdocs.yml
```
