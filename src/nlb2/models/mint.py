"""MINT as a native NLB2 model.

This file intentionally keeps the PyTorch MINT implementation self-contained
instead of importing a copied helper package. Training learns condition and
trajectory libraries from spikes, and prediction runs the
Poisson likelihood recursion plus interpolation against those libraries.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from pydantic import Field
from torch import Tensor
from torch.utils.data import Subset

from nlb2.models.base import BaseDynamicsModel, BaseModelConfig, OptimizationConfig
from nlb2.types import LossOutput, ModelOutput


TORCH_DTYPE = torch.float64

@dataclass
class Settings:
    task: str
    data_path: Optional[Path]
    results_path: Optional[Path]
    Ts: float = 0.001
    trial_alignment: range = range(0)
    test_alignment: range = range(0)
    CondInfo: Optional[object] = None
    lorenz_library_source: str = "smoothed_spikes"
    library_rate_source: str = "spikes"
    dmfc_event_offsets: Optional[Dict[str, np.ndarray]] = None
    dmfc_condition_rows: Optional[np.ndarray] = None
    dmfc_section_ids: Optional[np.ndarray] = None


@dataclass
class HyperParams:
    soft_norm: float = 5.0
    min_prob: float = 1e-6
    min_lambda: float = 1.0
    min_rate: float = 0.0
    interp: int = 2
    n_candidates: int = 2
    interp_within_trajectories: bool = False
    min_k_dist: int = 1000
    causal: bool = True
    Delta: int = 20
    window_length: int = 0
    trajectories_alignment: range = range(0)
    sigma: int = 0
    n_neural_dims: Optional[int] = None
    n_cond_dims: Optional[int] = None
    n_trial_dims: Optional[int] = 1
    dmfc_section_count: int = 0
    dmfc_section_ridge: float = 100.0
    dmfc_set_go_weight: float = 4.0


@dataclass
class InterpOptions:
    max_iters: int = 10
    step_tol: float = 0.01


@dataclass
class MatTable:
    fields: Dict[str, np.ndarray]

    def __getitem__(self, key: str) -> np.ndarray:
        return self.fields[key]

    def subset_time(self, mask: np.ndarray) -> "MatTable":
        out = dict(self.fields)
        for key, value in out.items():
            if key == "time":
                out[key] = value[mask]
            elif value.ndim == 2 and value.shape[1] == mask.shape[0]:
                out[key] = value[:, mask]
        return MatTable(out)

    def subset_trials(self, mask: np.ndarray) -> "MatTable":
        return MatTable({key: value[mask] for key, value in self.fields.items()})

    @property
    def n_trials(self) -> int:
        first = next(iter(self.fields.values()))
        return int(first.shape[0])


DATASET_FIELDS: Mapping[str, Mapping[str, Mapping[str, str]]] = {
    "mc_maze": {
        "T": {
            "time": "d",
            "hand_pos": "e",
            "hand_vel": "f",
            "spikes": "g",
            "heldout_spikes": "h",
        },
        "TrialInfo": {
            "start_time": "B",
            "move_onset_time": "C",
            "end_time": "D",
            "trial_type": "E",
            "trial_version": "F",
            "maze_id": "G",
            "success": "H",
            "split": "I",
        },
    },
    "mc_rtt": {
        "T": {
            "time": "d",
            "spikes": "e",
            "heldout_spikes": "f",
            "finger_pos": "g",
            "finger_vel": "h",
            "target_pos": "i",
            "autolfads_rates": "j",
        },
        "TrialInfo": {
            "start_time": "H",
            "end_time": "I",
            "split": "J",
        },
    },
    "area2_bump": {
        "T": {
            "time": "d",
            "spikes": "e",
            "heldout_spikes": "f",
            "hand_pos": "g",
            "hand_vel": "h",
            "force": "i",
            "joint_ang": "j",
            "joint_vel": "k",
            "muscle_len": "l",
            "muscle_vel": "m",
        },
        "TrialInfo": {
            "start_time": "Q",
            "end_time": "R",
            "result": "S",
            "ctr_hold_bump": "do",
            "bump_dir": "eo",
            "target_dir": "fo",
            "move_onset_time": "go",
            "cond_dir": "ho",
            "split": "io",
        },
    },
}


@BaseModelConfig.register
class MINTConfig(BaseModelConfig):
    """Config for the MINT trajectory-library decoder.

    The default `train_source="h5"` estimates trajectories from training
    spikes by smoothing and averaging, using one library-fitting epoch.
    With `train_source="lfads"`, each `trainer.epochs` iteration trains the
    rate estimator for one pass and updates the trajectory library. Both
    paths use `optimization.name="library_fit"` and report training and
    validation Poisson negative log likelihood.

    Prepared NLB H5 tensors are the default input. Neuron dimensions and sample
    intervals come from the dataset; condition metadata, when available, groups
    training trials. `train_source="lfads"` trains LFADS from those same spikes
    before fitting templates. Explicit NWB/MAT sources retain the original
    task-specific reproduction adapters.

    `Experiment` uses `trainer.epochs` as its training budget. The model field
    `lfads_epochs` controls the direct `fit_training_data` helper and legacy
    reproduction runners only.

    For synthetic Lorenz and chaotic-RNN tasks, NLB2 builds the MINT trajectory
    library from repeated training trials. With `train_source="h5"`, the default
    `lorenz_library_source="smoothed_spikes"` estimates library rates by
    Gaussian-smoothing training spikes and averaging by condition.
    `lorenz_library_source="true_rates"` is an oracle sanity-check mode only
    and should not be used for fair method comparisons.
    The default Lorenz split repeats the same initial-condition trajectories
    across train and validation trials, so this benchmark measures denoising of
    seen trajectories rather than interpolation to unseen trajectories.
    """

    name: Literal["mint"] = "mint"
    objective: str = "mint_likelihood_recursion"
    dataset: Literal[
        "auto",
        "area2_bump",
        "chaotic_rnn",
        "dmfc_rsg",
        "mc_maze",
        "mc_rtt",
        "lorenz",
        "allen_vcn",
    ] = "auto"
    train_source: Literal["h5", "lfads", "mat", "nwb"] = "h5"
    train_split: Literal["auto", "train", "trainval"] = "trainval"
    nlb_neural_state_defaults: bool = True
    nwb_root: Optional[str] = None
    mat_data_root: Optional[str] = None
    target_h5: Optional[str] = None
    eval_bin_size_ms: int = 5
    lorenz_library_source: Literal["smoothed_spikes", "true_rates"] = "smoothed_spikes"
    n_candidates: Optional[int] = Field(default=None, ge=1)
    window_length: Optional[int] = Field(default=None, ge=1)
    delta: Optional[int] = Field(default=None, ge=1)
    interp: Optional[Literal[0, 1, 2]] = None
    interp_within_trajectories: Optional[bool] = None
    allen_condition_mode: Literal["condition_id", "trial_index"] = "condition_id"
    allen_library_source: Literal["spikes", "lfads_checkpoint"] = "spikes"
    allen_lfads_run_dir: Optional[str] = None
    sigma: Optional[int] = Field(default=None, ge=0)
    min_rate: Optional[float] = None
    causal: Optional[bool] = None
    n_neural_dims: Optional[int] = None
    n_cond_dims: Optional[int] = None
    n_trial_dims: Optional[int] = None
    lfads_epochs: int = 25
    lfads_batch_size: int = 16
    lfads_train_bin_size: int = 1
    lfads_lr: float = 1e-3
    lfads_generator_dim: int = 64
    lfads_factor_dim: int = 20
    lfads_inferred_input_dim: int = 2
    lfads_encoder_dim: int = 64
    lfads_controller_dim: int = 64
    lfads_keep_prob: float = 0.95
    lfads_seed: int = 0
    optimization: OptimizationConfig = Field(
        default_factory=lambda: OptimizationConfig(name="library_fit")
    )

    def build(self, n_neurons: int, n_time: int) -> "MINT":
        del n_neurons, n_time
        return MINT(self)

    def build_from_data(self, data) -> "MINT":
        config = self
        if self.dataset == "auto":
            name = str(getattr(data.config, "name", "auto"))
            if name.startswith("allen_vcn"):
                name = "allen_vcn"
            if name not in {"area2_bump", "dmfc_rsg", "mc_maze", "mc_rtt", "lorenz", "chaotic_rnn", "allen_vcn"}:
                name = "auto"
            config = self.model_copy(update={"dataset": name})
        model = config.build(data.n_neurons, data.n_time)
        model.configure_training_data(data.train_dataset)
        return model


class MINT(BaseDynamicsModel):
    """Mesh of Idealized Neural Trajectories adapted to the NLB2 API.

    ## Method

    MINT builds a library of idealized neural trajectories (`Omega_plus`) and
    paired task-state trajectories (`Phi_plus`). Prediction does not optimize
    model parameters. Instead, it bins incoming spikes, updates a Poisson
    likelihood recursion over the library, and estimates rates by interpolating
    between likely library states. Training is a statistical template fit;
    optional LFADS rate estimation has its own gradient training stage.

    ## NLB datasets

    The native NLB2 MINT port supports the three MINT/NLB datasets used in the
    original repository: `area2_bump`, `mc_maze`, and `mc_rtt`, plus a
    NLB2-native `dmfc_rsg` adapter built from the NLB 5 ms H5 tensors. Area2
    and Maze smooth and average repeated condition-aligned trials; the RTT
    config trains LFADS from raw training spikes and fits single-trial rate
    trajectories. DMFC averages prepared condition-indexed trials. This H5
    adapter does not reproduce the original event-warped DMFC NWB procedure.
    Prepared NLB and Allen data use the standard `Experiment` and `nlb2 run`
    fit/save/load workflow. Legacy NWB/MAT adapters remain explicit options.

    ## Synthetic datasets

    The synthetic Lorenz and chaotic-RNN adapters are NLB2-specific trajectory
    builders. With the default `lorenz_library_source="smoothed_spikes"`, the
    library is estimated from training spikes by Gaussian smoothing and
    condition averaging. This keeps the comparison non-oracular while still
    matching MINT's assumption that useful trajectory templates are learned
    before inference.

    The `true_rates` library source is intentionally exposed for debugging. It
    reproduces an oracle/template-retrieval sanity check, not a fair method
    comparison. Use it only when validating the likelihood/interpolation code.

    ## Outputs

    `forward` accepts raw `(batch, time, neurons)` spike counts and returns expected spike
    counts per input bin. The training epoch learns the trajectory library and
    reports Poisson negative log likelihood; checkpoints contain the complete
    fitted library and neuron layout.
    """

    def __init__(self, config: MINTConfig) -> None:
        super().__init__()
        self.config = config
        self.objective = config.objective
        self.settings, self.hyperparams = get_mint_config(
            config.dataset,
            nlb_neural_state_defaults=config.nlb_neural_state_defaults and config.train_source in {"nwb", "mat"},
        )
        self._apply_config_overrides()
        if config.mat_data_root is not None:
            self.settings.data_path = Path(config.mat_data_root) / f"{config.dataset}.mat"

        self.Ts = self.settings.Ts
        self.Delta = self.hyperparams.Delta
        self.dt = self.Delta * self.Ts
        self.window_length = self.hyperparams.window_length
        self.tau_prime = round(self.window_length / self.Delta) - 1
        self.causal = self.hyperparams.causal
        self.interp = self.hyperparams.interp

        self.min_prob = self.hyperparams.min_prob
        self.min_lambda = self.hyperparams.min_lambda
        self.n_rates = 2000
        self.min_rate = 0.0
        self.max_spikes = 0
        self.InterpOptions = InterpOptions()

        self.Omega_plus: List[Tensor] = []
        self.Phi_plus: List[Tensor] = []
        self.behavior_labels: List[str] = []
        self.library_ids: Optional[Tensor] = None
        self.V: Optional[Tensor] = None
        self.first_idx0: Optional[Tensor] = None
        self.last_idx0: Optional[Tensor] = None
        self.first_tau_prime_idx0: Optional[Tensor] = None
        self.shifted_idx1: Optional[Tensor] = None
        self.shifted_idx2: Optional[Tensor] = None
        self.n_heldin: Optional[int] = None
        self.n_heldout: int = 0
        self._training_conditions: Optional[np.ndarray] = None
        self._prepared_data = False
        self.library_training: dict = {}

        self.register_buffer("_device_anchor", torch.empty(0))
        self._refresh_runtime_params()

    def configure_training_data(self, dataset) -> None:
        """Read the training layout without fitting or inspecting validation data."""
        indices = np.arange(len(dataset))
        while hasattr(dataset, "dataset"):
            if isinstance(dataset, Subset):
                indices = np.asarray(dataset.indices, dtype=np.int64)[indices]
            dataset = dataset.dataset
        if len(dataset) == 0:
            raise ValueError("MINT requires nonempty training data.")
        sample = dataset[0]
        self.settings.Ts = float(sample["dt"])
        n_time = int(sample["spikes"].shape[0])
        self.settings.trial_alignment = range(n_time)
        self.settings.test_alignment = range(n_time)
        self.hyperparams.trajectories_alignment = range(n_time)
        # Prepared tensors are already aligned and binned by the dataset adapter.
        self._prepared_data = True
        if "heldout_spikes" in sample:
            self.n_heldin = int(sample["heldin_spikes"].shape[-1])
            self.n_heldout = int(sample["heldout_spikes"].shape[-1])
            self.output_neuron_start = self.n_heldin
            self.hyperparams.Delta = self.config.delta or 1
            self.hyperparams.window_length = self.config.window_length or min(6, n_time)
            self.hyperparams.sigma = self.config.sigma if self.config.sigma is not None else 2
            self.hyperparams.n_neural_dims = self.config.n_neural_dims
            self.hyperparams.n_cond_dims = self.config.n_cond_dims
            self.hyperparams.n_trial_dims = self.config.n_trial_dims
            self.hyperparams.causal = self.config.causal if self.config.causal is not None else False
            cfg = getattr(dataset, "config", None)
            path = getattr(cfg, "resolved_data_path", None)
            if path is not None:
                self._training_conditions = _prepared_training_conditions(
                    Path(path), getattr(cfg, "group", None), self.config.dataset, len(dataset)
                )
        else:
            cfg = getattr(dataset, "config", None)
            n_inits = getattr(cfg, "num_inits", None)
            if n_inits is not None:
                self._training_conditions = np.arange(len(dataset)) % int(n_inits)
            elif getattr(cfg, "num_conditions", None) is not None:
                repeats = int(cfg.train_fraction * cfg.num_trials)
                self._training_conditions = np.repeat(np.arange(cfg.num_conditions), repeats)
                if len(self._training_conditions) != len(dataset):
                    raise ValueError("MINT condition metadata does not match the training trial count.")
        if self._training_conditions is not None:
            self._training_conditions = self._training_conditions[indices]
        self._refresh_runtime_params()

    def fit_training_data(self, loader, *, device) -> None:
        """Fit a complete library directly; lfads_epochs controls this helper's budget."""
        if self.V is not None:
            return
        trials, oracle, conditions = self._library_training_inputs(loader, device=device)
        rate_trials = None
        if self.config.train_source == "lfads":
            from nlb2.mint_nlb import _fit_lfads_rate_trials

            rate_trials = _fit_lfads_rate_trials(trials, self, self.config, torch.device(device))
        elif self.config.dataset == "allen_vcn" and self.config.allen_library_source == "lfads_checkpoint":
            adapter = _MINTAllenVCNAdapter(library_source="lfads_checkpoint", lfads_run_dir=self.config.allen_lfads_run_dir)
            rates = adapter._lfads_rate_trials(
                [trial[:self.n_heldin].T for trial in trials], n_time=trials[0].shape[1],
                n_full=trials[0].shape[0], device=torch.device(device),
            )
            rate_trials = [rate * self.Delta for rate in rates]
        elif oracle:
            rate_trials = oracle
        self.settings.library_rate_source = "prepared_rates" if rate_trials is not None else "prepared_spikes"
        self.fit_library(trials, rate_trials or trials, conditions)
        self.library_training.update(source=self.config.train_source, trials=len(trials), trajectories=len(self.Omega_plus))

    def create_library_trainer(self):
        if self.config.train_source == "lfads":
            from nlb2.mint_nlb import _MINTLFADSTrainer

            return _MINTLFADSTrainer(self)
        return None

    def _library_training_inputs(self, loader, *, device):
        if self.config.train_source in {"nwb", "mat"}:
            raise ValueError("Use run_mint_nlb for explicit NWB/MAT reproduction sources.")
        if loader is None or not hasattr(loader, "dataset"):
            raise ValueError("MINT fitting requires a training DataLoader.")
        self.configure_training_data(loader.dataset)
        self.to(device)
        trials, oracle, conditions = [], [], []
        for index in range(len(loader.dataset)):
            sample = loader.dataset[index]
            if self.n_heldin is not None:
                full = torch.cat([sample["heldin_spikes"], sample["heldout_spikes"]], dim=-1)
            else:
                full = sample.get("raw_spikes", sample["spikes"])
            trials.append(full.T.to(device=device, dtype=TORCH_DTYPE).contiguous())
            cond = sample.get("condition_id")
            if self.config.allen_condition_mode == "trial_index" and self.config.dataset == "allen_vcn":
                cond = index
            elif cond is None:
                cond = self._training_conditions[index] if self._training_conditions is not None else index
            conditions.append(int(cond))
            if self.config.lorenz_library_source == "true_rates" and self.config.train_source != "lfads":
                if "rates" not in sample:
                    raise ValueError("Oracle MINT fitting requires explicitly provided training rates.")
                unit = sample.get("rates_unit", "counts")
                if unit not in {"counts", "hz"}:
                    raise ValueError("MINT training rates must declare 'counts' or 'hz' units.")
                scale = self.dt if unit == "hz" else self.Delta
                oracle.append(sample["rates"].T.to(device=device, dtype=TORCH_DTYPE) * scale)
        return trials, oracle, conditions

    def get_extra_state(self):
        return {
            "version": 1, "hyperparams": asdict(self.hyperparams) | {"trajectories_alignment": None},
            "Ts": self.Ts, "Omega_plus": self.Omega_plus, "Phi_plus": self.Phi_plus,
            "behavior_labels": self.behavior_labels, "library_ids": self.library_ids,
            "n_heldin": self.n_heldin, "n_heldout": self.n_heldout,
            "prepared_data": self._prepared_data, "library_training": self.library_training,
            "n_rates": self.n_rates, "interp_options": asdict(self.InterpOptions),
        }

    def set_extra_state(self, state) -> None:
        if state.get("version") != 1:
            raise ValueError("Unsupported MINT checkpoint format.")
        hp = dict(state["hyperparams"])
        hp["trajectories_alignment"] = range(0)
        self.hyperparams = HyperParams(**hp)
        self.n_rates = int(state["n_rates"])
        self.InterpOptions = InterpOptions(**state["interp_options"])
        self.settings.Ts = float(state["Ts"])
        self.Omega_plus = [item.to(self.device) for item in state["Omega_plus"]]
        self.Phi_plus = [item.to(self.device) for item in state["Phi_plus"]]
        self.behavior_labels = list(state["behavior_labels"])
        ids = state["library_ids"]
        self.library_ids = None if ids is None else ids.to(self.device)
        self.n_heldin, self.n_heldout = state["n_heldin"], state["n_heldout"]
        if self.n_heldin is not None:
            self.output_neuron_start = self.n_heldin
        self._prepared_data = bool(state["prepared_data"])
        self.library_training = dict(state["library_training"])
        self._refresh_runtime_params()
        if self.Omega_plus:
            self._index_library()

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        self.Omega_plus = [fn(item) for item in self.Omega_plus]
        self.Phi_plus = [fn(item) for item in self.Phi_plus]
        for name in ("library_ids", "V", "first_idx0", "last_idx0", "first_tau_prime_idx0", "shifted_idx1", "shifted_idx2"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, fn(value))
        self._refresh_runtime_params()
        return self

    def _apply_config_overrides(self) -> None:
        self.settings.lorenz_library_source = self.config.lorenz_library_source
        if self.config.n_candidates is not None:
            self.hyperparams.n_candidates = int(self.config.n_candidates)
        if self.config.window_length is not None:
            self.hyperparams.window_length = int(self.config.window_length)
        if self.config.delta is not None:
            self.hyperparams.Delta = int(self.config.delta)
        if self.config.interp is not None:
            self.hyperparams.interp = int(self.config.interp)
        if self.config.interp_within_trajectories is not None:
            self.hyperparams.interp_within_trajectories = bool(
                self.config.interp_within_trajectories
            )
        if self.config.sigma is not None:
            self.hyperparams.sigma = int(self.config.sigma)
        if self.config.min_rate is not None:
            self.hyperparams.min_rate = float(self.config.min_rate)
        if self.config.causal is not None:
            self.hyperparams.causal = bool(self.config.causal)
        if self.config.n_neural_dims is not None:
            self.hyperparams.n_neural_dims = int(self.config.n_neural_dims)
        if self.config.n_cond_dims is not None:
            self.hyperparams.n_cond_dims = int(self.config.n_cond_dims)
        if self.config.n_trial_dims is not None:
            self.hyperparams.n_trial_dims = int(self.config.n_trial_dims)

    def _refresh_runtime_params(self) -> None:
        """Refresh derived tensors after config/hyperparameter overrides."""

        self.Ts = self.settings.Ts
        self.Delta = self.hyperparams.Delta
        self.dt = self.Delta * self.Ts
        self.window_length = self.hyperparams.window_length
        self.tau_prime = round(self.window_length / self.Delta) - 1
        self.causal = self.hyperparams.causal
        self.interp = self.hyperparams.interp
        self.min_prob = self.hyperparams.min_prob
        self.min_lambda = self.hyperparams.min_lambda
        self.lambda_range = (
            torch.as_tensor([self.min_lambda, 500.0], dtype=TORCH_DTYPE, device=self.device)
            * self.dt
        )
        self.rates = torch.linspace(
            float(self.lambda_range[0]),
            float(self.lambda_range[1]),
            self.n_rates,
            dtype=TORCH_DTYPE,
            device=self.device,
        )
        self.max_spikes = round(self.dt * 1000)
        self.L = self._build_likelihood_table()
        self.min_rate = self.hyperparams.min_rate * self.dt

    def _build_likelihood_table(self) -> Tensor:
        counts = torch.arange(self.max_spikes + 1, dtype=TORCH_DTYPE, device=self.device).reshape(1, -1)
        rates = self.rates.reshape(-1, 1)
        logp = counts * torch.log(rates) - rates - torch.lgamma(counts + 1.0)
        p = torch.exp(logp)
        invalid = p <= self.min_prob
        p_nan = p.masked_fill(invalid, float("nan"))
        nan_count = invalid.sum(dim=1, keepdim=True).to(TORCH_DTYPE)
        row_sum = torch.nansum(p_nan, dim=1, keepdim=True)
        scale = (1.0 - self.min_prob * nan_count) / row_sum
        p_scaled = p_nan * scale
        p_scaled = torch.where(torch.isnan(p_scaled), torch.full_like(p_scaled, self.min_prob), p_scaled)
        return torch.log(p_scaled)

    def fit_library(self, S: Sequence[Tensor], Z: Sequence[Tensor], condition: Sequence[int]) -> "MINT":
        """Build MINT trajectory libraries from spike and behavior/rate trials."""

        self._refresh_runtime_params()
        fit_result = fit_trajectories(S, Z, condition, self.settings, self.hyperparams)
        if len(fit_result) == 4:
            self.Omega_plus, self.Phi_plus, self.behavior_labels, library_ids = fit_result
            if len(library_ids) != len(self.Omega_plus):
                raise ValueError(
                    f"MINT got {len(library_ids)} library ids for {len(self.Omega_plus)} trajectories."
                )
            self.library_ids = torch.as_tensor(library_ids, dtype=torch.long, device=self.device)
        else:
            self.Omega_plus, self.Phi_plus, self.behavior_labels = fit_result
            self.library_ids = None
        self._index_library()
        return self

    def _index_library(self) -> None:
        if not self.Omega_plus:
            raise ValueError("MINT fitting produced no trajectories.")
        if self.Delta < 1 or self.window_length < self.Delta:
            raise ValueError("MINT requires window_length >= delta >= 1 (in input bins).")
        lambdas = [bin_data(omega, self.Delta, "mean") for omega in self.Omega_plus]
        v_cells = [get_rate_indices(lam, self.lambda_range, self.n_rates) for lam in lambdas]

        lengths = [int(v.shape[1]) for v in v_cells]
        if min(lengths) <= self.tau_prime:
            raise ValueError("MINT likelihood window exceeds a fitted trajectory's duration.")
        starts = []
        total = 0
        for length in lengths:
            starts.append(total)
            total += length
        self.first_idx0 = torch.as_tensor(starts, dtype=torch.long, device=self.device)
        self.last_idx0 = torch.as_tensor(
            [start + length - 1 for start, length in zip(starts, lengths)],
            dtype=torch.long,
            device=self.device,
        )
        self.first_tau_prime_idx0 = torch.cat(
            [start + torch.arange(self.tau_prime, dtype=torch.long, device=self.device) for start in self.first_idx0]
        ).sort().values
        self.V = torch.cat([v.T for v in v_cells], dim=0).to(torch.long)
        self._build_shifted_indices()

    def _build_shifted_indices(self) -> None:
        idx1, idx2 = [], []
        assert self.first_idx0 is not None and self.last_idx0 is not None
        for start, last in zip(self.first_idx0.tolist(), self.last_idx0.tolist()):
            if last - self.tau_prime - 1 >= start:
                idx1.append(torch.arange(start, last - self.tau_prime, dtype=torch.long, device=self.device))
                idx2.append(torch.arange(start + self.tau_prime + 1, last + 1, dtype=torch.long, device=self.device))
        if idx1:
            self.shifted_idx1 = torch.cat(idx1)
            self.shifted_idx2 = torch.cat(idx2)
        else:
            self.shifted_idx1 = torch.empty(0, dtype=torch.long, device=self.device)
            self.shifted_idx2 = torch.empty(0, dtype=torch.long, device=self.device)

    def forward(self, x: Tensor) -> ModelOutput:
        if x.ndim != 3:
            raise ValueError("MINT expects batched spikes with shape (batch, time, neurons).")
        mask = None
        if self.n_heldin is not None:
            if x.shape[-1] < self.n_heldin:
                raise ValueError("MINT query has fewer held-in neurons than its fitted library.")
            observed = x[..., :self.n_heldin]
            x = torch.cat([observed, observed.new_zeros(*observed.shape[:-1], self.n_heldout)], dim=-1)
            mask = torch.arange(x.shape[-1], device=self.device) < self.n_heldin
        spikes = [trial.T.contiguous() for trial in x]
        states, _ = self.predict_spike_trials(spikes, likelihood_neuron_mask=mask, verbose=False)
        full_counts = torch.stack([item.T for item in states], dim=0) / self.Delta
        if self.n_heldin is not None:
            return ModelOutput(rates=full_counts[..., self.n_heldin:], rates_unit="counts",
                               extras={"full_rates": full_counts}, full_rates_unit="counts")
        return ModelOutput(rates=full_counts, rates_unit="counts")

    def loss_forward_observations(self, batch, x: Tensor) -> Tensor:
        if isinstance(batch, dict):
            if "heldin_spikes" in batch:
                return batch["heldin_spikes"]
            return batch.get("raw_spikes", x)
        return x

    def loss(
        self,
        batch: Tensor | dict[str, Tensor],
        output: ModelOutput,
        epoch: int = 0,
    ) -> LossOutput:
        from nlb2.metrics import poisson_negative_log_likelihood

        del epoch
        if output.rates is None:
            raise RuntimeError("MINT loss requires predicted count rates.")
        if isinstance(batch, dict):
            spikes = batch.get("heldout_spikes", batch.get("raw_spikes", batch["spikes"]))
        else:
            spikes = batch if self.n_heldin is None else batch[..., self.n_heldin:]
        rates = output.rates
        spikes = spikes.to(device=rates.device, dtype=rates.dtype)
        if spikes.shape != rates.shape:
            raise ValueError("MINT loss requires matching spike and count-rate shapes.")
        observed = torch.isfinite(spikes)
        if not bool(observed.any()):
            raise ValueError("MINT loss requires observed spike counts.")
        total = poisson_negative_log_likelihood(rates[observed], spikes[observed]).mean()
        return LossOutput(
            total=total,
            named_terms={"poisson_nll": total},
            objective=self.objective,
        )

    def evaluation_adapter(self, task: str):
        if task == "nlb":
            return _MINTPreparedAdapter()
        if task == "synthetic":
            from nlb2.metrics import SyntheticEvaluationAdapter

            return SyntheticEvaluationAdapter(use_raw_spikes=True)
        return None

    def predict_spike_trials(
        self,
        S: Sequence[Tensor],
        return_aux: bool = False,
        likelihood_neuron_mask: Optional[Tensor] = None,
        verbose: bool = True,
    ):
        """Predict neural states for trials shaped ``(neurons, time)``."""

        if self.V is None:
            raise RuntimeError("MINT must be fit with fit_library() before predict().")

        S_bar = []
        for spikes in S:
            binned = bin_data(spikes, self.Delta, "sum")
            if binned.shape[1] <= self.tau_prime:
                raise ValueError("MINT query duration must cover its likelihood window.")
            binned = torch.nan_to_num(binned, nan=0.0, posinf=float(self.max_spikes), neginf=0.0)
            S_bar.append(torch.clamp(binned, 0, self.max_spikes).to(torch.long))

        n_trials = len(S)
        X_hat, Z_hat, C_hat, K_hat, Alpha_hat = [], [], [], [], []
        n_early_samples = self.Delta * (self.tau_prime + 1) - 1
        if likelihood_neuron_mask is not None:
            likelihood_neuron_mask = likelihood_neuron_mask.to(device=self.device, dtype=torch.bool)

        for tr in range(n_trials):
            T = S[tr].shape[1]
            T_prime = S_bar[tr].shape[1]
            Q = torch.zeros(self.V.shape[0], dtype=TORCH_DTYPE, device=self.device)
            x_hat = torch.zeros((self.Omega_plus[0].shape[0], T), dtype=TORCH_DTYPE, device=self.device)
            z_hat = torch.zeros((self.Phi_plus[0].shape[0], T), dtype=TORCH_DTYPE, device=self.device)
            if self.interp == 2:
                c_hat = torch.zeros((2, T), dtype=TORCH_DTYPE, device=self.device)
                k_hat = torch.zeros((4, T), dtype=TORCH_DTYPE, device=self.device)
                alpha_hat = torch.zeros((3, T), dtype=TORCH_DTYPE, device=self.device)
            else:
                c_hat = torch.zeros((1, T), dtype=TORCH_DTYPE, device=self.device)
                k_hat = torch.zeros((1, T), dtype=TORCH_DTYPE, device=self.device)
                alpha_hat = torch.zeros((1, T), dtype=TORCH_DTYPE, device=self.device)

            for t0 in range(T_prime):
                t_prime_one = t0 + 1
                s_new = S_bar[tr][:, t0]
                if t_prime_one > self.tau_prime + 1:
                    s_old = S_bar[tr][:, t0 - self.tau_prime - 1]
                else:
                    s_old = torch.zeros_like(s_new)
                Q = self._recursion(Q, s_new, s_old, t_prime_one, likelihood_neuron_mask)

                if t_prime_one > self.tau_prime:
                    t_idx0, f = get_time_indices(t_prime_one, T_prime, T, self.Delta, self.tau_prime, self.causal)
                    t_idx0 = t_idx0.to(self.device)
                    s_curr = S_bar[tr][:, t0 - self.tau_prime : t0 + 1].to(TORCH_DTYPE)
                    x, z, c, k, a = self._estimate_states(Q, s_curr, f, likelihood_neuron_mask)
                    x_hat[:, t_idx0] = x
                    z_hat[:, t_idx0] = z
                    c_hat[:, t_idx0] = c
                    k_hat[:, t_idx0] = k
                    alpha_hat[:, t_idx0] = a

            x_hat = torch.clamp(x_hat, min=self.min_rate)
            if self.causal:
                x_hat[:, :n_early_samples] = float("nan")
                z_hat[:, :n_early_samples] = float("nan")

            X_hat.append(x_hat)
            Z_hat.append(z_hat)
            C_hat.append(c_hat)
            K_hat.append(k_hat)
            Alpha_hat.append(alpha_hat)
            if verbose:
                print(f"Completed trial {tr + 1}")

        if return_aux:
            return X_hat, Z_hat, C_hat, K_hat, Alpha_hat
        return X_hat, Z_hat

    def _recursion(
        self,
        Q: Tensor,
        s_new: Tensor,
        s_old: Tensor,
        t_prime_one: int,
        likelihood_neuron_mask: Optional[Tensor] = None,
    ) -> Tensor:
        assert self.V is not None and self.first_idx0 is not None
        assert self.shifted_idx1 is not None and self.shifted_idx2 is not None
        updated = torch.empty_like(Q)
        updated[0] = 0.0
        updated[1:] = Q[:-1]
        updated[self.first_idx0] = 0.0
        Q = updated

        if likelihood_neuron_mask is not None:
            V = self.V[:, likelihood_neuron_mask]
            s_new = s_new[likelihood_neuron_mask]
            s_old = s_old[likelihood_neuron_mask]
        else:
            V = self.V

        s_new = torch.clamp(s_new.to(torch.long), 0, self.max_spikes)
        gathered = self.L[V, s_new.reshape(1, -1).expand_as(V)]
        Q = Q + gathered.sum(dim=1)
        if t_prime_one > self.tau_prime + 1:
            s_old = torch.clamp(s_old.to(torch.long), 0, self.max_spikes)
            old_v = V[self.shifted_idx1]
            old_gathered = self.L[old_v, s_old.reshape(1, -1).expand_as(old_v)]
            Q[self.shifted_idx2] = Q[self.shifted_idx2] - old_gathered.sum(dim=1)
        return Q

    def _estimate_states(
        self,
        Q: Tensor,
        S_curr: Tensor,
        f: Callable[[int], Tensor],
        likelihood_neuron_mask: Optional[Tensor] = None,
    ):
        K_lengths = [phi.shape[1] for phi in self.Phi_plus]
        if self.interp == 0:
            c0, k_hats = self._maximum_likelihood(Q, restricted_conds=[])
            k_idx = get_state_indices(k_hats, f, K_lengths[c0]).to(self.device)
            x = self.Omega_plus[c0][:, k_idx[0]]
            z = self.Phi_plus[c0][:, k_idx[0]]
            return (
                x,
                z,
                torch.full((1, z.shape[1]), c0 + 1, dtype=TORCH_DTYPE, device=self.device),
                (k_idx[0] + 1).reshape(1, -1).to(TORCH_DTYPE),
                torch.full((1, z.shape[1]), float("nan"), dtype=TORCH_DTYPE, device=self.device),
            )

        if self.interp == 1 or self.hyperparams.n_candidates == 1 or (
            len(self.Omega_plus) == 1 and not self.hyperparams.interp_within_trajectories
        ):
            c0, k_hats = self._maximum_likelihood(Q, restricted_conds=[])
            k_idx = get_state_indices(k_hats, f, K_lengths[c0]).to(self.device)
            x, z, _, alpha = self._interp_adjacent_states(S_curr, c0, k_hats, k_idx, likelihood_neuron_mask)
            return (
                x,
                z,
                torch.full((1, z.shape[1]), float("nan"), dtype=TORCH_DTYPE, device=self.device),
                torch.full((1, z.shape[1]), float("nan"), dtype=TORCH_DTYPE, device=self.device),
                torch.full((1, z.shape[1]), alpha, dtype=TORCH_DTYPE, device=self.device),
            )

        if self.library_ids is not None:
            library_interp = self._estimate_states_across_libraries(
                Q, S_curr, f, likelihood_neuron_mask, K_lengths
            )
            if library_interp is not None:
                return library_interp

        candidates = []
        states_to_exclude = []
        conds_to_exclude: List[int] = []
        min_k_prime_dist = self.hyperparams.min_k_dist / self.Delta
        limit = self.hyperparams.n_candidates
        if not self.hyperparams.interp_within_trajectories:
            limit = min(limit, len(self.Omega_plus))
        for _ in range(limit):
            try:
                if self.hyperparams.interp_within_trajectories:
                    c0, k_hats = self._maximum_likelihood(
                        Q, states_to_exclude=states_to_exclude, min_k_prime_dist=min_k_prime_dist
                    )
                    states_to_exclude.append((c0, k_hats[0]))
                else:
                    c0, k_hats = self._maximum_likelihood(Q, restricted_conds=conds_to_exclude)
                    conds_to_exclude.append(c0)
            except ValueError:
                if not candidates:
                    raise
                break
            k_idx = get_state_indices(k_hats, f, K_lengths[c0]).to(self.device)
            candidates.append(
                (*self._interp_adjacent_states(S_curr, c0, k_hats, k_idx, likelihood_neuron_mask), c0, k_idx)
            )

        if len(candidates) == 1:
            x, z, _, _, _, _ = candidates[0]
            return (x, z, *[
                torch.full((rows, z.shape[1]), float("nan"), dtype=TORCH_DTYPE, device=self.device)
                for rows in (2, 4, 3)
            ])

        interps = []
        for a, b in combinations(range(len(candidates)), 2):
            x_a, z_a, lam_a, alpha_a, c_a, k_a = candidates[a]
            x_b, z_b, lam_b, alpha_b, c_b, k_b = candidates[b]
            beta = fit_poisson_interp(
                _masked_rows(S_curr, likelihood_neuron_mask),
                _masked_rows(lam_a, likelihood_neuron_mask),
                _masked_rows(lam_b, likelihood_neuron_mask),
                self.InterpOptions,
                0.0,
            )
            lam = (1.0 - beta) * lam_a + beta * lam_b
            x = (1.0 - beta) * x_a + beta * x_b
            z = (1.0 - beta) * z_a + beta * z_b
            interps.append((x, z, lam, beta, alpha_a, alpha_b, c_a, c_b, k_a, k_b))

        x, z, _, beta, alpha_a, alpha_b, c_a, c_b, k_a, k_b = use_best_interp(
            S_curr, interps, likelihood_neuron_mask
        )
        return (
            x,
            z,
            torch.as_tensor([c_a + 1, c_b + 1], dtype=TORCH_DTYPE, device=self.device).reshape(2, 1).expand(2, z.shape[1]),
            torch.cat([(k_a + 1).to(TORCH_DTYPE), (k_b + 1).to(TORCH_DTYPE)], dim=0),
            torch.as_tensor([beta, alpha_a, alpha_b], dtype=TORCH_DTYPE, device=self.device).reshape(3, 1).expand(3, z.shape[1]),
        )

    def _estimate_states_across_libraries(
        self,
        Q: Tensor,
        S_curr: Tensor,
        f: Callable[[int], Tensor],
        likelihood_neuron_mask: Optional[Tensor],
        K_lengths: Sequence[int],
    ):
        assert self.library_ids is not None
        if torch.unique(self.library_ids).numel() < 2:
            return None
        try:
            c_a, k_hats_a = self._maximum_likelihood(Q, restricted_conds=[])
            lib_a = int(self.library_ids[c_a].item())
            cand_a = self._estimate_within_library(
                Q, S_curr, f, likelihood_neuron_mask, K_lengths, c_a, k_hats_a, lib_a
            )
            c_b, k_hats_b = self._maximum_likelihood(Q, restricted_library_ids=[lib_a])
            lib_b = int(self.library_ids[c_b].item())
            cand_b = self._estimate_within_library(
                Q, S_curr, f, likelihood_neuron_mask, K_lengths, c_b, k_hats_b, lib_b
            )
        except ValueError:
            return None

        x_a, z_a, lam_a, alpha_a, c_a_primary, k_a_primary = cand_a
        x_b, z_b, lam_b, alpha_b, c_b_primary, k_b_primary = cand_b
        beta = fit_poisson_interp(
            _masked_rows(S_curr, likelihood_neuron_mask),
            _masked_rows(lam_a, likelihood_neuron_mask),
            _masked_rows(lam_b, likelihood_neuron_mask),
            self.InterpOptions,
            0.0,
        )
        x = (1.0 - beta) * x_a + beta * x_b
        z = (1.0 - beta) * z_a + beta * z_b
        return (
            x,
            z,
            torch.as_tensor([c_a_primary + 1, c_b_primary + 1], dtype=TORCH_DTYPE, device=self.device)
            .reshape(2, 1)
            .expand(2, z.shape[1]),
            torch.cat([(k_a_primary + 1).to(TORCH_DTYPE), (k_b_primary + 1).to(TORCH_DTYPE)], dim=0),
            torch.as_tensor([beta, alpha_a, alpha_b], dtype=TORCH_DTYPE, device=self.device)
            .reshape(3, 1)
            .expand(3, z.shape[1]),
        )

    def _estimate_within_library(
        self,
        Q: Tensor,
        S_curr: Tensor,
        f: Callable[[int], Tensor],
        likelihood_neuron_mask: Optional[Tensor],
        K_lengths: Sequence[int],
        c_primary: int,
        k_hats_primary: Sequence[int],
        library_id: int,
    ):
        k_primary = get_state_indices(k_hats_primary, f, K_lengths[c_primary]).to(self.device)
        x_primary, z_primary, lam_primary, alpha_primary = self._interp_adjacent_states(
            S_curr, c_primary, k_hats_primary, k_primary, likelihood_neuron_mask
        )
        try:
            c_secondary, k_hats_secondary = self._maximum_likelihood(
                Q,
                restricted_conds=[c_primary],
                allowed_library_ids=[library_id],
            )
        except ValueError:
            return x_primary, z_primary, lam_primary, alpha_primary, c_primary, k_primary

        k_secondary = get_state_indices(k_hats_secondary, f, K_lengths[c_secondary]).to(self.device)
        x_secondary, z_secondary, lam_secondary, _ = self._interp_adjacent_states(
            S_curr, c_secondary, k_hats_secondary, k_secondary, likelihood_neuron_mask
        )
        beta = fit_poisson_interp(
            _masked_rows(S_curr, likelihood_neuron_mask),
            _masked_rows(lam_primary, likelihood_neuron_mask),
            _masked_rows(lam_secondary, likelihood_neuron_mask),
            self.InterpOptions,
            0.0,
        )
        lam = (1.0 - beta) * lam_primary + beta * lam_secondary
        x = (1.0 - beta) * x_primary + beta * x_secondary
        z = (1.0 - beta) * z_primary + beta * z_secondary
        return x, z, lam, beta, c_primary, k_primary

    def _maximum_likelihood(
        self,
        Q: Tensor,
        restricted_conds: Optional[Sequence[int]] = None,
        restricted_library_ids: Optional[Sequence[int]] = None,
        allowed_library_ids: Optional[Sequence[int]] = None,
        states_to_exclude: Optional[Sequence[Tuple[int, int]]] = None,
        min_k_prime_dist: Optional[float] = None,
    ) -> Tuple[int, List[int]]:
        assert self.first_idx0 is not None and self.first_tau_prime_idx0 is not None
        q = Q.clone()
        q[self.first_tau_prime_idx0] = float("nan")
        lengths = (torch.cat([self.first_idx0[1:], torch.as_tensor([len(Q)], device=self.device)]) - self.first_idx0).tolist()
        if restricted_conds:
            for c0 in restricted_conds:
                q[self.first_idx0[c0] : self.first_idx0[c0] + lengths[c0]] = float("nan")
        if restricted_library_ids and self.library_ids is not None:
            restricted = {int(item) for item in restricted_library_ids}
            for c0, library_id in enumerate(self.library_ids.tolist()):
                if int(library_id) in restricted:
                    q[self.first_idx0[c0] : self.first_idx0[c0] + lengths[c0]] = float("nan")
        if allowed_library_ids is not None and self.library_ids is not None:
            allowed = {int(item) for item in allowed_library_ids}
            for c0, library_id in enumerate(self.library_ids.tolist()):
                if int(library_id) not in allowed:
                    q[self.first_idx0[c0] : self.first_idx0[c0] + lengths[c0]] = float("nan")
        if states_to_exclude:
            assert min_k_prime_dist is not None
            for c0, k_one in states_to_exclude:
                start = int(self.first_idx0[c0])
                center = start + k_one - 1
                exclude_start = max(int(center - min_k_prime_dist), start)
                exclude_end = min(int(center + min_k_prime_dist), start + lengths[c0] - 1)
                q[exclude_start : exclude_end + 1] = float("nan")

        if not bool(torch.isfinite(q).any()):
            raise ValueError("No finite MINT library state remained after candidate restrictions.")
        idx0 = int(torch.argmax(torch.nan_to_num(q, nan=-torch.inf)).item())
        c0, k1 = ind2ck(idx0, self.first_idx0)
        q_c = q[self.first_idx0[c0] : self.first_idx0[c0] + lengths[c0]]
        if lengths[c0] == self.tau_prime + 1:
            k2 = k1
        elif k1 > self.tau_prime + 1 and k1 < lengths[c0]:
            if q_c[k1 - 2] > q_c[k1]:
                k2 = k1 - 1
            else:
                k2 = k1 + 1
        elif k1 > self.tau_prime + 1:
            k2 = k1 - 1
        else:
            k2 = k1 + 1
        return c0, [k1, k2]

    def _interp_adjacent_states(
        self,
        S_curr: Tensor,
        c0: int,
        k_hats: Sequence[int],
        k_idx: Tensor,
        likelihood_neuron_mask: Optional[Tensor] = None,
    ):
        assert self.first_idx0 is not None and self.V is not None
        offsets = torch.arange(-self.tau_prime, 1, dtype=torch.long, device=self.device)
        idx1 = ck2ind(c0, torch.as_tensor(k_hats[0], device=self.device) + offsets, self.first_idx0)
        idx2 = ck2ind(c0, torch.as_tensor(k_hats[1], device=self.device) + offsets, self.first_idx0)
        lambda1 = self.rates[self.V[idx1]].T
        lambda2 = self.rates[self.V[idx2]].T
        alpha = fit_poisson_interp(
            _masked_rows(S_curr, likelihood_neuron_mask),
            _masked_rows(lambda1, likelihood_neuron_mask),
            _masked_rows(lambda2, likelihood_neuron_mask),
            self.InterpOptions,
            0.0,
        )
        lam = (1.0 - alpha) * lambda1 + alpha * lambda2
        x = (1.0 - alpha) * self.Omega_plus[c0][:, k_idx[0]] + alpha * self.Omega_plus[c0][:, k_idx[1]]
        z = (1.0 - alpha) * self.Phi_plus[c0][:, k_idx[0]] + alpha * self.Phi_plus[c0][:, k_idx[1]]
        return x, z, lam, alpha


class _MINTPreparedAdapter:
    """Score held-out counts from an already fitted MINT library."""

    task = "nlb"

    def fit(self, model, loader, device) -> None:
        if model.V is None:
            raise RuntimeError("Train MINT on the training loader before evaluation.")

    def evaluate(self, model, loader, device):
        from nlb2.metrics import EvaluationResult, compute_available_metrics

        predictions, targets = [], []
        with torch.no_grad():
            for batch in loader:
                output = model(batch["heldin_spikes"].to(device))
                predictions.append(output.rates.detach().cpu())
                targets.append(batch["heldout_spikes"].cpu())
        pred = {"rates": torch.cat(predictions)}
        truth = {"spikes": torch.cat(targets)}
        return EvaluationResult(
            metrics=compute_available_metrics(pred, truth),
            predictions={key: value.numpy() for key, value in pred.items()},
            targets={key: value.numpy() for key, value in truth.items()},
        )


def _prepared_training_conditions(path: Path, group_name: str | None, dataset: str, n_trials: int) -> np.ndarray:
    """Use saved training condition membership; unmatched trials keep unique templates."""
    with h5py.File(path, "r") as handle:
        group = handle[group_name] if group_name else handle[dataset] if dataset in handle else handle
        if "train_cond_idx" not in group:
            return np.arange(n_trials)
        rows = group["train_cond_idx"][()]
        conditions = np.arange(n_trials) + len(rows)
        assigned = np.zeros(n_trials, dtype=bool)
        for cond, row in enumerate(rows):
            idx = np.asarray(row, dtype=np.int64).reshape(-1)
            idx = idx[(idx >= 0) & (idx < n_trials)]
            if np.any(assigned[idx]):
                raise ValueError("MINT training condition memberships overlap.")
            conditions[idx] = cond
            assigned[idx] = True
    return conditions


class _MINTAllenVCNAdapter:
    """Fit a full-neuron Allen VCN MINT library and score held-out neurons."""

    task = "nlb"

    def __init__(
        self,
        condition_mode: str = "condition_id",
        library_source: str = "spikes",
        lfads_run_dir: str | None = None,
    ) -> None:
        if condition_mode not in {"condition_id", "trial_index"}:
            raise ValueError(f"Unsupported Allen VCN MINT condition mode: {condition_mode}")
        if library_source not in {"spikes", "lfads_checkpoint"}:
            raise ValueError(f"Unsupported Allen VCN MINT library source: {library_source}")
        self.condition_mode = condition_mode
        self.library_source = library_source
        self.lfads_run_dir = lfads_run_dir

    def fit(self, model: BaseDynamicsModel, loader, device: torch.device) -> None:
        if not isinstance(model, MINT):
            raise TypeError("_MINTAllenVCNAdapter requires a MINT model.")
        full_trials: list[Tensor] = []
        heldin_trials: list[Tensor] = []
        conditions: list[int] = []
        with torch.no_grad():
            for batch in loader:
                full = batch.get("full_spikes") if isinstance(batch, dict) else None
                if full is None:
                    raise KeyError("Allen VCN MINT requires full_spikes in training batches.")
                full = full.to(device=device, dtype=TORCH_DTYPE)
                heldin = batch.get("heldin_spikes") if isinstance(batch, dict) else None
                if heldin is None:
                    heldin = batch.get("spikes") if isinstance(batch, dict) else None
                if heldin is None:
                    raise KeyError("Allen VCN MINT requires held-in spikes in training batches.")
                heldin = heldin.to(device=device, dtype=TORCH_DTYPE)
                cond = batch.get("condition_id") if isinstance(batch, dict) else None
                cond_values = cond.reshape(-1) if cond is not None else None
                for idx, (trial, heldin_trial) in enumerate(zip(full, heldin)):
                    full_trials.append(trial.T.contiguous())
                    heldin_trials.append(heldin_trial.contiguous())
                    if self.condition_mode == "trial_index" or cond_values is None:
                        conditions.append(len(full_trials) - 1)
                    else:
                        conditions.append(int(cond_values[idx].item()))
        if not full_trials:
            raise RuntimeError("Allen VCN MINT received no training trials.")
        if self.library_source == "lfads_checkpoint":
            rate_trials = self._lfads_rate_trials(
                heldin_trials,
                n_time=int(full_trials[0].shape[1]),
                n_full=int(full_trials[0].shape[0]),
                device=device,
            )
            previous_source = model.settings.lorenz_library_source
            model.settings.lorenz_library_source = "true_rates"
            try:
                model.fit_library(full_trials, rate_trials, conditions)
            finally:
                model.settings.lorenz_library_source = previous_source
            return
        model.fit_library(full_trials, full_trials, conditions)

    def _lfads_rate_trials(
        self,
        heldin_trials: Sequence[Tensor],
        *,
        n_time: int,
        n_full: int,
        device: torch.device,
    ) -> list[Tensor]:
        if self.lfads_run_dir is None:
            raise ValueError("Allen VCN MINT lfads_checkpoint source requires allen_lfads_run_dir.")
        from nlb2.models.base import BaseModelConfig

        run_dir = Path(self.lfads_run_dir)
        config_path = run_dir / "config.json"
        checkpoint_path = run_dir / "best_model.pt"
        if not checkpoint_path.exists():
            checkpoint_path = run_dir / "model.pt"
        if not config_path.exists() or not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Missing LFADS config/checkpoint in {run_dir} for Allen VCN MINT library."
            )
        config = json.loads(config_path.read_text())
        model_config = BaseModelConfig.from_dict(config["model"])
        n_heldin = int(heldin_trials[0].shape[-1])
        lfads = model_config.build(n_neurons=n_heldin, n_time=n_time).to(device)
        state = torch.load(checkpoint_path, map_location=device)
        lfads.load_state_dict(state)
        lfads.eval()
        dt = float(getattr(lfads, "dt", 1.0))
        rates: list[Tensor] = []
        with torch.no_grad():
            for start in range(0, len(heldin_trials), 16):
                batch = torch.stack(list(heldin_trials[start : start + 16]), dim=0).to(
                    device=device,
                    dtype=torch.float32,
                )
                pred = lfads.predict_rates(batch)
                if pred.shape[-1] != n_full:
                    raise ValueError(
                        f"LFADS library predicted {pred.shape[-1]} neurons, expected {n_full}."
                    )
                count_rates = (pred * dt).to(dtype=TORCH_DTYPE)
                rates.extend([trial.T.contiguous() for trial in count_rates])
        return rates

    def evaluate(self, model: BaseDynamicsModel, loader, device: torch.device):
        if not isinstance(model, MINT):
            raise TypeError("_MINTAllenVCNAdapter requires a MINT model.")
        from nlb2.metrics import EvaluationResult, compute_available_metrics

        predictions: list[Tensor] = []
        targets: list[Tensor] = []
        with torch.no_grad():
            for batch in loader:
                if not isinstance(batch, dict):
                    raise TypeError("Allen VCN MINT evaluation requires dict batches.")
                heldin = batch["heldin_spikes"].to(device=device, dtype=TORCH_DTYPE)
                heldout = batch["heldout_spikes"].to(device=device, dtype=TORCH_DTYPE)
                zeros = torch.zeros_like(heldout)
                full_query = torch.cat([heldin, zeros], dim=-1)
                n_heldin = int(heldin.shape[-1])
                n_heldout = int(heldout.shape[-1])
                mask = torch.zeros(n_heldin + n_heldout, dtype=torch.bool, device=device)
                mask[:n_heldin] = True
                trials = [trial.T.contiguous() for trial in full_query]
                rates, _ = model.predict_spike_trials(
                    trials,
                    likelihood_neuron_mask=mask,
                    verbose=False,
                )
                pred = torch.stack(
                    [item[n_heldin : n_heldin + n_heldout].T for item in rates],
                    dim=0,
                )
                pred = torch.nan_to_num(
                    pred,
                    nan=1.0e-9,
                    posinf=1.0e6,
                    neginf=1.0e-9,
                ).clamp_min(1.0e-9)
                predictions.append(pred.detach().cpu())
                targets.append(heldout.detach().cpu())
        pred_dict = {"rates": torch.cat(predictions, dim=0)}
        target_dict = {"spikes": torch.cat(targets, dim=0)}
        metrics = compute_available_metrics(pred_dict, target_dict)
        return EvaluationResult(
            metrics=metrics,
            predictions={key: value.numpy() for key, value in pred_dict.items()},
            targets={key: value.numpy() for key, value in target_dict.items()},
        )


def get_mint_config(dataset: str, *, nlb_neural_state_defaults: bool = False) -> Tuple[Settings, HyperParams]:
    """Load paper reproduction presets; prepared data overrides sampling/layout."""
    from importlib.resources import files
    import yaml

    presets = yaml.safe_load(files("nlb2.models").joinpath("mint_presets.yaml").read_text())
    if dataset not in presets:
        raise ValueError(f"Unknown MINT dataset: {dataset}")
    preset = presets[dataset]
    settings_values = dict(preset.get("settings", {}))
    hp_values = dict(preset.get("hyperparams", {}))
    if nlb_neural_state_defaults:
        hp_values.update(preset.get("nlb_hyperparams", {}))
    for name in ("trial_alignment", "test_alignment"):
        if name in settings_values:
            settings_values[name] = range(*settings_values[name])
    if "trajectories_alignment" in hp_values:
        hp_values["trajectories_alignment"] = range(*hp_values["trajectories_alignment"])
    return Settings(task=dataset, data_path=None, results_path=None, **settings_values), HyperParams(**hp_values)


def as_tensor(array: np.ndarray, device: Optional[torch.device] = None) -> Tensor:
    return torch.as_tensor(array, dtype=TORCH_DTYPE, device=device)


def bin_data(data: Tensor, bin_size: int, method: str) -> Tensor:
    data = data.to(TORCH_DTYPE)
    n_bins = data.shape[1] // bin_size
    trimmed = data[:, : n_bins * bin_size]
    reshaped = trimmed.reshape(data.shape[0], n_bins, bin_size)
    if method == "mean":
        return torch.nanmean(reshaped, dim=2)
    if method == "sum":
        return torch.sum(reshaped, dim=2)
    raise ValueError(f"Unrecognized binning method: {method}")


def gaussian_window(length: int, sigma: int) -> np.ndarray:
    center = (length - 1) / 2.0
    n = np.arange(length, dtype=np.float64) - center
    return np.exp(-0.5 * (n / sigma) ** 2)


def gauss_filt(spikes: np.ndarray, sigma: int, bin_size: int) -> np.ndarray:
    spikes = np.asarray(spikes, dtype=np.float64)
    nan_mask = np.any(np.isnan(spikes), axis=0)
    had_nan = bool(np.any(nan_mask))
    prepend_nan = False
    if had_nan:
        nan_idx = np.flatnonzero(nan_mask)
        if not np.all(np.diff(nan_idx) == 1):
            raise ValueError("Non-consecutive NaNs encountered while filtering.")
        prepend_nan = bool(nan_mask[0])
        if not prepend_nan and not bool(nan_mask[-1]):
            raise ValueError("Time series broken up by a stretch of NaNs.")
        spikes_work = spikes[:, ~nan_mask]
    else:
        spikes_work = spikes

    width = 4
    pad = width * sigma
    length = 2 * pad + 1
    kernel = gaussian_window(length, sigma)
    kernel = kernel / kernel.sum() * bin_size

    pre = np.repeat(np.mean(spikes_work[:, :sigma], axis=1, dtype=np.float64)[:, None], pad, axis=1)
    post = np.repeat(np.mean(spikes_work[:, -sigma:], axis=1, dtype=np.float64)[:, None], pad, axis=1)
    padded = np.concatenate([pre, spikes_work, post], axis=1)

    filtered = np.zeros_like(padded, dtype=np.float64)
    for n in range(padded.shape[0]):
        conv = np.convolve(padded[n], kernel)
        filtered[n] = conv[pad : conv.shape[0] - pad]

    filtered = filtered[:, pad : filtered.shape[1] - pad]
    if had_nan:
        nan_block = np.full((filtered.shape[0], int(nan_mask.sum())), np.nan, dtype=np.float64)
        filtered = np.concatenate([nan_block, filtered], axis=1) if prepend_nan else np.concatenate([filtered, nan_block], axis=1)
    return filtered


def _pca_coeff(data: Tensor, n_components: int) -> Tensor:
    centered = data - torch.mean(data, dim=0, keepdim=True)
    if centered.shape[0] <= 1:
        cov = centered.T @ centered
    else:
        cov = centered.T @ centered / (centered.shape[0] - 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    return eigvecs[:, order][:, :n_components]


def smooth_average(grouped_trials: Sequence[Sequence[Tensor]], hyperparams, Ts: float) -> List[Tensor]:
    x_avg = [torch.nanmean(torch.stack(list(group), dim=2), dim=2) for group in grouped_trials]
    all_avg = torch.cat(x_avg, dim=1)
    soft_norm = hyperparams.soft_norm * hyperparams.Delta * Ts
    mu = torch.mean(all_avg, dim=1, keepdim=True)
    norm_factor = 1.0 / (soft_norm + torch.max(all_avg, dim=1, keepdim=True).values)

    normalized: List[List[Tensor]] = []
    for group in grouped_trials:
        normalized.append([(trial - mu) * norm_factor for trial in group])

    if hyperparams.n_trial_dims is not None:
        for c, group in enumerate(normalized):
            rows = [trial.T.reshape(1, -1) for trial in group]
            x_nt = torch.cat(rows, dim=0).T
            coeff = _pca_coeff(x_nt, int(hyperparams.n_trial_dims))
            projected = coeff @ coeff.T @ x_nt.T
            n_neurons, n_times = group[0].shape
            normalized[c] = [projected[i].reshape(n_times, n_neurons).T for i in range(len(group))]

    x_bar = [torch.nanmean(torch.stack(group, dim=2), dim=2) for group in normalized]

    if hyperparams.n_neural_dims is not None:
        data = torch.cat(x_bar, dim=1).T
        coeff = _pca_coeff(data, int(hyperparams.n_neural_dims))
        x_bar = [coeff @ coeff.T @ item for item in x_bar]

    if hyperparams.n_cond_dims is not None:
        n_conds = len(x_bar)
        n_neurons, n_times = x_bar[0].shape
        x_bar_nt = torch.cat([item.T.reshape(1, -1) for item in x_bar], dim=0).T
        coeff = _pca_coeff(x_bar_nt, int(hyperparams.n_cond_dims))
        projected = coeff @ coeff.T @ x_bar_nt.T
        x_bar = [projected[i].reshape(n_times, n_neurons).T for i in range(n_conds)]

    return [torch.clamp(item / norm_factor + mu, min=0.0) for item in x_bar]


def get_rate_indices(lambda_values: Tensor, lambda_range: Tensor, n_rates: int) -> Tensor:
    lam_min = lambda_range[0]
    lam_max = lambda_range[1]
    clipped = torch.clamp(lambda_values, min=float(lam_min), max=float(lam_max))
    scaled = (clipped - lam_min) / (lam_max - lam_min) * (n_rates - 1) + 1.0
    matlab_uint = torch.floor(scaled + 0.5).to(torch.long)
    return torch.clamp(matlab_uint - 1, 0, n_rates - 1)


def ck2ind(c0: int, k_one_based: Tensor, first_idx0: Tensor) -> Tensor:
    return first_idx0[c0] + k_one_based.to(torch.long) - 1


def ind2ck(index0: int, first_idx0: Tensor) -> Tuple[int, int]:
    starts = first_idx0.cpu().numpy()
    c0 = int(np.searchsorted(starts, index0, side="right") - 1)
    return c0, int(index0 - starts[c0] + 1)


def get_time_indices(
    t_prime_one: int,
    T_prime: int,
    T: int,
    Delta: int,
    tau_prime: int,
    causal: bool,
) -> Tuple[Tensor, Callable[[int], Tensor]]:
    t = t_prime_one * Delta
    if not causal:
        tau = (tau_prime + 1) * Delta - 1
        adjustment = round((tau + 1 + Delta) / 2)
        t = t - adjustment
    t_idx = list(range(t, t + Delta))
    if t_prime_one == tau_prime + 1:
        t_idx = list(range(1, t_idx[0])) + t_idx
    if t_prime_one == T_prime and t_idx[-1] < T:
        t_idx = t_idx + list(range(t_idx[-1] + 1, T + 1))
    t_idx = [idx for idx in t_idx if idx <= T]
    t_idx0 = torch.as_tensor([idx - 1 for idx in t_idx], dtype=torch.long)

    def f(k_prime_one: int) -> Tensor:
        return (k_prime_one - t_prime_one) * Delta + t_idx0

    return t_idx0, f


def get_state_indices(k_prime_hats: Sequence[int], f: Callable[[int], Tensor], K: int) -> Tensor:
    out = torch.stack([f(k_prime_hats[0]), f(k_prime_hats[1])], dim=0)
    return torch.clamp(out, 0, K - 1)


def _masked_rows(tensor: Tensor, mask: Optional[Tensor]) -> Tensor:
    return tensor if mask is None else tensor[mask]


def fit_poisson_interp(S: Tensor, X1: Tensor, X2: Tensor, options: InterpOptions, default_alpha: float) -> float:
    x2_minus_x1 = X2 - X1
    x2_minus_x1_sum = torch.sum(x2_minus_x1)
    alpha = 0.5
    i = 0
    while i < options.max_iters:
        denom = X1 + alpha * x2_minus_x1
        fraction = x2_minus_x1 / denom
        deriv1 = torch.sum(S * fraction) - x2_minus_x1_sum
        deriv2 = -torch.sum(S * (fraction**2))
        alpha_step = float((deriv1 / deriv2).item())
        alpha = alpha - alpha_step
        if abs(alpha_step) < options.step_tol or alpha < 0.0 or alpha > 1.0:
            alpha = max(min(alpha, 1.0), 0.0)
            break
        i += 1
    if alpha != alpha:
        return default_alpha
    return alpha


def use_best_interp(S_curr: Tensor, interps, likelihood_neuron_mask: Optional[Tensor] = None):
    S_curr = _masked_rows(S_curr, likelihood_neuron_mask)
    scores = []
    for interp in interps:
        lam = _masked_rows(interp[2], likelihood_neuron_mask)
        scores.append(torch.sum(S_curr * torch.log(lam) - lam))
    idx = int(torch.argmax(torch.stack(scores)).item())
    return interps[idx]


def _decode_char(dataset: h5py.Dataset) -> str:
    arr = np.asarray(dataset[()])
    return "".join(chr(int(x)) for x in arr.ravel() if int(x) != 0)


def _decode_cellstr(file: h5py.File, dataset: h5py.Dataset) -> np.ndarray:
    values: List[str] = []
    for ref in np.asarray(dataset[()]).ravel():
        values.append(_decode_char(file[ref]))
    return np.asarray(values, dtype=object)


class MintMatFile:
    def __init__(self, path: Path, dataset: str):
        self.path = Path(path)
        self.dataset = dataset

    def load(self) -> Tuple[MatTable, MatTable]:
        if self.dataset not in DATASET_FIELDS:
            raise ValueError(f"Unknown dataset: {self.dataset}")
        with h5py.File(self.path, "r") as file:
            t_fields = self._load_group(file, DATASET_FIELDS[self.dataset]["T"])
            trial_fields = self._load_group(file, DATASET_FIELDS[self.dataset]["TrialInfo"])
        return MatTable(t_fields), MatTable(trial_fields)

    @staticmethod
    def _load_group(file: h5py.File, mapping: Mapping[str, str]) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        refs = file["#refs#"]
        for field, ref_name in mapping.items():
            dataset = refs[ref_name]
            matlab_class = dataset.attrs.get("MATLAB_class")
            if matlab_class == b"cell":
                out[field] = _decode_cellstr(file, dataset)
            else:
                arr = np.asarray(dataset[()])
                if arr.ndim == 2 and arr.shape[0] == 1:
                    arr = arr.ravel()
                out[field] = arr
        return out


def find_time_index(time: np.ndarray, value: float) -> int:
    idx = np.flatnonzero(time == value)
    if idx.size == 0:
        idx = np.flatnonzero(np.isclose(time, value, rtol=0.0, atol=1e-12))
    if idx.size == 0:
        raise ValueError(f"Could not find time value {value}")
    return int(idx[0])


def take_time_window(matrix: np.ndarray, indices: Sequence[int], pad_value: float = np.nan) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    valid = (indices >= 0) & (indices < matrix.shape[1])
    out = np.full((matrix.shape[0], len(indices)), pad_value, dtype=np.float64)
    out[:, valid] = matrix[:, indices[valid]]
    return out


def stack_rows(parts: Iterable[np.ndarray]) -> np.ndarray:
    return np.vstack([np.asarray(part, dtype=np.float64) for part in parts])


def _alignment_array(values: range) -> np.ndarray:
    return np.asarray(list(values), dtype=np.int64)


def _mask_by_alignment(source: range, target: range) -> np.ndarray:
    return np.isin(_alignment_array(source), _alignment_array(target))


def _split_labels(split) -> List[str]:
    if isinstance(split, (list, tuple, set)):
        labels = []
        for item in split:
            labels.extend(_split_labels(item))
        return list(dict.fromkeys(labels))
    if split == "train":
        return ["train"]
    if split == "val":
        return ["val"]
    if split == "test":
        return ["val"]
    if split == "trainval":
        return ["train", "val"]
    raise ValueError(f"Unsupported split: {split}")


def _spikes_tensor(spikes: np.ndarray, device: Optional[torch.device] = None) -> Tensor:
    spikes = np.asarray(spikes)
    finite = spikes[np.isfinite(spikes)]
    if finite.size and finite.min() >= 0 and finite.max() <= 255:
        return torch.as_tensor(spikes, dtype=torch.uint8, device=device)
    return as_tensor(spikes, device)


def get_trial_data(settings, split: str, max_trials: Optional[int] = None, device: Optional[torch.device] = None):
    T, TrialInfo = MintMatFile(settings.data_path, settings.task).load()
    if settings.task == "area2_bump":
        return _area2_get_trial_data(T, TrialInfo, settings, split, max_trials, device)
    if settings.task == "mc_maze":
        return _mc_maze_get_trial_data(T, TrialInfo, settings, split, max_trials, device)
    if settings.task == "mc_rtt":
        return _mc_rtt_get_trial_data(T, TrialInfo, settings, split, max_trials, device)
    raise ValueError(f"Unknown task: {settings.task}")


def _area2_get_trial_data(T, TrialInfo, settings, split, max_trials, device):
    good = TrialInfo["result"] == "R"
    TrialInfo = TrialInfo.subset_trials(good)
    cond_mat = np.column_stack([TrialInfo["cond_dir"], TrialInfo["ctr_hold_bump"].astype(np.float64)])
    cond_list = np.unique(cond_mat, axis=0)
    S, Z, condition = [], [], []
    alignment = _alignment_array(settings.trial_alignment)
    time = T["time"]
    for tr in range(TrialInfo.n_trials):
        move_idx = find_time_index(time, TrialInfo["move_onset_time"][tr])
        idx = move_idx + alignment
        spikes = stack_rows([take_time_window(T["heldout_spikes"], idx), take_time_window(T["spikes"], idx)])
        behavior = stack_rows(
            [
                take_time_window(T["hand_pos"], idx),
                take_time_window(T["hand_vel"], idx),
                take_time_window(T["force"], idx),
                take_time_window(T["joint_ang"], idx),
                take_time_window(T["joint_vel"], idx),
                take_time_window(T["muscle_len"], idx),
                take_time_window(T["muscle_vel"], idx),
            ]
        )
        cond_row = np.asarray([TrialInfo["cond_dir"][tr], float(TrialInfo["ctr_hold_bump"][tr])])
        cond = int(np.flatnonzero(np.all(cond_list == cond_row, axis=1))[0])
        S.append(_spikes_tensor(spikes, device))
        Z.append(as_tensor(behavior, device))
        condition.append(cond)
    idx = np.flatnonzero(np.isin(TrialInfo["split"], _split_labels(split)))
    idx = idx[:max_trials] if max_trials is not None else idx
    return [S[i] for i in idx], [Z[i] for i in idx], np.asarray([condition[i] for i in idx]), cond_list


def _mc_maze_get_trial_data(T, TrialInfo, settings, split, max_trials, device):
    cond_mat = np.column_stack([TrialInfo["trial_type"], TrialInfo["trial_version"]])
    cond_list = np.unique(cond_mat, axis=0)
    S, Z, condition = [], [], []
    alignment = _alignment_array(settings.trial_alignment)
    time = T["time"]
    for tr in range(TrialInfo.n_trials):
        move_idx = find_time_index(time, TrialInfo["move_onset_time"][tr])
        idx = move_idx + alignment
        spikes = stack_rows([take_time_window(T["heldout_spikes"], idx), take_time_window(T["spikes"], idx)])
        behavior = stack_rows([take_time_window(T["hand_pos"], idx), take_time_window(T["hand_vel"], idx)])
        cond_row = np.asarray([TrialInfo["trial_type"][tr], TrialInfo["trial_version"][tr]])
        cond = int(np.flatnonzero(np.all(cond_list == cond_row, axis=1))[0])
        S.append(_spikes_tensor(spikes, device))
        Z.append(as_tensor(behavior, device))
        condition.append(cond)
    idx = np.flatnonzero(np.isin(TrialInfo["split"], _split_labels(split)))
    idx = idx[:max_trials] if max_trials is not None else idx
    return [S[i] for i in idx], [Z[i] for i in idx], np.asarray([condition[i] for i in idx]), cond_list


def _mc_rtt_get_trial_data(T, TrialInfo, settings, split, max_trials, device):
    if split == "train":
        train_idx = np.flatnonzero(TrialInfo["split"] == "train")
        last_train = int(train_idx[-1])
        end_time = TrialInfo["end_time"][last_train]
        T = T.subset_time(T["time"] <= end_time)
        return _mc_rtt_get_continuous_data(T, max_trials, device)
    if split == "trainval":
        public_idx = np.flatnonzero(np.isin(TrialInfo["split"], ["train", "val"]))
        last_public = int(public_idx[-1])
        end_time = TrialInfo["end_time"][last_public]
        T = T.subset_time(T["time"] <= end_time)
        return _mc_rtt_get_continuous_data(T, max_trials, device)
    raise ValueError("MC_RTT MINT NLB training supports train or trainval splits.")


def _mc_rtt_get_continuous_data(T: MatTable, max_trials, device):
    gap_mask = np.isnan(T["finger_pos"][0]) | np.isnan(T["autolfads_rates"][0])
    diff = np.diff(gap_mask.astype(np.int8))
    starts = np.concatenate([[0], np.flatnonzero(diff == -1) + 1])
    ends = np.concatenate([np.flatnonzero(diff == 1), [len(gap_mask) - 1]])
    S, Z = [], []
    for start, end in zip(starts, ends):
        idx = np.arange(start, end + 1)
        spikes = stack_rows([T["heldout_spikes"][:, idx], T["spikes"][:, idx]])
        behavior = stack_rows([T["finger_pos"][:2, idx], T["finger_vel"][:, idx], T["autolfads_rates"][:, idx]])
        S.append(_spikes_tensor(spikes, device))
        Z.append(as_tensor(behavior, device))
    if max_trials is not None:
        S = S[:max_trials]
        Z = Z[:max_trials]
    condition = np.arange(len(S), dtype=np.int64)
    return S, Z, condition, condition[:, None]


def preprocess_behavior(Z: Sequence[Tensor], settings):
    if settings.task == "area2_bump":
        zero_idx = list(settings.trial_alignment).index(0)
        out = []
        for item in Z:
            pos = item[:2] - item[:2, zero_idx : zero_idx + 1]
            out.append(torch.cat([pos, item[2:]], dim=0))
        labels = (
            ["xpos", "ypos", "xvel", "yvel"]
            + [f"force_{i}" for i in range(1, 7)]
            + [f"joint_ang_{i}" for i in range(1, 8)]
            + [f"joint_vel_{i}" for i in range(1, 8)]
            + [f"muscle_len_{i}" for i in range(1, 40)]
            + [f"muscle_vel_{i}" for i in range(1, 40)]
        )
        return out, labels
    if settings.task == "mc_maze":
        zero_idx = list(settings.trial_alignment).index(0)
        out = []
        for item in Z:
            pos = item[:2] - item[:2, zero_idx : zero_idx + 1]
            out.append(torch.cat([pos, item[2:]], dim=0))
        return out, ["xpos", "ypos", "xvel", "yvel"]
    if settings.task == "mc_rtt":
        return [item[2:4] for item in Z], ["xvel", "yvel"]
    if settings.task == "dmfc_rsg":
        labels = ["is_eye", "theta", "is_short", "ts", "tp"]
        return list(Z), labels
    raise ValueError(f"Unknown task: {settings.task}")


def _fit_dmfc_rsg_trajectories(S, Z, condition, settings, hyperparams):
    S_smooth = [as_tensor(gauss_filt(spikes.cpu().numpy(), hyperparams.sigma, hyperparams.Delta), spikes.device) for spikes in S]
    _, labels = preprocess_behavior(Z, settings)
    condition = np.asarray(condition, dtype=np.int64)
    condition_rows = np.asarray(settings.dmfc_condition_rows, dtype=np.float64)
    section_ids = np.asarray(settings.dmfc_section_ids, dtype=np.int64)
    cond_ids = np.unique(condition)
    cond_trial_indices = [np.flatnonzero(condition == cond) for cond in cond_ids]
    cond_rows = [condition_rows[trial_idx[0]] for trial_idx in cond_trial_indices]
    z_rows = [
        torch.nanmean(torch.stack([Z[i][:, 0].to(TORCH_DTYPE) for i in trial_idx], dim=1), dim=1)
        .detach()
        .cpu()
        .numpy()
        for trial_idx in cond_trial_indices
    ]

    x_session_raw, set_go_masks = _dmfc_event_average_library(
        S_smooth,
        condition,
        condition_rows,
        cond_ids,
        cond_rows,
        settings,
        base_mask=None,
    )
    x_session = smooth_average([[item] for item in x_session_raw], hyperparams, settings.Ts)
    z_session = _dmfc_condition_state_trajectories(z_rows, len(settings.trial_alignment), S[0].device)

    section_count = int(hyperparams.dmfc_section_count)
    if section_count <= 1:
        return x_session, z_session, labels

    raw_scaled = [spikes.to(TORCH_DTYPE) * float(hyperparams.Delta) for spikes in S]
    x_libraries: List[Tensor] = []
    z_libraries: List[Tensor] = []
    library_ids: List[int] = []
    for section in range(section_count):
        section_mask = section_ids == section
        section_target, _ = _dmfc_event_average_library(
            raw_scaled,
            condition,
            condition_rows,
            cond_ids,
            cond_rows,
            settings,
            base_mask=section_mask,
        )
        transform = _fit_dmfc_identity_ridge_transform(x_session, section_target, set_go_masks, hyperparams)
        for source, z_cond in zip(x_session, z_session):
            x_libraries.append(_apply_dmfc_transform(source, transform))
            z_libraries.append(z_cond)
            library_ids.append(section)

    return x_libraries, z_libraries, labels, np.asarray(library_ids, dtype=np.int64)


def _dmfc_condition_state_trajectories(
    cond_rows: Sequence[np.ndarray],
    n_time: int,
    device: Optional[torch.device],
) -> List[Tensor]:
    out = []
    for row in cond_rows:
        values = np.repeat(np.asarray(row, dtype=np.float64)[:, None], n_time, axis=1)
        out.append(as_tensor(values, device))
    return out


def _dmfc_event_average_library(
    rate_trials: Sequence[Tensor],
    condition: np.ndarray,
    condition_rows: np.ndarray,
    cond_ids: Sequence[int],
    cond_rows: Sequence[np.ndarray],
    settings,
    base_mask: Optional[np.ndarray],
) -> Tuple[List[Tensor], List[Tensor]]:
    alignment = _alignment_array(settings.trial_alignment)
    align_start = int(alignment[0])
    events = settings.dmfc_event_offsets
    if events is None:
        raise ValueError("DMFC paper trajectory fitting requires event offsets.")
    n_trials = len(rate_trials)
    all_trials = np.ones(n_trials, dtype=bool)
    base = all_trials if base_mask is None else np.asarray(base_mask, dtype=bool)
    full_stop = int(settings.trial_alignment.stop)

    x_bar: List[Tensor] = []
    set_go_masks: List[Tensor] = []
    for cond, row in zip(cond_ids, cond_rows):
        epoch_defs = [
            ("fix_time", "target_on_time", 0, "median"),
            ("target_on_time", "ready_time", 1, "median"),
            ("ready_time", "set_time", 2, "median"),
            ("set_time", "go_time", 3, "mean_warp"),
            ("go_time", None, 4, "fixed"),
        ]
        pieces: List[Tensor] = []
        mask_pieces: List[Tensor] = []
        for start_field, end_field, epoch, mode in epoch_defs:
            pool = _dmfc_epoch_pool(condition, condition_rows, int(cond), row, epoch) & base
            if not np.any(pool):
                pool = _dmfc_epoch_pool(condition, condition_rows, int(cond), row, epoch)
            trial_idx = np.flatnonzero(pool)
            start_offsets = np.asarray(events[start_field], dtype=np.int64)
            if end_field is None:
                end_offsets = start_offsets + full_stop
                length = max(1, full_stop)
                piece = _dmfc_average_aligned_epoch(rate_trials, trial_idx, start_offsets, end_offsets, length, align_start)
            elif mode == "mean_warp":
                end_offsets = np.asarray(events[end_field], dtype=np.int64)
                length = _dmfc_epoch_length(start_offsets, end_offsets, trial_idx, "mean")
                piece = _dmfc_average_warped_epoch(rate_trials, trial_idx, start_offsets, end_offsets, length, align_start)
            else:
                end_offsets = np.asarray(events[end_field], dtype=np.int64)
                length = _dmfc_epoch_length(start_offsets, end_offsets, trial_idx, "median")
                piece = _dmfc_average_aligned_epoch(rate_trials, trial_idx, start_offsets, end_offsets, length, align_start)
            pieces.append(piece)
            mask_pieces.append(torch.full((piece.shape[1],), epoch == 3, dtype=torch.bool, device=piece.device))

        concat = torch.cat(pieces, dim=1)
        set_go_concat = torch.cat(mask_pieces, dim=0)
        go_idx = sum(piece.shape[1] for piece in pieces[:4])
        trim_idx = torch.as_tensor(go_idx + alignment, dtype=torch.long, device=concat.device)
        trimmed = _take_time_window_tensor(concat, trim_idx)
        set_go_mask = _take_bool_window_tensor(set_go_concat, trim_idx)
        x_bar.append(torch.nan_to_num(trimmed, nan=0.0, posinf=0.0, neginf=0.0))
        set_go_masks.append(set_go_mask)
    return x_bar, set_go_masks


def _dmfc_epoch_pool(
    condition: np.ndarray,
    condition_rows: np.ndarray,
    cond: int,
    cond_row: np.ndarray,
    epoch: int,
) -> np.ndarray:
    if epoch == 0:
        return np.ones(condition.shape[0], dtype=bool)
    if epoch == 1:
        return np.all(np.isclose(condition_rows[:, :2], cond_row[:2]), axis=1)
    if epoch == 2:
        return np.all(np.isclose(condition_rows[:, :3], cond_row[:3]), axis=1)
    return condition == cond


def _dmfc_epoch_length(
    start_offsets: np.ndarray,
    end_offsets: np.ndarray,
    trial_idx: np.ndarray,
    mode: str,
) -> int:
    if trial_idx.size == 0:
        return 1
    durations = np.asarray(end_offsets[trial_idx] - start_offsets[trial_idx], dtype=np.float64)
    durations = durations[np.isfinite(durations) & (durations > 0)]
    if durations.size == 0:
        return 1
    if mode == "mean":
        return max(1, int(round(float(np.mean(durations)))))
    return max(1, int(round(float(np.median(durations)))))


def _dmfc_average_aligned_epoch(
    rate_trials: Sequence[Tensor],
    trial_idx: np.ndarray,
    start_offsets: np.ndarray,
    end_offsets: np.ndarray,
    length: int,
    align_start: int,
) -> Tensor:
    pieces = []
    for trial in trial_idx:
        start = int(start_offsets[trial] - align_start)
        end = int(end_offsets[trial] - align_start)
        if end <= start:
            continue
        pieces.append(_take_limited_time_window_tensor(rate_trials[trial], start, end, length))
    return _nanmean_tensor_list(pieces, rate_trials[0].shape[0], length, rate_trials[0].device)


def _dmfc_average_warped_epoch(
    rate_trials: Sequence[Tensor],
    trial_idx: np.ndarray,
    start_offsets: np.ndarray,
    end_offsets: np.ndarray,
    length: int,
    align_start: int,
) -> Tensor:
    pieces = []
    for trial in trial_idx:
        start = int(start_offsets[trial] - align_start)
        end = int(end_offsets[trial] - align_start)
        valid_start = max(start, 0)
        valid_end = min(end, rate_trials[trial].shape[1])
        if valid_end <= valid_start:
            continue
        segment = rate_trials[trial][:, valid_start:valid_end].to(TORCH_DTYPE)
        if segment.shape[1] == length:
            warped = segment
        elif segment.shape[1] == 1:
            warped = segment.expand(segment.shape[0], length)
        else:
            warped = torch.nn.functional.interpolate(
                segment.reshape(1, segment.shape[0], segment.shape[1]),
                size=length,
                mode="linear",
                align_corners=True,
            )[0]
        pieces.append(warped)
    return _nanmean_tensor_list(pieces, rate_trials[0].shape[0], length, rate_trials[0].device)


def _take_limited_time_window_tensor(trial: Tensor, start: int, end: int, length: int) -> Tensor:
    idx = torch.arange(start, start + length, dtype=torch.long, device=trial.device)
    valid = (idx >= 0) & (idx < trial.shape[1]) & (idx < end)
    out = torch.full((trial.shape[0], length), float("nan"), dtype=TORCH_DTYPE, device=trial.device)
    if bool(valid.any()):
        out[:, valid] = trial[:, idx[valid]].to(TORCH_DTYPE)
    return out


def _take_time_window_tensor(matrix: Tensor, idx: Tensor) -> Tensor:
    valid = (idx >= 0) & (idx < matrix.shape[1])
    out = torch.full((matrix.shape[0], idx.numel()), float("nan"), dtype=TORCH_DTYPE, device=matrix.device)
    if bool(valid.any()):
        out[:, valid] = matrix[:, idx[valid]]
    return out


def _take_bool_window_tensor(mask: Tensor, idx: Tensor) -> Tensor:
    valid = (idx >= 0) & (idx < mask.numel())
    out = torch.zeros(idx.numel(), dtype=torch.bool, device=mask.device)
    if bool(valid.any()):
        out[valid] = mask[idx[valid]]
    return out


def _nanmean_tensor_list(pieces: Sequence[Tensor], n_rows: int, length: int, device: torch.device) -> Tensor:
    if not pieces:
        return torch.zeros((n_rows, length), dtype=TORCH_DTYPE, device=device)
    stacked = torch.stack([piece.to(TORCH_DTYPE) for piece in pieces], dim=2)
    return torch.nan_to_num(torch.nanmean(stacked, dim=2), nan=0.0, posinf=0.0, neginf=0.0)


def _fit_dmfc_identity_ridge_transform(
    source: Sequence[Tensor],
    target: Sequence[Tensor],
    set_go_masks: Sequence[Tensor],
    hyperparams,
) -> Tuple[Tensor, Tensor]:
    n_neurons = source[0].shape[0]
    X = torch.cat([item.T for item in source], dim=0).detach().cpu().numpy()
    Y = torch.cat([item.T for item in target], dim=0).detach().cpu().numpy()
    weights = np.concatenate(
        [
            np.where(mask.detach().cpu().numpy(), float(hyperparams.dmfc_set_go_weight), 1.0)
            for mask in set_go_masks
        ]
    ).astype(np.float64)
    finite = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(Y), axis=1) & np.isfinite(weights)
    X = X[finite]
    Y = Y[finite]
    weights = weights[finite]
    if X.shape[0] == 0:
        identity = torch.eye(n_neurons, dtype=TORCH_DTYPE, device=source[0].device)
        bias = torch.zeros(n_neurons, dtype=TORCH_DTYPE, device=source[0].device)
        return identity, bias

    X_aug = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float64)], axis=1)
    sqrt_w = np.sqrt(weights)[:, None]
    Xw = X_aug * sqrt_w
    Yw = Y * sqrt_w
    penalty = np.zeros((n_neurons + 1, n_neurons + 1), dtype=np.float64)
    penalty[:n_neurons, :n_neurons] = float(hyperparams.dmfc_section_ridge) * np.eye(n_neurons)
    prior = np.zeros((n_neurons + 1, n_neurons), dtype=np.float64)
    prior[:n_neurons, :] = np.eye(n_neurons)
    lhs = Xw.T @ Xw + penalty
    rhs = Xw.T @ Yw + penalty @ prior
    try:
        coeff = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coeff = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    A = torch.as_tensor(coeff[:n_neurons], dtype=TORCH_DTYPE, device=source[0].device)
    b = torch.as_tensor(coeff[n_neurons], dtype=TORCH_DTYPE, device=source[0].device)
    return A, b


def _apply_dmfc_transform(source: Tensor, transform: Tuple[Tensor, Tensor]) -> Tensor:
    A, b = transform
    transformed = source.T @ A + b.reshape(1, -1)
    return torch.clamp(transformed.T, min=0.0)


def fit_trajectories(S, Z, condition, settings, hyperparams):
    if settings.library_rate_source in {"prepared_spikes", "prepared_rates"}:
        if settings.library_rate_source == "prepared_rates":
            rate_trials = [item.to(TORCH_DTYPE) for item in Z]
        elif hyperparams.sigma > 0:
            rate_trials = [as_tensor(gauss_filt(item.cpu().numpy(), hyperparams.sigma, hyperparams.Delta), item.device) for item in S]
        else:
            rate_trials = [item.to(TORCH_DTYPE) * hyperparams.Delta for item in S]
        if settings.task == "mc_rtt":
            grouped = [[item] for item in rate_trials]
        else:
            labels = np.asarray(condition)
            grouped = [[rate_trials[i] for i in np.flatnonzero(labels == cond)] for cond in np.unique(labels)]
        fitted = smooth_average(grouped, hyperparams, settings.Ts)
        return fitted, [item.clone() for item in fitted], [f"rate_{i}" for i in range(fitted[0].shape[0])]
    if (
        settings.task == "dmfc_rsg"
        and settings.dmfc_event_offsets is not None
        and settings.dmfc_condition_rows is not None
        and settings.dmfc_section_ids is not None
    ):
        return _fit_dmfc_rsg_trajectories(S, Z, condition, settings, hyperparams)

    if settings.task in {"area2_bump", "dmfc_rsg", "mc_maze"}:
        if settings.task == "dmfc_rsg" and getattr(settings, "library_rate_source", "spikes") == "lfads":
            S_smooth = [spikes.to(TORCH_DTYPE) for spikes in S]
        else:
            S_smooth = [as_tensor(gauss_filt(spikes.cpu().numpy(), hyperparams.sigma, hyperparams.Delta), spikes.device) for spikes in S]
        Z_proc, labels = preprocess_behavior(Z, settings)
        t_mask = _mask_by_alignment(settings.trial_alignment, hyperparams.trajectories_alignment)
        S_smooth = [item[:, t_mask] for item in S_smooth]
        Z_proc = [item[:, t_mask] for item in Z_proc]
        cond_list = np.unique(condition)
        grouped_x, grouped_z = [], []
        for cond in cond_list:
            trial_idx = np.flatnonzero(condition == cond)
            grouped_x.append([S_smooth[i] for i in trial_idx])
            grouped_z.append([Z_proc[i] for i in trial_idx])
        z_bar = [torch.mean(torch.stack(group, dim=2), dim=2) for group in grouped_z]
        x_bar = smooth_average(grouped_x, hyperparams, settings.Ts)
        return x_bar, z_bar, labels
    if settings.task == "mc_rtt":
        vel, labels = preprocess_behavior(Z, settings)
        rates = [item[4:] * settings.Ts * hyperparams.Delta for item in Z]
        return rates, vel, labels
    if settings.task in {"auto", "chaotic_rnn", "lorenz", "allen_vcn"}:
        source = getattr(settings, "lorenz_library_source", "smoothed_spikes")
        if source == "true_rates":
            rate_trials = [item.to(TORCH_DTYPE) for item in Z]
        elif source == "smoothed_spikes":
            if int(hyperparams.sigma) > 0:
                rate_trials = [
                    as_tensor(
                        gauss_filt(spikes.cpu().numpy(), int(hyperparams.sigma), hyperparams.Delta),
                        spikes.device,
                    )
                    for spikes in S
                ]
            else:
                rate_trials = [spikes.to(TORCH_DTYPE) * hyperparams.Delta for spikes in S]
        else:
            raise ValueError(f"Unknown Lorenz MINT library source: {source}")

        n_time = rate_trials[0].shape[1]
        if len(settings.trial_alignment) == n_time:
            t_mask = _mask_by_alignment(settings.trial_alignment, hyperparams.trajectories_alignment)
        else:
            t_mask = np.ones(n_time, dtype=bool)
        rate_trials = [item[:, t_mask] for item in rate_trials]
        cond_list = np.unique(condition)
        grouped_rates = []
        for cond in cond_list:
            trial_idx = np.flatnonzero(condition == cond)
            grouped_rates.append([rate_trials[i] for i in trial_idx])
        x_bar = smooth_average(grouped_rates, hyperparams, settings.Ts)
        return x_bar, [item.clone() for item in x_bar], [f"rate_{i}" for i in range(x_bar[0].shape[0])]
    raise ValueError(f"Unknown task: {settings.task}")


def default_train_nwb_path(dataset: str, nwb_dir: Path) -> Path:
    from nlb2.datasets.nlb import DATASET_TO_DANDISET, DATASET_TO_TRAIN_NWB

    return Path(nwb_dir) / DATASET_TO_DANDISET[dataset] / DATASET_TO_TRAIN_NWB[dataset]


def _to_ms(value) -> int:
    if hasattr(value, "total_seconds"):
        return int(round(value.total_seconds() * 1000.0))
    return int(round(float(value) * 1000.0))


def _field_matrix(ds, field: str, dtype=np.float64) -> np.ndarray:
    return ds.data[field].to_numpy(dtype=dtype).T


def _condition_index(cond_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    cond_list = np.unique(cond_mat, axis=0)
    condition = []
    for row in cond_mat:
        condition.append(int(np.flatnonzero(np.all(cond_list == row, axis=1))[0]))
    return np.asarray(condition, dtype=np.int64), cond_list


def get_nwb_trial_data(
    settings,
    split: str,
    nwb_path: Path,
    max_trials: Optional[int] = None,
    device: Optional[torch.device] = None,
):
    from nlb_tools.nwb_interface import NWBDataset

    ds = NWBDataset(Path(nwb_path))
    if settings.task == "area2_bump":
        return _area2_nwb_trial_data(ds, settings, split, max_trials, device)
    if settings.task == "dmfc_rsg":
        return _dmfc_nwb_trial_data(ds, settings, split, max_trials, device)
    if settings.task == "mc_maze":
        return _mc_maze_nwb_trial_data(ds, settings, split, max_trials, device)
    raise ValueError(f"Direct trial NWB loading is not implemented for {settings.task}.")


def _area2_nwb_trial_data(ds, settings, split, max_trials, device):
    trial_info = ds.trial_info
    split_mask = trial_info["split"].isin(_split_labels(split)).to_numpy()
    good_mask = (trial_info["result"] == "R").to_numpy()
    idx = np.flatnonzero(split_mask & good_mask)
    idx = idx if max_trials is None else idx[:max_trials]
    cond_mat = np.column_stack(
        [
            trial_info.iloc[idx]["cond_dir"].to_numpy(dtype=np.float64),
            trial_info.iloc[idx]["ctr_hold_bump"].to_numpy(dtype=np.float64),
        ]
    )
    condition, cond_list = _condition_index(cond_mat)
    heldout = _field_matrix(ds, "heldout_spikes")
    spikes = _field_matrix(ds, "spikes")
    behavior_fields = [
        _field_matrix(ds, "hand_pos"),
        _field_matrix(ds, "hand_vel"),
        _field_matrix(ds, "force"),
        _field_matrix(ds, "joint_ang"),
        _field_matrix(ds, "joint_vel"),
        _field_matrix(ds, "muscle_len"),
        _field_matrix(ds, "muscle_vel"),
    ]
    alignment = _alignment_array(settings.trial_alignment)
    S, Z = [], []
    for tr in idx:
        time_idx = _to_ms(trial_info.iloc[tr]["move_onset_time"]) + alignment
        spikes_trial = stack_rows([take_time_window(heldout, time_idx), take_time_window(spikes, time_idx)])
        behavior = stack_rows(take_time_window(field, time_idx) for field in behavior_fields)
        if np.isnan(behavior).any():
            raise ValueError(f"area2_bump trial {tr}: behavior window contains NaNs.")
        S.append(_spikes_tensor(spikes_trial, device))
        Z.append(as_tensor(behavior, device))
    return S, Z, condition, cond_list


def _mc_maze_nwb_trial_data(ds, settings, split, max_trials, device):
    trial_info = ds.trial_info
    split_mask = trial_info["split"].isin(_split_labels(split)).to_numpy()
    idx = np.flatnonzero(split_mask)
    idx = idx if max_trials is None else idx[:max_trials]
    cond_mat = np.column_stack(
        [
            trial_info.iloc[idx]["trial_type"].to_numpy(dtype=np.float64),
            trial_info.iloc[idx]["trial_version"].to_numpy(dtype=np.float64),
        ]
    )
    condition, cond_list = _condition_index(cond_mat)
    heldout = _field_matrix(ds, "heldout_spikes")
    spikes = _field_matrix(ds, "spikes")
    hand_pos = _field_matrix(ds, "hand_pos")
    hand_vel = _field_matrix(ds, "hand_vel")
    alignment = _alignment_array(settings.trial_alignment)
    S, Z = [], []
    for tr in idx:
        time_idx = _to_ms(trial_info.iloc[tr]["move_onset_time"]) + alignment
        spikes_trial = stack_rows([take_time_window(heldout, time_idx), take_time_window(spikes, time_idx)])
        behavior = stack_rows([take_time_window(hand_pos, time_idx), take_time_window(hand_vel, time_idx)])
        if np.isnan(behavior).any():
            raise ValueError(f"mc_maze trial {tr}: behavior window contains NaNs.")
        S.append(_spikes_tensor(spikes_trial, device))
        Z.append(as_tensor(behavior, device))
    return S, Z, condition, cond_list


def _dmfc_nwb_trial_data(ds, settings, split, max_trials, device):
    trial_info = ds.trial_info
    event_fields = ["fix_time", "target_on_time", "ready_time", "set_time", "go_time"]
    behavior_fields = ["is_eye", "theta", "is_short", "ts", "tp"]
    split_mask = trial_info["split"].isin(_split_labels(split)).to_numpy()
    event_mask = trial_info[event_fields].notna().all(axis=1).to_numpy()
    behavior = trial_info[behavior_fields].to_numpy(dtype=np.float64)
    behavior_mask = np.all(np.isfinite(behavior), axis=1)
    idx = np.flatnonzero(split_mask & event_mask & behavior_mask)
    idx = idx if max_trials is None else idx[:max_trials]

    cond_mat = behavior[idx, :4]
    condition, cond_list = _condition_index(cond_mat)
    settings.dmfc_condition_rows = cond_mat
    section_count = 6
    settings.dmfc_section_ids = np.floor(np.arange(len(idx), dtype=np.float64) * section_count / len(idx)).astype(np.int64)

    go_ms = trial_info.iloc[idx]["go_time"].map(_to_ms).to_numpy(dtype=np.int64)
    settings.dmfc_event_offsets = {
        field: trial_info.iloc[idx][field].map(_to_ms).to_numpy(dtype=np.int64) - go_ms
        for field in event_fields
    }

    heldout = _field_matrix(ds, "heldout_spikes")
    spikes = _field_matrix(ds, "spikes")
    alignment = _alignment_array(settings.trial_alignment)
    S, Z = [], []
    n_time = len(alignment)
    for local_idx, tr in enumerate(idx):
        time_idx = go_ms[local_idx] + alignment
        spikes_trial = stack_rows([take_time_window(heldout, time_idx), take_time_window(spikes, time_idx)])
        z = np.repeat(behavior[tr, :, None], n_time, axis=1)
        S.append(_spikes_tensor(spikes_trial, device))
        Z.append(as_tensor(z, device))
    return S, Z, condition, cond_list


def observed_neuron_mask(n_heldout: int, n_neurons: int, device=None) -> Tensor:
    mask = torch.ones(n_neurons, dtype=torch.bool, device=device)
    mask[:n_heldout] = False
    return mask
