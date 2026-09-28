"""Portable training checkpoints and isolated evaluation randomness."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch


def seed_training(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def capture_rng(device: str) -> dict[str, Any]:
    name, values, position, has_gauss, cached = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if torch.device(device).type == "cuda" else None,
        "numpy": [name, values.tolist(), position, has_gauss, cached],
        "python": random.getstate(),
    }


def restore_rng(state: dict[str, Any], device: str) -> None:
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.device(device).type == "cuda":
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)
    name, values, position, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.asarray(values, dtype=np.uint32), position, has_gauss, cached))
    random.setstate(state["python"])


@contextmanager
def evaluation_rng(seed: int | None, device: str):
    """Keep evaluation frequency from changing the training RNG stream."""
    state = capture_rng(device)
    try:
        if seed is not None:
            seed_training(seed)
        yield
    finally:
        restore_rng(state, device)


def save_training_checkpoint(path: Path, state: dict[str, Any]) -> None:
    """Atomically commit a state loadable with torch's weights-only loader."""
    temporary = path.with_suffix(".tmp")
    torch.save(_portable(state), temporary)
    temporary.replace(path)


def _portable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _portable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_portable(item) for item in value)
    return value


def validate_resume_config(saved: dict, current: dict) -> None:
    def comparable(payload):
        payload = deepcopy(payload)
        payload["trainer"].pop("device", None)
        for field in ("output_dir", "run_name", "save_predictions", "save_plots", "save_training_state"):
            payload["experiment"].pop(field, None)
        return payload

    if comparable(saved) != comparable(current):
        raise ValueError("Resume requires the same experiment, seeds, selection, and training budget.")
