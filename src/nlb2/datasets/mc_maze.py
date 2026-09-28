"""Backward-compatible MC_Maze aliases for the generic NLB dataset."""

from nlb2.datasets.nlb import NLBArrays as MCMazeArrays
from nlb2.datasets.nlb import NLBDataset as MCMazeDataset
from nlb2.datasets.nlb import NLBDatasetConfig as MCMazeDatasetConfig
from nlb2.datasets.nlb import load_nlb_h5 as load_mc_maze_h5

__all__ = [
    "MCMazeArrays",
    "MCMazeDataset",
    "MCMazeDatasetConfig",
    "load_mc_maze_h5",
]
