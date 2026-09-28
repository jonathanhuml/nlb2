"""Dataset registry."""

from nlb2.datasets.chaotic_rnn import (
    ChaoticRNNDataset,
    ChaoticRNNDatasetConfig,
    generate_chaotic_rnn_data,
)
from nlb2.datasets.allen_vcn import (
    ALLEN_VCN_DATASETS,
    AllenVCNDataset,
    AllenVCNDatasetConfig,
    load_allen_vcn_h5,
)
from nlb2.datasets.ctd import CTDDataset, CTDDatasetConfig, load_ctd_h5
from nlb2.datasets.lorenz import LorenzDataset, LorenzDatasetConfig, generate_lorenz_data
from nlb2.datasets.mc_maze import MCMazeDataset, MCMazeDatasetConfig
from nlb2.datasets.nlb import (
    NLB_DATASETS,
    NLBDataset,
    NLBDatasetConfig,
    prepare_nlb_data,
)

__all__ = [
    "ALLEN_VCN_DATASETS",
    "AllenVCNDataset",
    "AllenVCNDatasetConfig",
    "ChaoticRNNDataset",
    "ChaoticRNNDatasetConfig",
    "CTDDataset",
    "CTDDatasetConfig",
    "LorenzDataset",
    "LorenzDatasetConfig",
    "MCMazeDataset",
    "MCMazeDatasetConfig",
    "NLB_DATASETS",
    "NLBDataset",
    "NLBDatasetConfig",
    "generate_chaotic_rnn_data",
    "generate_lorenz_data",
    "load_allen_vcn_h5",
    "load_ctd_h5",
    "prepare_nlb_data",
]
