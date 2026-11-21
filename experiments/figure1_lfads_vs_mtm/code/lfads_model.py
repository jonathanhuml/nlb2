"""NLB2 LFADS with the exact hyperparameters of the original Figure 1 run (figure1_models.make_lfads)."""
import numpy as np
import torch
from nlb2.models.lfads import LFADSConfig
from figure1_data import NT, DT


def make_lfads(data, epochs, batch_size):
    steps = int(np.ceil(len(data['splits']['train']['counts']) / batch_size)) * epochs
    cfg = LFADSConfig(generator_dim=128, g0_encoder_dim=128, controller_encoder_dim=128,
                      controller_dim=64, factor_dim=64, inferred_input_dim=4,
                      keep_prob=.8, readout_neurons=len(data['unit_ids']), dt=DT,
                      kl_weight_schedule_dur=max(20, int(.25 * steps)),
                      l2_weight_schedule_dur=max(20, int(.25 * steps)),
                      l2_gen_scale=1e-4, l2_con_scale=1e-4,
                      loss_scale=1. / (NT * len(data['unit_ids'])))
    model = cfg.build(len(data['heldin']), NT)
    model.initialize_log_rate_bias_from_counts(torch.from_numpy(data['splits']['train']['counts']))
    return model
