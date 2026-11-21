"""Frozen pretrained MtM (BrainWideBench release) adapted to the Figure 1 protocol.

Mirrors the earlier POYO+ adapter: the shared pretrained backbone is frozen with
dropout disabled, and only new-session parameters are learned from neural targets.
For MtM those are its native per-session modules: ``in_stitcher`` (held-in units ->
shared projection), ``out_stitcher`` (latents -> log-rates of all units), and the
session token. Held-out spikes are targets only and never enter the encoder.
Inference uses MtM's "neuron" (co-smoothing) prompt token, as in BWB TS2.
"""
import numpy as np
import torch
from torch import nn
from omegaconf import OmegaConf

from figure1_data import ROOT, NT, DT
from pretrain.models.mtm.mtm import MtM

CKPT = ROOT / 'checkpoints' / 'mtm.pt'
NEW_SESSION_PREFIXES = ('in_stitcher.', 'out_stitcher.', 'session_emb.')


def load_checkpoint():
    return torch.load(CKPT, map_location='cpu', weights_only=False)


def model_config(ckpt):
    cfg = ckpt['cfg'].model
    cfg = OmegaConf.to_container(cfg, resolve=True) if not isinstance(cfg, dict) else dict(cfg)
    cfg.pop('_target_', None)
    return cfg


class MtMNeural(nn.Module):
    def __init__(self, data, ckpt=None, random_backbone=False):
        super().__init__()
        ckpt = ckpt or load_checkpoint()
        cfg = model_config(ckpt)
        cfg['finetune_enable'] = True
        self.backbone = MtM(**cfg)
        m = self.backbone
        eid = data['eid']
        self.eid = eid
        nin, nall = len(data['heldin']), len(data['unit_ids'])
        # Equivalent of link_datasets for a single new session.
        m.in_stitcher[eid] = nn.Linear(nin, m.shared_proj_input_dim)
        m.out_stitcher[eid] = nn.Linear(m.hidden_dim, nall)
        m.num_bins = NT
        from core.nn import Embedding
        m.position_emb = Embedding(NT, m.hidden_dim)
        m.session_ids = [eid]
        original = ckpt['model_state_dict']
        vocab = original.get('session_emb.vocab', {})
        assert eid not in vocab, 'evaluation session found in pretraining checkpoint'
        shared = {k: v for k, v in original.items() if not k.startswith(NEW_SESSION_PREFIXES)}
        if random_backbone:
            # Control: same architecture and protocol, pretrained weights NOT loaded.
            if getattr(m, 'tfixup_scale_base', None) is not None:
                m.custom_init()
            shared = {}
        else:
            # Load every shared submodule strictly; new-session modules stay freshly initialized.
            children = {name for name, _ in m.named_children()} - {'in_stitcher', 'out_stitcher', 'session_emb'}
            loaded = set()
            for name in sorted(children):
                sub = {k[len(name) + 1:]: v for k, v in shared.items() if k.startswith(name + '.')}
                if not sub and not any(True for _ in getattr(m, name).state_dict()):
                    continue
                getattr(m, name).load_state_dict(sub, strict=True)
                loaded.update(name + '.' + k for k in sub)
            leftover = set(shared) - loaded
            if leftover:
                raise RuntimeError(f'Unloaded pretrained keys: {sorted(leftover)}')
        m.session_emb.initialize_vocab([eid])
        for name, p in m.named_parameters():
            p.requires_grad_(name.startswith(NEW_SESSION_PREFIXES))
        # Start from the per-unit training mean rate, as the POYO+ head did.
        rate = data['splits']['train']['counts'].mean((0, 1))
        with torch.no_grad():
            m.out_stitcher[eid].bias.copy_(torch.from_numpy(np.log(np.maximum(rate, 1e-3 * DT))))
        self.session_token = int(np.asarray(m.session_emb.tokenizer(eid)).reshape(-1)[0])
        self.feature_dim = m.hidden_dim
        self.load_report = {
            'checkpoint': str(CKPT), 'model_config': cfg,
            'pretrained_parameters_loaded': int(sum(v.numel() for v in shared.values() if torch.is_tensor(v))),
            'new_session_modules': list(NEW_SESSION_PREFIXES), 'backbone_frozen': True,
            'random_backbone': random_backbone,
            'trainable_parameters': int(sum(p.numel() for p in m.parameters() if p.requires_grad)),
            'prompt': 'neuron', 'latent_shape': [NT, m.hidden_dim], 'behavior_supervision': False,
        }

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()  # keep pretrained dropout disabled, as for POYO+
        return self

    def forward(self, b):
        m = self.backbone
        x = b['x']
        B = x.shape[0]
        positions = torch.arange(NT, device=x.device).unsqueeze(0).expand(B, -1)
        tokens = torch.full((B,), self.session_token, device=x.device, dtype=torch.long)
        z = m.encode(x, positions, tokens, self.eid, 'neuron')
        log_rates = m.out_stitcher[self.eid](z)
        rates = log_rates.float().clamp(-12, 8).exp() / DT  # Hz, matching the LFADS/POYO+ interface
        return rates, z


def forward(model, method, b):
    rates, z = model(b)
    return rates, z, None


def neural_loss(model, method, b, rates, output):
    means = (rates * DT).clamp_min(1e-8)
    return (means - b['counts'] * means.log()).mean()
