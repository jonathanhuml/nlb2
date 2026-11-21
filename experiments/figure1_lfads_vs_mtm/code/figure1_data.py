"""One split, one neuron mask, and identical windows for the two Figure 1 models."""
from pathlib import Path
import hashlib
import json
import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DT = 0.02
NT = 50
PILOT = 'f56194bc-8215-4ae8-bc6a-89781ad8e050'


def text_array(ds):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in ds[:]])


def holdout_mask(ids, regions):
    """Stable region-stratified holdout independent of values and training seed."""
    mask = np.zeros(len(ids), dtype=bool)
    for region in sorted(set(regions)):
        indices = np.flatnonzero(regions == region)
        order = sorted(indices, key=lambda i: hashlib.sha256(('figure1-mask-v1/'+ids[i]).encode()).digest())
        k = min(len(order)-1, max(1, int(round(.2*len(order))))) if len(order)>1 else 0
        mask[order[:k]] = True
    if mask.sum() < 2:
        raise ValueError('Insufficient held-out units')
    return mask


def prepare(eid):
    import fcntl
    (ROOT/'prepared').mkdir(parents=True, exist_ok=True)
    with open(ROOT/'prepared'/f'{eid}.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _prepare(eid)


def _prepare(eid):
    dest = ROOT/'prepared'/f'{eid}.pt'
    if dest.exists():
        return torch.load(dest, weights_only=False)
    with h5py.File(ROOT/'data'/f'{eid}.h5','r') as f:
        if f['brainset'].attrs['unit_filtering'] != 'selected_units':
            raise ValueError('Expected quality-filtered data')
        u=f['units']
        ids=text_array(u['id']); regions=text_array(u['region_beryl'])
        mask=holdout_mask(ids,regions)
        heldin=np.flatnonzero(~mask); heldout=np.flatnonzero(mask)
        unit_map=np.full(len(ids),-1,dtype=np.int64); unit_map[heldin]=np.arange(len(heldin))
        times=f['spikes/timestamps'][:]; units=f['spikes/unit_index'][:].astype(np.int64)
        if np.any(np.diff(times)<0):
            raise ValueError('Spikes must be sorted')
        w=f['wheel']; wheel=w['wheel_speed'][:]
        wt=float(w['domain/start'][0])+np.arange(len(wheel))/float(w.attrs['sampling_rate'])
        windows=f['task_aligned_intervals/movement_window']
        starts=windows['start'][:]; ends=windows['end'][:]; trials=windows['trial_index'][:]
        splits={}; dropped={}
        for split in ['train','val','test']:
            lo=f[f'ts1_{split}_domain/start'][:]; hi=f[f'ts1_{split}_domain/end'][:]
            eligible=np.flatnonzero(np.any((starts[:,None]>=lo)&(ends[:,None]<=hi),axis=1))
            counts=[]; behavior=[]; st=[]; ui=[]; kept=[]; last_end=-np.inf
            for i in eligible:
                a=float(starts[i]); b=a+1.
                if a < last_end or b>float(ends[i])+1e-6:
                    continue
                center=a+(np.arange(NT)+.5)*DT
                if center[0]<wt[0] or center[-1]>wt[-1]:
                    continue
                y=np.interp(center,wt,wheel)
                if not np.isfinite(y).all():
                    continue
                left,right=np.searchsorted(times,[a,b],side='left')
                tt=times[left:right]-a; uu=units[left:right]
                bins=np.floor(tt/DT).astype(int)
                cc=np.zeros((NT,len(ids)),dtype=np.float32)
                np.add.at(cc,(bins,uu),1)
                keep=unit_map[uu]>=0
                counts.append(cc); behavior.append(y.astype(np.float32))
                st.append(tt[keep].astype(np.float32)); ui.append(unit_map[uu[keep]])
                kept.append(int(i)); last_end=b
            if len(kept)<20:
                raise ValueError(f'{eid} {split}: only {len(kept)} valid windows')
            splits[split]={'counts':np.stack(counts),'behavior':np.stack(behavior),
                           'spike_times':st,'spike_units':ui,'starts':starts[kept],
                           'trial_ids':trials[kept], 'domain_start':lo,'domain_end':hi}
            dropped[split]=int(len(eligible)-len(kept))
        data={'eid':eid,'subject':str(f['subject'].attrs['id']),'unit_ids':ids,
              'regions':regions,'cosmos':text_array(u['region_cosmos']),
              'xyz':u['location'][:],'heldin':heldin,'heldout':heldout,
              'splits':splits,'dropped_windows':dropped,'dt':DT}
    audit_data(data)
    dest.parent.mkdir(parents=True,exist_ok=True)
    torch.save(data,dest)
    summary={k:data[k] for k in ['eid','subject','dropped_windows']}
    summary.update(units=len(ids),heldin=len(heldin),heldout=len(heldout),
                   windows={k:len(v['counts']) for k,v in splits.items()},
                   mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
                   sha256=hashlib.file_digest((ROOT/'data'/f'{eid}.h5').open('rb'),'sha256').hexdigest())
    dest.with_suffix('.json').write_text(json.dumps(summary,indent=2))
    print('PREPARED',json.dumps(summary),flush=True)
    return data


def audit_data(data):
    seen=set()
    for split,v in data['splits'].items():
        assert np.isfinite(v['counts']).all() and np.isfinite(v['behavior']).all()
        assert (v['counts']>=0).all()
        assert set(v['trial_ids']).isdisjoint(seen)
        seen.update(v['trial_ids'])
        assert np.all(v['starts'][1:] >= v['starts'][:-1]+1.-1e-8)
        for cc,tt,uu in zip(v['counts'],v['spike_times'],v['spike_units']):
            reconstructed=np.zeros_like(cc[:,data['heldin']])
            np.add.at(reconstructed,(np.floor(tt.astype(np.float64)/DT).clip(0,49).astype(int),uu),1)
            # Float32 timestamps can move spikes very close to a bin boundary.
            assert int(reconstructed.sum())==int(cc[:,data['heldin']].sum())
            assert len(uu)==0 or (uu.min()>=0 and uu.max()<len(data['heldin']))
    assert data['splits']['train']['starts'].max()+1 < data['splits']['val']['starts'].min()
    assert data['splits']['val']['starts'].max()+1 < data['splits']['test']['starts'].min()


def batch(data,split,indices,device):
    indices=np.asarray(indices)
    v=data['splits'][split]
    counts=torch.from_numpy(v['counts'][indices]).to(device)
    return {'counts':counts,'x':counts[:,:,data['heldin']],
            'spike_times':[v['spike_times'][i] for i in indices],
            'spike_units':[v['spike_units'][i] for i in indices]}
