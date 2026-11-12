"""Fixed normal-training probes; no optimizer updates or test-set access."""
import math


def rank99(eigenvalues):
    total = eigenvalues.sum()
    if float(total) <= 1e-24:
        return 0
    return min(len(eigenvalues), int(((eigenvalues/total).cumsum(0)<.99).sum())+1)


def measure(model, dataset, patient_indices, entropy_weight, memory_decay):
    import numpy as np
    import torch
    import torch.nn.functional as F
    was_training = model.training
    model.eval()
    memory = model.mem_rep.memory
    channels, slots = memory.shape[1], memory.shape[0]
    moments = {name: {'sum': torch.zeros(channels, dtype=torch.float64, device='cuda'),
                      'outer': torch.zeros(channels, channels, dtype=torch.float64, device='cuda'), 'n': 0}
               for name in ('query', 'readout')}
    attention_sum = torch.zeros(slots, dtype=torch.float64, device='cuda')
    winners = torch.zeros(slots, dtype=torch.float64, device='cuda')
    h_sum = top1_sum = mse_sum = 0.
    tokens = patches = 0
    patient_rows = []
    with torch.no_grad():
        for patient, indices in patient_indices.items():
            local_mass = torch.zeros(slots, device='cuda')
            local_tokens = local_patches = 0
            local_mse = 0.
            for offset in range(0, len(indices), 2):
                x = dataset[indices[offset:offset+2]].cuda()
                z = model.encode(x)
                readout, attention = model.mem_rep(z)
                reconstruction = model.decode(readout)
                a = attention.reshape(-1, slots)
                entropy = -(a * a.clamp_min(1e-30).log()).sum(-1)
                errors = (reconstruction-x).square().flatten(1).mean(1)
                h_sum += entropy.sum().item()
                top1_sum += a.max(-1).values.sum().item()
                mse_sum += errors.sum().item()
                tokens += len(a)
                patches += len(x)
                attention_sum += a.sum(0).double()
                winners += torch.bincount(a.argmax(-1), minlength=slots).double()
                local_mass += a.sum(0)
                local_tokens += len(a)
                local_mse += errors.sum().item()
                local_patches += len(x)
                for name, value in [('query', z), ('readout', readout)]:
                    v = value.flatten(2).transpose(1, 2).reshape(-1, channels).double()
                    m = moments[name]
                    m['sum'] += v.sum(0)
                    m['outer'] += v.T @ v
                    m['n'] += len(v)
            p = local_mass/local_tokens
            patient_rows.append({'patient': patient, 'patches': local_patches,
                                 'mse': local_mse/local_patches,
                                 'mean_slot_mass': p.cpu().tolist(),
                                 'effective_global_slots': float((-(p*p.clamp_min(1e-30).log()).sum()).exp())})
        ranks = {}
        for name, m in moments.items():
            mean = m['sum']/m['n']
            cov = m['outer']/m['n']-mean[:, None]*mean[None, :]
            e = torch.linalg.eigvalsh(cov).flip(0).clamp_min(0)
            v = e/e.sum().clamp_min(1e-30)
            ranks[name] = {'rank99': rank99(e),
                           'top1_variance_fraction': float(v[0]), 'total_variance': float(e.sum())}
        m = memory.detach().cpu().double()
        distances = torch.cdist(m, m, compute_mode='donot_use_mm_for_euclid_dist')
        neighbourhood = (distances <= 1e-6).sum(1)
        center = int(neighbourhood.argmax())
        cluster = (distances[center] <= 1e-6).cuda()
        singular = torch.linalg.svdvals(m-m.mean(0))
        fractions = singular.square()/singular.square().sum().clamp_min(1e-30)
        p = attention_sum/tokens
        global_h = float(-(p*p.clamp_min(1e-30).log()).sum())
        norms = memory.detach().norm(dim=1)

    # Gradient signal of a fixed normal training patch, separately from decay.
    model.zero_grad(set_to_none=True)
    first = dataset[[next(iter(patient_indices.values()))[0]]].cuda()
    reconstruction, attention = model(first)
    mse = F.mse_loss(reconstruction, first)
    entropy = -(attention*torch.log(attention+1e-12)).sum(1).mean()
    gr = torch.autograd.grad(mse, memory, retain_graph=True)[0]
    ge = torch.autograd.grad(entropy_weight*entropy, memory)[0]
    gradients = {}
    for name, mask in [('largest_duplicate_neighbourhood', cluster), ('remaining_slots', ~cluster)]:
        if not mask.any():
            continue
        gradients[name] = {'slots': int(mask.sum()),
                           'reconstruction_row_norm_median': float(gr[mask].norm(dim=1).median()),
                           'weighted_entropy_row_norm_median': float(ge[mask].norm(dim=1).median()),
                           'weight_decay_row_norm_median': float((memory_decay*memory.detach()[mask]).norm(dim=1).median())}
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return {'probe_scope': 'Fixed first/middle/last patches of each NORMAL TRAINING patient only',
            'patches': patches, 'tokens': tokens, 'mse': mse_sum/patches,
            'mean_query_entropy': h_sum/tokens, 'effective_slots_exp_mean_query_entropy': math.exp(h_sum/tokens),
            'mean_top1_mass': top1_sum/tokens, 'global_effective_slots': math.exp(global_h),
            'global_slot_mass': p.cpu().tolist(), 'argmax_distinct_slots': int((winners>0).sum()),
            'most_common_argmax_fraction': float(winners.max()/tokens),
            'largest_duplicate_neighbourhood_l2_1e_minus6': int(neighbourhood.max()),
            'memory_rank99': rank99(singular.square()),
            'memory_norm_min_median_max': [float(norms.min()), float(norms.median()), float(norms.max())],
            'feature_covariance': ranks, 'first_training_patch_gradients': gradients,
            'patients': patient_rows}
