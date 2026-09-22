"""Fix the scalar-k bug and the RCH invariance bug. Run once.

k = (1-t)^2/t (flow) or sigma^2/alpha (VP) scales Sigma everywhere. It was
taken as a PYTHON FLOAT from t[0], so any batch whose t varies within it got
one sample's k applied to all of them. During sampling every sample shares t,
so this never fired there -- but `fit_rch.py` deliberately draws per-sample t,
and its targets were wrong for every row but the first.
"""
import io

p = "proj1/src/guidance.py"
s = io.open(p, encoding="utf-8").read()

# ---- Posterior.k becomes a per-sample tensor
s = s.replace('''    mean_coords: torch.Tensor
    mean_feats: torch.Tensor
    k: float''',
'''    mean_coords: torch.Tensor
    mean_feats: torch.Tensor
    k: torch.Tensor          # [B], per sample -- NOT a scalar (see fix note below)''')

s = s.replace('''    tv = float(t.reshape(-1)[0])
    k = (1.0 - tv) ** 2 / max(tv, 1e-6)
    return Posterior(m_c, m_f, k)''',
'''    tv = t.reshape(-1).clamp(min=1e-6)
    k = (1.0 - tv) ** 2 / tv                     # [B]
    return Posterior(m_c, m_f, k)''')

s = s.replace('''    av = float(alpha.reshape(-1)[0])
    sv = float(sigma.reshape(-1)[0])
    k = sv ** 2 / max(av, 1e-6)
    return Posterior(m_c, m_f, k)''',
'''    av = alpha.reshape(-1).clamp(min=1e-6)
    sv = sigma.reshape(-1)
    k = sv ** 2 / av                             # [B]
    return Posterior(m_c, m_f, k)''')

# ---- every consumer of k must broadcast it per sample
s = s.replace('''    _, (jv_c, jv_f) = torch.func.jvp(mfun, (coords, feats), (vec_c, vec_f))
    cost.gen_jvp += 1
    return k * jv_c.detach(), k * jv_f.detach()''',
'''    _, (jv_c, jv_f) = torch.func.jvp(mfun, (coords, feats), (vec_c, vec_f))
    cost.gen_jvp += 1
    kb = torch.as_tensor(k, device=jv_c.device, dtype=jv_c.dtype).reshape(-1, 1, 1)
    return kb * jv_c.detach(), kb * jv_f.detach()''')

s = s.replace('''    m3 = (k3_gg_c * g_c).sum(dim=(1, 2)) + (k3_gg_f * g_f).sum(dim=(1, 2))
    return m3 / v_f.clamp(min=1e-12).pow(1.5)''',
'''    m3 = (k3_gg_c * g_c).sum(dim=(1, 2)) + (k3_gg_f * g_f).sum(dim=(1, 2))
    return m3 / v_f.clamp(min=1e-12).pow(1.5)''')

s = s.replace('''    k3_gg_c = k * d_c.detach()
    k3_gg_f = k * (d_f.detach() if d_f is not None else torch.zeros_like(f_in))''',
'''    kb = torch.as_tensor(k, device=d_c.device, dtype=d_c.dtype).reshape(-1, 1, 1)
    k3_gg_c = kb * d_c.detach()
    k3_gg_f = kb * (d_f.detach() if d_f is not None else torch.zeros_like(f_in))''')

s = s.replace('''        sg_c, sg_f = torch.func.jvp(_mean, (c, f), (g_c, g_f))[1]
        return k * ((g_c * sg_c).sum(dim=(1, 2)) + (g_f * sg_f).sum(dim=(1, 2)))''',
'''        sg_c, sg_f = torch.func.jvp(_mean, (c, f), (g_c, g_f))[1]
        kb = torch.as_tensor(k, device=sg_c.device, dtype=sg_c.dtype).reshape(-1)
        return kb * ((g_c * sg_c).sum(dim=(1, 2)) + (g_f * sg_f).sum(dim=(1, 2)))''')

# ---- trace_scale: d overcounts by 3 (coords are zero-CoM, so Sigma has 3 null
# directions). Correct dimension is 3(N-1) + N*n_types.
s = s.replace('''    msk = mask.unsqueeze(-1)
    d = (mask.sum(dim=1) * (coords.shape[-1] + feats.shape[-1])).clamp(min=1.0)''',
'''    msk = mask.unsqueeze(-1)
    # Coordinates live on the zero-CoM subspace, so Sigma has exactly 3 null
    # directions and the coordinate block contributes 3(N-1), not 3N.
    n_at = mask.sum(dim=1)
    d = (3.0 * (n_at - 1.0).clamp(min=1.0) + n_at * feats.shape[-1]).clamp(min=1.0)''')

# ---- RCH features: the L1 coordinate term is NOT rotation invariant
s = s.replace('''                (feats ** 2 * m).sum((1, 2)) / n,
                (c.abs() * m).sum((1, 2)) / n]''',
'''                (feats ** 2 * m).sum((1, 2)) / n,
                # NOT the L1 norm of coordinates -- that is not rotation
                # invariant. Mean pairwise distance is.
                (d * (mask.unsqueeze(1) * mask.unsqueeze(2))).sum((1, 2))
                / (n * (n - 1).clamp(min=1.0))]''')

s = s.replace('''        d = torch.cdist(c, c) + (1 - mask).unsqueeze(1) * 1e3
        dmin = d.masked_fill(torch.eye(d.shape[-1], device=d.device,
                                       dtype=torch.bool), 1e3).min(-1).values''',
'''        d = torch.cdist(c, c)
        dpad = d + (1 - mask).unsqueeze(1) * 1e3
        dmin = dpad.masked_fill(torch.eye(d.shape[-1], device=d.device,
                                          dtype=torch.bool), 1e3).min(-1).values''')

s = s.replace('''        tv = t.view(-1)
        kk = torch.as_tensor(float(k), device=coords.device).expand_as(tv)''',
'''        tv = t.view(-1)
        kk = torch.as_tensor(k, device=coords.device, dtype=coords.dtype).reshape(-1)
        if kk.numel() == 1:
            kk = kk.expand_as(tv)''')

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("guidance.py: k is now per-sample; trace_scale dimension corrected; RCH features made rotation-invariant")
