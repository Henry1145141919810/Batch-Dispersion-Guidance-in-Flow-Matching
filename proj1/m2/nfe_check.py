"""Independent check of the blade NFE finding. Own sampler, not theirs."""
import sys, torch
sys.path.insert(0, '/vast/projects/pranam/lab/boboli/cis6270-project1-group2/proj1/m2')
import simplex_fm as S
torch.set_num_threads(28)

ck = torch.load('/vast/projects/pranam/lab/boboli/cis6270-project1-group2/proj1/m2/'
                'blade_bundle/fm_m2_dfb500.pt', map_location='cpu', weights_only=False)
net = S.SimplexFM(ck['hidden'], ck['layers']); net.load_state_dict(ck['state_dict']); net.eval()
L, n = ck['crop'], 512

@torch.no_grad()
def sample(nfe, seed):
    g = torch.Generator().manual_seed(seed)
    _e = torch.empty(n, L, 4).exponential_(generator=g)
    x = _e / _e.sum(-1, keepdim=True)   # Dirichlet(1), actually seeded
    dt = 1.0 / nfe
    for i in range(nfe):
        t = torch.full((n,), i * dt)
        x = S.to_simplex(x + dt * net(x, t))
    return x

print(f"checkpoint: epochs={ck['epochs']} val={ck['val_loss']:.5f} gc_std(data)={ck['gc_std']:.4f}")
print(f"n={n}, L={L}\n")
print(f"{'NFE':>5} {'gc_hard sd':>11} {'%of .0552':>10} {'gc_soft sd':>11} "
      f"{'soft/hard':>10} {'GC mean':>9} {'A/C/G/T':>28}")
for nfe in (100, 400):
    x = sample(nfe, 0)
    hard = S.gc_hard(x); soft = S.gc_soft(x)
    comp = x.argmax(-1).flatten().bincount(minlength=4).float(); comp /= comp.sum()
    print(f"{nfe:5d} {hard.std():11.4f} {hard.std()/0.0552*100:9.0f}% {soft.std():11.4f} "
          f"{soft.std()/hard.std():10.3f} {hard.mean():9.4f}   "
          + "/".join(f"{c:.4f}" for c in comp))
print(f"\n{'real':>5} {0.0552:11.4f} {100:9.0f}% {0.0552:11.4f} {1.000:10.3f} {0.4552:9.4f}"
      "   0.2725/0.2282/0.2274/0.2719")
