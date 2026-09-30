"""30-second throughput probe. Run BEFORE committing to a long job."""
import sys, time, torch, simplex_fm as S
dev = sys.argv[1] if len(sys.argv) > 1 else "cuda"
h, l, B, L = 128, 10, 256, 500
net = S.SimplexFM(h, l).to(dev)
opt = torch.optim.Adam(net.parameters(), lr=2e-3)
d1 = torch.distributions.Dirichlet(torch.ones(4, device=dev))
Xa, _, _ = S.load_npz("dfb500.npz", crop=L)
Xa = Xa.to(dev)
print(f"device={dev} params={sum(p.numel() for p in net.parameters())/1e6:.2f}M "
      f"train={Xa.shape[0]}")
def run(n):
    t0 = time.time()
    for _ in range(n):
        i = torch.randint(0, Xa.shape[0], (B,), device=dev)
        x1 = Xa[i]; x0 = d1.sample((B, L)); t = torch.rand(B, device=dev)
        xt = t[:, None, None] * x1 + (1 - t[:, None, None]) * x0
        loss = ((net(xt, t) - (x1 - x0)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    if dev == "cuda": torch.cuda.synchronize()
    return n * B / (time.time() - t0)
run(3)                                   # warmup
v = run(30)
print(f"  {v:,.0f} sample-views/sec")
print(f"  80,000 steps x 256 = 20.5M views  ->  {20.5e6/v/3600:.2f} h")
print(f"  published Linear FM budget 40.3M  ->  {40.3e6/v/3600:.2f} h")
