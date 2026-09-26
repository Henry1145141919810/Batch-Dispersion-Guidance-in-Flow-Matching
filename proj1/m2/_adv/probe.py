"""Adversarial probe: do the M2 arms actually separate? + non-affine candidate."""
import os, sys, math, time, json
import torch
HERE="/vast/projects/pranam/lab/boboli/cis6270-project1-group2/proj1/m2"
sys.path.insert(0,HERE)
import simplex_fm as S
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","28")))
ck=torch.load(os.path.join(HERE,"blade_bundle","fm_m2_dfb500.pt"),map_location="cpu",weights_only=False)
net=S.SimplexFM(ck["hidden"],ck["layers"]); net.load_state_dict(ck["state_dict"]); net.eval()
L=ck["crop"]; s=ck["gc_std"]; s2=s*s; y=ck["gc_mean"]; gn2=2.0/L
print("L=%d s=%.6f s2=%.6g y=%.5f |g|^2=%.6g  d=%d"%(L,s,s2,y,gn2,4*L))
def gvec(B):
    g=torch.zeros(B,L,4); g[:,:,1]=1.0/L; g[:,:,2]=1.0/L; return g

B=64
torch.manual_seed(20260921)
x0=torch.distributions.Dirichlet(torch.ones(4)).sample((B,L))

def mA(x,t):
    xt=x.detach().requires_grad_(True); tv=torch.full((B,),float(t))
    m=xt+(1.0-float(t))*net(xt,tv); f=S.gc_soft(m)
    A=torch.autograd.grad(f,xt,grad_outputs=torch.ones(B),retain_graph=True)[0]
    return m,f.detach(),A.detach(),xt,tv

print("\n=== (A) nu/s^2 for every arm across the guidance window t>=0.5 ===")
print("%5s %10s %10s %10s %10s %10s"%("t","vf/s2(tmpd)","lgdR2|g|2/s2","tfg.02/s2","tfg.35/s2","coef spread"))
rows=[]
for t in (0.5,0.6,0.7,0.8,0.9,0.95,0.99):
    m,F,A,xt,tv=mA(x0,t)
    k=(1-t)**2/max(t,1e-6)
    g=gvec(B)
    vf=k*(g*A).sum(dim=(1,2))                     # EXACT, free from the same VJP
    # lgd_mc trace probe: tr(Sigma)/d, 8 gaussian probes (exact JVP via autograd.functional)
    est=[]
    gg=torch.Generator().manual_seed(7)
    for _ in range(8):
        z=torch.randn(B,L,4,generator=gg)
        xz=x0.detach().requires_grad_(True)
        mz=xz+(1.0-float(t))*net(xz,tv)
        vj=torch.autograd.grad((mz*z).sum(),xz)[0]
        est.append((z*vj).sum(dim=(1,2)))
    E=torch.stack(est)                            # [8,B] estimates of tr(J)
    trJ=E.mean(0); r2=(k*trJ/(4*L)).clamp(min=0)
    nu_lgd=r2*gn2
    nu_t02=0.02**2*gn2; nu_t35=0.35**2*gn2
    # how much do the coefficients differ, as a ratio to plug?
    cs=[1.0, float((s2/(s2+vf)).mean()), float((s2/(s2+nu_lgd)).mean()),
        s2/(s2+nu_t02), s2/(s2+nu_t35)]
    print("%5.2f %10.4f %10.4f %10.6f %10.4f   plug1.0 tmpd%.4f lgd%.4f tfg02 %.6f tfg35 %.4f"%(
        t,float((vf/s2).mean()),float((nu_lgd/s2).mean()),nu_t02/s2,nu_t35/s2,cs[1],cs[2],cs[3],cs[4]))
    # single-probe Hutchinson spread (what m2_sweep actually uses)
    rel=float((E.std(0)/E.mean(0).abs()).mean())
    rows.append((t,float((vf/s2).mean()),float((nu_lgd/s2).mean()),rel,float((E.min(0).values<0).float().mean())))
print("\n  single-probe tr(J) Hutchinson relative sd (m2_sweep uses ONE probe):")
for t,a,b,rel,neg in rows:
    print("   t=%.2f  rel sd of 1 probe = %.4f   frac of probes negative = %.3f"%(t,rel,neg))

print("\n=== (B) FD-JVP (what m2_sweep does) vs EXACT free v_f ===")
for t in (0.5,0.7,0.9):
    m,F,A,xt,tv=mA(x0,t)
    k=(1-t)**2/max(t,1e-6); g=gvec(B)
    vf_exact=k*(g*A).sum(dim=(1,2))
    gm=torch.autograd.grad(F.sum() if F.requires_grad else S.gc_soft(m).sum(),None,allow_unused=True) if False else g
    with torch.no_grad():
        h=1e-3/gm.reshape(B,-1).norm(dim=1).clamp(min=1e-12)
        xp=(xt+h.view(-1,1,1)*gm).detach()
        mp=xp+(1.0-float(t))*net(xp,tv)
    Jg=(mp-m.detach())/h.view(-1,1,1)
    vf_fd=k*(gm*Jg).sum(dim=(1,2))
    r=(vf_fd/vf_exact)
    print("  t=%.2f  vf_exact/s2=%.5f  vf_fd/s2=%.5f  ratio mean %.5f sd %.5f  max|rel err| %.4f"%(
        t,float((vf_exact/s2).mean()),float((vf_fd/s2).mean()),float(r.mean()),float(r.std()),
        float((r-1).abs().max())))

print("\n=== (C) non-affine candidates on REAL data ===")
Xtr,Xva,Xte=S.load_dfb(crop=L)
def cpg(x):  # C then G, differentiable, exactly QUADRATIC
    return (x[...,:-1,1]*x[...,1:,2]).sum(-1)/(x.shape[-2]-1)
def gcwinvar(x,W=50):  # variance of GC across non-overlapping windows: QUADRATIC
    gcl=x[...,1:3].sum(-1)            # [.., L]
    nw=x.shape[-2]//W
    wm=gcl[...,:nw*W].reshape(*gcl.shape[:-1],nw,W).mean(-1)
    return wm.var(-1,unbiased=True)
def gcskew(x):  # AFFINE -- the negative control
    h=x.shape[-2]//2
    return x[...,:h,1:3].sum(-1).mean(-1)-x[...,h:,1:3].sum(-1).mean(-1)
for nm,fn in (("gc_soft",S.gc_soft),("cpg_frac",cpg),("gc_window_var(W=50)",gcwinvar),("gc_skew(affine)",gcskew)):
    v=fn(Xte.float())
    vt=fn(Xtr.float())
    print("  %-22s test mean %.6f sd %.6f | train mean %.6f sd %.6f | q50 %.6f q90 %.6f  corr(gc) %+.3f"%(
        nm,float(v.mean()),float(v.std()),float(vt.mean()),float(vt.std()),
        float(v.median()),float(v.quantile(0.9)),
        float(torch.corrcoef(torch.stack([v,S.gc_soft(Xte.float())]))[0,1])))

print("\n=== (D) does cpg activate the curvature terms? tr(H Sigma) and grad state-dependence ===")
t=0.7
xt=x0.detach().requires_grad_(True); tv=torch.full((B,),t)
m=xt+(1-t)*net(xt,tv)
fc=cpg(m)
gc_=torch.autograd.grad(fc.sum(),m,retain_graph=True,create_graph=True)[0]
print("  cpg grad state-dependence: sd across batch / mean |grad| = %.4f  (gc_soft would be 0)"%(
   float(gc_.detach().std(0).mean()/gc_.detach().abs().mean())))
# tr(H Sigma) = k tr(H J); Hutchinson with 16 Rademacher probes
k=(1-t)**2/t
gh=torch.Generator().manual_seed(3); acc=[]
for _ in range(16):
    z=(torch.randint(0,2,(B,L,4),generator=gh).float()*2-1)
    Hz=torch.autograd.grad((gc_*z).sum(),m,retain_graph=True)[0]   # H z  (at m)
    # need z^T H J z : J z via VJP of m wrt xt with grad_outputs -> that's J^T; use fwd diff
    with torch.no_grad():
        hh=1e-3/z.reshape(B,-1).norm(dim=1)
        mq=(xt+hh.view(-1,1,1)*z).detach(); mq=mq+(1-t)*net(mq,tv)
    Jz=(mq-m.detach())/hh.view(-1,1,1)
    acc.append((Hz.detach()*Jz).sum(dim=(1,2)))
trHS=k*torch.stack(acc).mean(0)
cpg_sd=float(cpg(Xte.float()).std())
print("  c = 0.5*tr(H Sigma) = %.6g  vs cpg sd = %.6g  ->  c/sd = %.4f  (gc_soft: exactly 0)"%(
   float(0.5*trHS.mean()),cpg_sd,float(0.5*trHS.mean())/cpg_sd))
gcp=torch.autograd.grad(fc.sum(),xt,retain_graph=True)[0]
vfc=k*(gc_.detach()*torch.zeros_like(gc_)).sum() # placeholder
print("  |grad_m cpg| mean %.4g   (gc_soft |g| = %.4g)"%(float(gc_.detach().reshape(B,-1).norm(dim=1).mean()),math.sqrt(gn2)))
