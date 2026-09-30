"""Is M1's Tweedie scale k=(1-t)^2/t right for the simplex flow?

Because gc_soft is affine, f(m) = f(E[x1|x_t]) = E[f(x1)|x_t] exactly, so the
residual E[(f(x1) - f(m))^2] IS the conditional variance the denominator wants.
Compare it against v_f = k * g^T J g, which m2_sweep currently uses."""
import sys, torch
R='$PROJECT_ROOT'
sys.path.insert(0, R+'/proj1/m2')
import simplex_fm as S
torch.set_num_threads(28)
ck=torch.load(R+'/proj1/m2/blade_bundle/fm_m2_dfb500.pt',map_location='cpu',weights_only=False)
net=S.SimplexFM(ck['hidden'],ck['layers']); net.load_state_dict(ck['state_dict']); net.eval()
L=ck['crop']; _,Xv,_=S.load_dfb(crop=L)
B=256; g=torch.Generator().manual_seed(0)
print(f"{'t':>6} {'measured Var(f|x_t)':>20} {'v_f = k g^T J g':>17} {'ratio':>12} {'s^2':>10}")
s2=ck['gc_std']**2
for t in (0.1,0.3,0.5,0.7,0.9,0.99):
    idx=torch.randperm(Xv.shape[0], generator=g)[:B]
    x1=Xv[idx]
    e=torch.empty(B,L,4).exponential_(generator=g); x0=e/e.sum(-1,keepdim=True)
    xt=(t*x1+(1-t)*x0).detach().requires_grad_(True)
    tv=torch.full((B,),float(t))
    m=xt+(1.0-t)*net(xt,tv)
    fval=S.gc_soft(m)
    # measured: residual of the (exact, since affine) posterior-mean prediction
    meas=float(((S.gc_hard(x1)-fval.detach())**2).mean())
    # v_f the code computes: one VJP gives A = J^T g, then v_f = k<g,A>
    A=torch.autograd.grad(fval.sum(), xt)[0]
    gv=torch.zeros(L,4); gv[:,1]=1.0/L; gv[:,2]=1.0/L
    k=(1.0-t)**2/max(t,1e-6)
    vf=float((k*(gv.unsqueeze(0)*A).sum(dim=(1,2))).mean())
    print(f"{t:6.2f} {meas:20.3e} {vf:17.3e} {vf/max(meas,1e-30):12.1f}x {s2:10.3e}")
print(f"\ns^2 = {s2:.3e}   (the likelihood width; v_f only matters if comparable to it)")
