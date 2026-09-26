"""Measure v_f(t) = Var(f(x1) | x_t) directly on held-out data.

gc_soft and cpg_soft are read off the posterior mean m. For AFFINE f,
f(m) = f(E[x1|x_t]) = E[f(x1)|x_t] exactly, so E[(f(x1) - f(m))^2] IS the
conditional variance the TMPD/PiGDM denominator asks for -- no Tweedie
assumption, no probe, no estimator in between.

We measure it rather than assume Sigma = k J with k = (1-t)^2/t, because that
relation is derived for a Gaussian VP diffusion with unit per-coordinate noise.
Our source is Dirichlet(1) (per-coordinate variance 3/80), and the transplanted
k overstates the conditional variance by 30x at t=0.1 and 2.0e4x at t=0.7.
Using it would inflate tmpd's denominator by ~33% at t=0.5 on a quantity whose
true value is 0.003% of s^2 -- a step-size change masquerading as a method.
"""
import sys, json, torch
R='/vast/projects/pranam/lab/boboli/cis6270-project1-group2'
sys.path.insert(0, R+'/proj1/m2')
import simplex_fm as S
torch.set_num_threads(28)
ck=torch.load(R+'/proj1/m2/blade_bundle/fm_m2_dfb500.pt',map_location='cpu',weights_only=False)
net=S.SimplexFM(ck['hidden'],ck['layers']); net.load_state_dict(ck['state_dict']); net.eval()
L=ck['crop']; _,Xv,_=S.load_dfb(crop=L)
GRID=[0.0,0.02,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99]
B, REP = 256, 4
out={}
for pname,(soft,hard) in S.PROPS.items():
    vals=[]
    for t in GRID:
        acc=0.0
        for rep in range(REP):
            g=torch.Generator().manual_seed(1000*rep+int(t*1000))
            idx=torch.randperm(Xv.shape[0],generator=g)[:B]
            x1=Xv[idx]
            e=torch.empty(B,L,4).exponential_(generator=g); x0=e/e.sum(-1,keepdim=True)
            xt=t*x1+(1-t)*x0
            with torch.no_grad():
                m=xt+(1.0-t)*net(xt,torch.full((B,),float(t)))
            acc+=float(((hard(x1)-soft(m))**2).mean())
        vals.append(acc/REP)
    out[pname]=vals
    print(f"{pname}: " + " ".join(f"{t:.2f}:{v:.3e}" for t,v in zip(GRID,vals)))
json.dump({"grid":GRID,"v_f":out,"n":B*REP,"source":"DeepFlyBrain valid split"},
          open(R+'/proj1/m2/vf_table.json','w'), indent=1)
print("\nwrote vf_table.json")
