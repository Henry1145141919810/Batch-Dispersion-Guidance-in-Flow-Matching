import numpy as np
rng=np.random.default_rng(0)
# Linearised one-step model: dF_i = h/s^2 * a_i * num_i, a_i=|J_i^T g_i|^2 >0
def step(F,a,y,eta,tau2,h=1e-3,s2=1.0,disp_only=False):
    Fb=F.mean(); V=F.var(ddof=1); e=V/tau2-1; d=F-Fb
    num=(-eta*e*d) if disp_only else ((y-F)-eta*e*d)
    F2=F+h/s2*a*num
    return F2.var(ddof=1)-V, 1+eta*e
B=512
# heterogeneous gains positively correlated with deviation, target far above mean (q90-like: |b|/sigma ~1.4)
F=rng.standard_normal(B); d=F-F.mean()
a=np.exp(0.8*d/ d.std() + 0.3*rng.standard_normal(B))  # corr(a,d)>0
print("corr(a,d)=%.2f CV(a)=%.2f"%(np.corrcoef(a,d)[0,1],a.std()/a.mean()))
for y in [1.4,-1.4,0.0]:
  for eta in [0.0,0.5,1.0]:
    V=F.var(ddof=1)
    for tau2 in [V*0.5, V*1.0, V*2.0]:
        dV,w=step(F,a,y,eta,tau2)
        dVd,_=step(F,a,y,eta,tau2,disp_only=True)
        print(f"y={y:+.1f} eta={eta} V/tau2={V/tau2:.2f} w_eff={w:+.3f} total dV={dV:+.3e} disp-only dV={dVd:+.3e}")
