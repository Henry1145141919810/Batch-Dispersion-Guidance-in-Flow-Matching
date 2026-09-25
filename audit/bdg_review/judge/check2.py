import numpy as np
exec(open('check.py').read().split('B=512')[0])
rng=np.random.default_rng(0); B=512
F=rng.standard_normal(B); d=F-F.mean()
a=np.exp(0.8*d/d.std()+0.3*rng.standard_normal(B))
V=F.var(ddof=1)
for y in [2.0,3.0]:
    dV,w=step(F,a,y,0.0,V); print(f"plug (eta=0) y={y} total dV={dV:+.3e}")
# closed form check: dV = 2h/((B-1)s2) [ (y-Fb) sum d a - w_eff sum d^2 a ] (first order)
h=1e-6; y=1.4; eta=1.0; tau2=2*V
e=V/tau2-1; w=1+eta*e
pred=2*h/(B-1)*((y-F.mean())*(d*a).sum()-w*(d*d*a).sum())
dV,_=step(F,a,y,eta,tau2,h=h)
print("closed form total dV",pred,"measured",dV,"rel",abs(pred-dV)/abs(pred))
