# Judge-3 toy: scalar-property batch loop with (i) heterogeneous per-sample gains a_i,
# (ii) a natural flow drift that contracts then re-widens the across-batch spread,
# (iii) per-step stochastic noise.  Tests: fixed point law, schedule replay on fresh
# noise, and robustness of closed loop vs replayed schedule when the START spread shifts.
import numpy as np
def run(F0, y, s, tau, eta, steps=50, h=0.02, w=1.0, rng=None, e_sched=None, noise=0.02, a=None, drift=True):
    F=F0.copy(); B=len(F); es=[]
    if a is None: a=np.ones(B)
    for n in range(steps):
        t=0.5+0.5*n/steps
        Fb=F.mean(); V=F.var(ddof=1)
        e=(V-tau**2)/tau**2 if e_sched is None else e_sched[n]
        es.append(e)
        num=(y-F)-eta*e*(F-Fb)
        mult=w*(1-t)/t*8.0
        # natural drift of across spread: contract early, re-widen late (non-monotone)
        d=(-0.6 if t<0.75 else +0.5)*(F-Fb)*h if drift else 0
        F=F+h*mult*a*num/s**2*s**2*0.5 + d + noise*rng.standard_normal(B)*np.sqrt(h)
    return F,np.array(es)
s=1.0; y=0.3; B=512
print("fixed-point law, no drift/noise, long horizon (homogeneous):")
for eta in [1.5,2,4,8]:
    rng=np.random.default_rng(0); F0=rng.standard_normal(B)
    F=F0.copy(); tau=1.2
    for _ in range(20000):
        V=F.var(ddof=1); e=(V-tau**2)/tau**2; Fb=F.mean()
        F=F+0.01*((y-F)-eta*e*(F-Fb))
    print(f"  eta={eta}: V/tau^2={F.var(ddof=1)/tau**2:.5f} predicted {1-1/eta:.5f}")
print("\nclosed loop vs e-schedule replayed on FRESH noise vs plug, and under START-spread shift")
for tm in [0.5,1.5]:
    tau=tm*s
    rng=np.random.default_rng(1); F0=rng.standard_normal(B); a=np.exp(0.3*rng.standard_normal(B))
    Fc,es=run(F0,y,s,tau,4,rng=np.random.default_rng(11),a=a)
    for shift in [1.0,0.7,1.4]:
        rng2=np.random.default_rng(2); F1=shift*rng2.standard_normal(B); a2=np.exp(0.3*rng2.standard_normal(B))
        Fu,_=run(F1,y,s,tau,0,rng=np.random.default_rng(22),a=a2,w=0.0)
        Fcl,_=run(F1,y,s,tau,4,rng=np.random.default_rng(22),a=a2)
        Fr,_=run(F1,y,s,tau,4,rng=np.random.default_rng(22),a=a2,e_sched=es)
        Fp,_=run(F1,y,s,tau,0,rng=np.random.default_rng(22),a=a2)
        su=Fu.std(ddof=1)
        print(f"  tau_mult={tm} start-shift x{shift}: sd/unguided closed {Fcl.std(ddof=1)/su:.3f} replay {Fr.std(ddof=1)/su:.3f} plug {Fp.std(ddof=1)/su:.3f} | sd/tau closed {Fcl.std(ddof=1)/tau:.3f} replay {Fr.std(ddof=1)/tau:.3f}")
