import numpy as np
exec(open('loop_toy.py').read().split('s=1.0; y=0.3')[0])
s=1.0; y=0.3; B=512
def sweep(gainmul):
    import types
    out=[]
    for tm in [0.5,1.5]:
        tau=tm*s
        rng=np.random.default_rng(1); F0=rng.standard_normal(B); a=gainmul*np.exp(0.3*rng.standard_normal(B))
        Fc,es=run(F0,y,s,tau,4,rng=np.random.default_rng(11),a=a)
        for shift in [1.0,0.7,1.4]:
            rng2=np.random.default_rng(2); F1=shift*rng2.standard_normal(B); a2=gainmul*np.exp(0.3*rng2.standard_normal(B))
            Fu,_=run(F1,y,s,tau,0,rng=np.random.default_rng(22),a=a2,w=0.0)
            Fcl,_=run(F1,y,s,tau,4,rng=np.random.default_rng(22),a=a2)
            Fr,_=run(F1,y,s,tau,4,rng=np.random.default_rng(22),a=a2,e_sched=es)
            Fp,_=run(F1,y,s,tau,0,rng=np.random.default_rng(22),a=a2)
            su=Fu.std(ddof=1)
            print(f"  gain x{gainmul} tau_mult={tm} shift x{shift}: sd/ung closed {Fcl.std(ddof=1)/su:.3f} replay {Fr.std(ddof=1)/su:.3f} plug {Fp.std(ddof=1)/su:.3f} | |replay-closed|/closed {abs(Fr.std()-Fcl.std())/Fcl.std():.3f}")
for gm in [0.02,0.05,0.2]:
    sweep(gm)
