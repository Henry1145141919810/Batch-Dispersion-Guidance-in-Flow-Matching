import os,sys,math,time
import torch
HERE="$PROJECT_ROOT/proj1/m2"
sys.path.insert(0,HERE); import simplex_fm as S
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","28")))
ck=torch.load(os.path.join(HERE,"blade_bundle","fm_m2_dfb500.pt"),map_location="cpu",weights_only=False)
net=S.SimplexFM(ck["hidden"],ck["layers"]); net.load_state_dict(ck["state_dict"]); net.eval()
L=ck["crop"]; s=ck["gc_std"]; s2=s*s
def cpg(x): return (x[...,:-1,1]*x[...,1:,2]).sum(-1)/(x.shape[-2]-1)
def gcwv(x,W=50):
    gl=x[...,1:3].sum(-1); nw=x.shape[-2]//W
    return gl[...,:nw*W].reshape(*gl.shape[:-1],nw,W).mean(-1).var(-1,unbiased=True)
PROPS={"gc_soft":S.gc_soft,"cpg_frac":cpg,"gc_window_var":gcwv}
B=48; torch.manual_seed(20260921)
x0=torch.distributions.Dirichlet(torch.ones(4)).sample((B,L))
Xte=S.load_dfb(crop=L)[2].float()
print("=== curvature budget: is c = 0.5 tr(H Sigma) big enough to separate smg_mean from plug? ===")
print("%-15s %6s %10s %10s %10s %10s %10s"%("prop","t","sd(data)","c","c/sd","v_f/s2","|g| spread"))
for nm,fn in PROPS.items():
    sd=float(fn(Xte).std())
    for t in (0.5,0.7,0.9):
        xt=x0.detach().requires_grad_(True); tv=torch.full((B,),t)
        m=xt+(1-t)*net(xt,tv); f=fn(m)
        gm=torch.autograd.grad(f.sum(),m,retain_graph=True,create_graph=True)[0]
        k=(1-t)**2/t
        # v_f = k g^T J g  via exact JVP on the g direction (forward diff, tight h)
        gd=gm.detach()
        with torch.no_grad():
            h=1e-4/gd.reshape(B,-1).norm(dim=1).clamp(min=1e-20)
            mp=(xt+h.view(-1,1,1)*gd).detach(); mp=mp+(1-t)*net(mp,tv)
        vf=k*(gd*((mp-m.detach())/h.view(-1,1,1))).sum(dim=(1,2))
        # tr(H Sigma) = k tr(H J), Hutchinson 24 Rademacher probes
        gh=torch.Generator().manual_seed(11); acc=[]
        for _ in range(24):
            z=(torch.randint(0,2,(B,L,4),generator=gh).float()*2-1)
            if gm.grad_fn is None:
                Hz=torch.zeros_like(z)          # exactly affine: H = 0
            else:
                Hz=torch.autograd.grad((gm*z).sum(),m,retain_graph=True,
                                       allow_unused=True,materialize_grads=True)[0].detach()
            with torch.no_grad():
                hh=1e-3/z.reshape(B,-1).norm(dim=1)
                mq=(xt+hh.view(-1,1,1)*z).detach(); mq=mq+(1-t)*net(mq,tv)
            acc.append((Hz*((mq-m.detach())/hh.view(-1,1,1))).sum(dim=(1,2)))
        c=0.5*k*torch.stack(acc).mean(0)
        gsp=float(gd.reshape(B,-1).norm(dim=1).std()/gd.reshape(B,-1).norm(dim=1).mean())
        print("%-15s %6.2f %10.6f %+10.3e %+10.4f %10.4f %10.4f"%(
          nm,t,sd,float(c.mean()),float(c.mean())/sd,float((vf/ (sd*sd)).mean()),gsp))

print("\n=== timing: 500bp NFE cost per cell, at this thread count (%d) ==="%torch.get_num_threads())
n=512
for arm,extra_fwd,extra_bwd in (("unguided",0,0),("plug",1,1),("tmpd_fd",2,1),("lgd_mc(K=8,1probe)",2,1)):
    pass
xs=torch.distributions.Dirichlet(torch.ones(4)).sample((n,L))
tv=torch.full((n,),0.7)
t0=time.time()
with torch.no_grad(): net(xs,tv)
fwd=time.time()-t0
xr=xs.detach().requires_grad_(True)
t0=time.time()
m=xr+0.3*net(xr,tv); f=S.gc_soft(m)
A=torch.autograd.grad(f,xr,grad_outputs=torch.ones(n))[0]
fb=time.time()-t0
print("n=%d L=%d : one fwd %.2fs ; one fwd+vjp %.2fs"%(n,L,fwd,fb))
for steps in (100,400):
    ng=steps//2
    for nm,nf,nfb in (("unguided",steps,0),("plug/bdg",steps,ng),("tmpd (FD, as written)",steps+ng,ng),
                      ("tmpd (free v_f)",steps,ng),("lgd_mc 1probe",steps+ng,ng),("lgd_mc 4probe",steps+4*ng,ng)):
        secs=nf*fwd+nfb*fb
        print("  NFE %4d %-24s ~%6.1f min"%(steps,nm,secs/60))
