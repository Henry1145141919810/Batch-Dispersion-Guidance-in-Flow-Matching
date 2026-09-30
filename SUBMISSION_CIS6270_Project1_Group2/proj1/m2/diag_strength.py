"""Why is guidance weak? Check the three throttles: clip, t_min, and the
(1-t)/t score->velocity factor."""
import sys, torch
R='$PROJECT_ROOT'
sys.path.insert(0, R+'/proj1/m2')
import simplex_fm as S, m2_sweep as M
torch.set_num_threads(28)
ck=torch.load(R+'/proj1/m2/blade_bundle/fm_m2_dfb500.pt',map_location='cpu',weights_only=False)
net=S.SimplexFM(ck['hidden'],ck['layers']); net.load_state_dict(ck['state_dict']); net.eval()
s=ck['gc_std']; L=ck['crop']
Xa,_,_=S.load_dfb(crop=L); gcd=S.gc_hard(Xa)
y90=float(gcd.quantile(0.90)); N,NFE=128,100
rk=M.kmer_freq(Xa[:4096]); delta=0.16*s
base,_=M.run_cell(net,ck,'unguided','-',0.0,y90,s,None,0.0,False,N,NFE,0.5,1.0,0,delta,rk)
b0=base['gc_mean']
print(f"unguided gc_mean {b0:.4f}   target {y90:.4f}   gap {y90-b0:.4f}")
print(f"\n{'w':>5} {'t_min':>6} {'clip':>6} {'gc_mean':>8} {'closed':>8} {'clipped/step':>13}")
def run(w,tmin,clip):
    r,_=M.run_cell(net,ck,'plug','-',w,y90,s,None,0.0,False,N,NFE,tmin,clip,0,delta,rk)
    nguide=int(NFE*(1-tmin))
    return r, r['clipped_sample_steps']/max(nguide*N,1)
for w in (4,16,64):
    r,cf=run(w,0.5,1.0); print(f"{w:5.0f} {0.5:6.2f} {1.0:6.1f} {r['gc_mean']:8.4f} "
        f"{(r['gc_mean']-b0)/(y90-b0)*100:7.1f}% {cf:13.3f}")
print("  -- raise the clip --")
for clip in (4.0, 16.0):
    r,cf=run(16,0.5,clip); print(f"{16:5.0f} {0.5:6.2f} {clip:6.1f} {r['gc_mean']:8.4f} "
        f"{(r['gc_mean']-b0)/(y90-b0)*100:7.1f}% {cf:13.3f}")
print("  -- guide earlier --")
for tmin in (0.2, 0.0):
    r,cf=run(16,tmin,1.0); print(f"{16:5.0f} {tmin:6.2f} {1.0:6.1f} {r['gc_mean']:8.4f} "
        f"{(r['gc_mean']-b0)/(y90-b0)*100:7.1f}% {cf:13.3f}")
print("  -- both --")
r,cf=run(64,0.0,8.0); print(f"{64:5.0f} {0.0:6.2f} {8.0:6.1f} {r['gc_mean']:8.4f} "
    f"{(r['gc_mean']-b0)/(y90-b0)*100:7.1f}% {cf:13.3f}   kmerJS {r['kmer_js']:.5f} conf {r['decode_conf']:.3f}")
