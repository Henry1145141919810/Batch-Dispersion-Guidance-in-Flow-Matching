"""Research smoke checks only; does not install or replace project predictors."""
from pathlib import Path
import argparse
import ast
import io
import json
import math
import pickle
import zipfile
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(2)
torch.manual_seed(1909)

class MetadataReader(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {('argparse', 'Namespace'): argparse.Namespace,
                   ('torch', 'device'): torch.device}
        if (module, name) not in allowed:
            raise pickle.UnpicklingError((module, name))
        return allowed[module, name]

def metadata(path):
    return vars(MetadataReader(io.BytesIO(path.read_bytes())).load())

def definitions(path, names, namespace=None):
    """Load only reviewed class/function definitions, never module top-level code."""
    scope = dict(torch=torch, nn=nn, math=math)
    scope.update(namespace or {})
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    tree.body = [n for n in tree.body
                 if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    exec(compile(tree, str(path), 'exec'), scope)
    return scope

def edges(mask):
    b, n = mask.shape
    idx = torch.arange(b*n).reshape(b,n)
    row = idx[:,:,None].expand(b,n,n).reshape(-1)
    col = idx[:,None,:].expand(b,n,n).reshape(-1)
    pm = mask[:,:,None] * mask[:,None,:]
    pm = pm * (1-torch.eye(n, dtype=mask.dtype)[None])
    return [row,col], pm.reshape(-1,1)

def check(fn, x, f, mask):
    def deriv(c, a):
        c=c.detach().clone().requires_grad_()
        a=a.detach().clone().requires_grad_()
        y=fn(c,a,mask)
        g=torch.autograd.grad(y.sum(),(c,a),create_graph=True)
        v=[torch.randn_like(z) for z in g]
        hv=torch.autograd.grad(sum((u*w).sum() for u,w in zip(g,v)),(c,a))
        return y.detach(), [z.detach() for z in g], [z.detach() for z in hv]
    y,g,hv=deriv(x,f)
    q,_=torch.linalg.qr(torch.randn(3,3,dtype=x.dtype))
    q[:,0]*=torch.linalg.det(q)
    perm=torch.randperm(x.shape[1])
    yr=fn((x@q).contiguous(),f,mask)
    yt=fn(x+torch.tensor([2.,-3.,4.],dtype=x.dtype),f,mask)
    yp=fn(x[:,perm].contiguous(),f[:,perm].contiguous(),mask[:,perm].contiguous())
    dirty_x=x.clone(); dirty_f=f.clone()
    dirty_x[mask==0]=23.; dirty_f[mask==0]=-7.
    yd,gd,_=deriv(dirty_x,dirty_f)
    collisions=[]
    for eps in [0.,1e-9]:
        xc=x.clone(); xc[:,1]=xc[:,0]+eps
        yc,gc,hc=deriv(xc,f)
        collisions.append(bool(all(torch.isfinite(z).all() for z in [yc,*gc,*hc])))
    # Exact input Hessian trace on the first molecule, including coordinates
    # and continuous atom features. This is a smoke check, not a QM9 statistic.
    tc=x[:1].detach().clone().requires_grad_()
    tf=f[:1].detach().clone().requires_grad_()
    ty=fn(tc,tf,mask[:1]).sum()
    tg=torch.autograd.grad(ty,(tc,tf),create_graph=True)
    traces=[]
    for grad, variable in zip(tg,(tc,tf)):
        trace=0.
        for j in range(variable.numel()):
            row=torch.autograd.grad(grad.reshape(-1)[j],variable,retain_graph=True)[0]
            trace+=float(row.reshape(-1)[j])
        traces.append(trace)
    rel=lambda a: float(((a-y).abs()/y.abs().clamp_min(1.)).max())
    return dict(first_and_second_finite=bool(all(torch.isfinite(z).all() for z in [y,*g,*hv])),
                hvp_norm=[float(z.norm()) for z in hv],
                exact_hessian_trace_coords_and_features=traces,
                invariance_scaled_error=dict(rotation=rel(yr),translation=rel(yt),permutation=rel(yp)),
                padding_output_error=float((yd-y).abs().max()),
                padding_gradient_error=max(float((u-v).abs().max()) for u,v in zip(g,gd)),
                padded_gradient_max=max(float(z[mask==0].abs().max()) for z in g),
                collision_and_near_collision_finite=collisions)

data=torch.load(ROOT.parents[1]/'data/qm9.pt',map_location='cpu',weights_only=True)
x=data['coords'][100:102].double()[:,:20].contiguous()
f=data['feats'][100:102].double()[:,:20].contiguous()
m=data['mask'][100:102].double()[:,:20].contiguous()
assert (m.sum(1)==data['mask'][100:102].sum(1)).all()

energy=definitions(ROOT/'TFG/tasks/networks/egnn/energy.py',
    {'GCL','EquivariantBlock','EGNN','SinusoidsEmbeddingNew','coord2diff','unsorted_segment_sum'})
ga=metadata(ROOT/'TFG/tf_predict_mu/args_2000.pickle')
guide=energy['EGNN'](in_node_nf=6,in_edge_nf=1,hidden_nf=ga['nf'],n_layers=ga['n_layers'],
    attention=ga['attention'],tanh=ga['tanh'],norm_constant=ga['norm_constant'],
    inv_sublayers=ga['inv_sublayers'],sin_embedding=ga['sin_embedding'],
    normalization_factor=ga['normalization_factor'],aggregation_method=ga['aggregation_method'])
state=torch.load(ROOT/'TFG/tf_predict_mu/model_ema_2000.npy',map_location='cpu',weights_only=True)
prefix='dynamics.egnn.'
guide.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)})
guide=guide.double().eval().requires_grad_(False)
def guide_fn(c,a,mask):
    b,n,_=c.shape
    edge,pm=edges(mask)
    h=torch.cat([a/ga['normalize_factors'][1],torch.zeros(b,n,1,dtype=c.dtype)],dim=-1)
    return guide((h*mask[...,None]).reshape(-1,6),
        (c*mask[...,None]/ga['normalize_factors'][0]).reshape(-1,3),edge,
        node_mask=mask.reshape(-1,1),edge_mask=pm,n_nodes=n)

gcl=definitions(ROOT/'OC-Flow/molecule/qm9/property_prediction/models/gcl.py',
    {'E_GCL','unsorted_segment_sum','unsorted_segment_mean'})
oracle_defs=definitions(ROOT/'OC-Flow/molecule/qm9/property_prediction/models_property.py',
    {'E_GCL_mask','EGNN'},gcl)
oa=metadata(ROOT/'TFG/evaluate_mu/args.pickle')
oracle=oracle_defs['EGNN'](in_node_nf=5,in_edge_nf=0,hidden_nf=oa['nf'],
    n_layers=oa['n_layers'],attention=oa['attention'],node_attr=oa['node_attr'])
oracle.load_state_dict(torch.load(ROOT/'TFG/evaluate_mu/best_checkpoint.npy',map_location='cpu',weights_only=True))
oracle=oracle.double().eval().requires_grad_(False)
def oracle_fn(c,a,mask):
    b,n,_=c.shape
    edge,pm=edges(mask)
    return oracle(a.reshape(-1,5),c.reshape(-1,3),edge,None,mask.reshape(-1,1),pm,n)

results={'scope':'Two real local molecules, float64 CPU. Raw normalized outputs. No held-out MAE or split-ID certification.',
    'guide_metadata':ga,'oracle_metadata':{k:str(v) if isinstance(v,torch.device) else v for k,v in oa.items()},
    'TFG_mu_guide':check(guide_fn,x,f,m),'TFG_mu_oracle':check(oracle_fn,x,f,m)}

flow=definitions(ROOT/'TFG-Flow/networks/egnn.py',
    {'GCL','EquivariantUpdate','EquivariantBlock','EGNN','SinusoidsEmbeddingNew','coord2diff','unsorted_segment_sum'})
with zipfile.ZipFile(ROOT/'TFG-Flow/storage/guide_clf_ckpt.zip') as z:
    s=torch.load(io.BytesIO(z.read('guide_clf_ckpt/best_loss=0.20151185317914316.pth')),
                 map_location='cpu',weights_only=True)
fg=flow['EGNN'](in_node_nf=128,in_edge_nf=1,hidden_nf=128,n_layers=6,attention=True,
    tanh=True,norm_constant=1,inv_sublayers=1,sin_embedding=False,normalization_factor=1,aggregation_method='sum')
fg.load_state_dict({k[4:]:v for k,v in s.items() if k.startswith('gnn.')})
fg=fg.double().eval().requires_grad_(False)
emb=s['atom_emb.weight'].double()[[1,0,2,3,4]]
pw=s['predictor.weight'].double(); pb=s['predictor.bias'].double()
def flow_fn(c,a,mask):
    h=(a@emb)*mask[...,None]
    h,_=fg(h,c,mask.bool())
    pooled=(h*mask[...,None]).sum(1)/mask.sum(1,keepdim=True)
    return (pooled@pw.T+pb).squeeze(-1)
# Keep the TFG-Flow test within its heavy-atom training domain.
fx=torch.zeros(2,9,3,dtype=torch.float64); ff=torch.zeros(2,9,5,dtype=torch.float64); fm=torch.zeros(2,9,dtype=torch.float64)
for b in range(2):
    select=(m[b]>0)&(f[b,:,0]==0)
    n=int(select.sum()); fx[b,:n]=x[b,select]; ff[b,:n]=f[b,select]; fm[b,:n]=1
results['TFG_Flow_mu_guide_with_linear_embedding_adapter']=check(flow_fn,fx,ff,fm)
(ROOT/'candidate_checks.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
