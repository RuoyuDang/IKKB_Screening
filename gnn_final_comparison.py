#!/usr/bin/env python3
"""
GNN Models for IKKβ Inhibitor Classification
Stratified 5-fold CV (consistent with baseline evaluation)
"""
import sys, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import (GCNConv, GATConv, NNConv, global_mean_pool, global_max_pool)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef)
from rdkit import Chem
warnings = __import__('warnings'); warnings.filterwarnings('ignore')

# ===== Atom Features =====
ATOM_F = {'atomic_num':list(range(1,119)),'degree':[0,1,2,3,4,5,6],
    'formal_charge':[-5,-4,-3,-2,-1,0,1,2,3,4,5],'chiral_tag':[0,1,2,3],
    'num_Hs':[0,1,2,3,4,5,6,7,8],'hybridization':[0,1,2,3,4,5,6,7],
    'is_aromatic':[0,1],'is_in_ring':[0,1]}

def oh(v,c): return [(1 if v in c and c.index(v)==i else 0) for i in range(len(c))]

def build_graph(smi, label=None):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    m = Chem.AddHs(m)
    af = []
    for a in m.GetAtoms():
        f = []; f+=oh(a.GetAtomicNum(),ATOM_F['atomic_num']); f+=oh(a.GetTotalDegree(),ATOM_F['degree'])
        f+=oh(a.GetFormalCharge(),ATOM_F['formal_charge']); f+=oh(int(a.GetChiralTag()),ATOM_F['chiral_tag'])
        f+=oh(a.GetTotalNumHs(),ATOM_F['num_Hs']); f+=oh(int(a.GetHybridization()),ATOM_F['hybridization'])
        f+=oh(int(a.GetIsAromatic()),ATOM_F['is_aromatic']); f+=oh(int(a.IsInRing()),ATOM_F['is_in_ring'])
        af.append(f)
    x = torch.tensor(af,dtype=torch.float)
    ei,ea=[],[]
    for b in m.GetBonds():
        i,j=b.GetBeginAtomIdx(),b.GetEndAtomIdx(); ei+=[[i,j],[j,i]]
        bt=float(b.GetBondTypeAsDouble()); ea+=[[bt],[bt]]
    ei = torch.zeros((2,0),dtype=torch.long) if not ei else torch.tensor(ei,dtype=torch.long).t().contiguous()
    ea = torch.tensor(ea,dtype=torch.float) if ea else torch.zeros((0,1))
    y = torch.tensor([label],dtype=torch.float) if label is not None else None
    return Data(x=x,edge_index=ei,edge_attr=ea,y=y)

# ===== MLP Head =====
class Head(nn.Module):
    def __init__(self,d,h=64,do=0.3):
        super().__init__()
        self.n=nn.Sequential(nn.Linear(d,h),nn.BatchNorm1d(h),nn.ReLU(),nn.Dropout(do),
            nn.Linear(h,h//2),nn.BatchNorm1d(h//2),nn.ReLU(),nn.Dropout(do),nn.Linear(h//2,1))
    def forward(self,x): return self.n(x).squeeze(-1)

# ===== MODELS =====
class GCN(nn.Module):
    def __init__(self,nd,h=64,L=3,do=0.2):
        super().__init__(); self.p=nn.Linear(nd,h)
        self.cs=nn.ModuleList([GCNConv(h,h) for _ in range(L)])
        self.bs=nn.ModuleList([nn.BatchNorm1d(h) for _ in range(L)])
        self.do=do; self.h=Head(h,h,do)
    def forward(self,d):
        x,ei,b=d.x,d.edge_index,d.batch; x=F.relu(self.p(x))
        for c,bn in zip(self.cs,self.bs): x=F.relu(bn(c(x,ei))); x=F.dropout(x,p=self.do,training=self.training)
        return self.h(global_max_pool(x,b))

class GAT(nn.Module):
    def __init__(self,nd,h=64,L=3,hd=2,do=0.2):
        super().__init__(); self.p=nn.Linear(nd,h)
        self.cs,self.bs=[],[]
        for i in range(L):
            ind=h*(hd if i>0 else 1)
            self.cs.append(GATConv(ind,h,heads=hd,dropout=do)); self.bs.append(nn.BatchNorm1d(h*hd))
        self.do=do; self.h=Head(h*hd,h*2,do)
    def forward(self,d):
        x,ei,b=d.x,d.edge_index,d.batch; x=F.relu(self.p(x))
        for c,bn in zip(self.cs,self.bs): x=F.relu(bn(c(x,ei))); x=F.dropout(x,p=self.do,training=self.training)
        return self.h(global_max_pool(x,b))

class MPNN(nn.Module):
    def __init__(self,nd,ed=1,h=64,L=3,do=0.2):
        super().__init__(); self.p=nn.Linear(nd,h)
        self.cs,self.bs=[],[]
        for _ in range(L):
            en=nn.Sequential(nn.Linear(ed,h),nn.ReLU(),nn.Linear(h,h*h))
            self.cs.append(NNConv(h,h,en,aggr='mean')); self.bs.append(nn.BatchNorm1d(h))
        self.do=do; self.h=Head(h,h,do)
    def forward(self,d):
        x,ei,ea,b=d.x,d.edge_index,d.edge_attr,d.batch; x=F.relu(self.p(x))
        for c,bn in zip(self.cs,self.bs): x=F.relu(bn(c(x,ei,ea))); x=F.dropout(x,p=self.do,training=self.training)
        return self.h(global_max_pool(x,b))

class AttFP(nn.Module):
    def __init__(self,nd,h=64,L=3,do=0.2):
        super().__init__(); self.p=nn.Linear(nd,h)
        self.gs=nn.ModuleList([GATConv(h,h,heads=1,dropout=do,concat=False) for _ in range(L)])
        self.bs=nn.ModuleList([nn.BatchNorm1d(h) for _ in range(L)])
        self.at=nn.MultiheadAttention(h,2,dropout=do,batch_first=True)
        self.do=do; self.h=Head(h*2,h,do)
    def forward(self,d):
        x,ei,b=d.x,d.edge_index,d.batch; x=F.relu(self.p(x))
        for g,bn in zip(self.gs,self.bs): x=F.relu(bn(g(x,ei))); x=F.dropout(x,p=self.do,training=self.training)
        xm=global_max_pool(x,b); xs=[]
        for g in torch.unique(b):
            m=(b==g); xg=x[m].unsqueeze(0); xa,_=self.at(xg,xg,xg); xs.append(xa.squeeze(0).mean(0))
        return self.h(torch.cat([xm,torch.stack(xs)],-1))

class DMPNN(nn.Module):
    def __init__(self,nd,ed=1,h=64,D=3,do=0.2):
        super().__init__(); self.p=nn.Linear(nd,h)
        self.Wi=nn.Linear(h,h,bias=False); self.Wh=nn.Linear(h,h,bias=False); self.Wo=nn.Linear(h*2,h)
        self.us=nn.ModuleList([nn.Sequential(nn.Linear(h*2,h),nn.ReLU(),nn.Dropout(do),nn.Linear(h,h)) for _ in range(D)])
        self.D=D; self.do=do; self.h=Head(h,h,do)
    def forward(self,d):
        x,ei,b=d.x,d.edge_index,d.batch; h=F.relu(self.p(x)); r,c=ei
        hr,hc=self.Wi(h[r]),self.Wh(h[c]); bh=F.relu(self.Wo(torch.cat([hr,hc],-1)))
        for t in range(self.D):
            ai=torch.zeros(h.shape[0],bh.shape[1],device=h.device); ai=ai.index_add(0,c,bh)
            cnt=torch.zeros(h.shape[0],1,device=h.device); cnt=cnt.index_add(0,c,torch.ones(len(r),1,device=h.device)).clamp(min=1)
            inc=(ai/cnt)[r]; bh=bh+self.us[t](torch.cat([bh,inc],-1)); bh=F.dropout(bh,p=self.do,training=self.training)
        ae=torch.zeros(h.shape[0],bh.shape[1],device=h.device); ae=ae.index_add(0,c,bh)
        return self.h(global_max_pool(ae,b))

class Graphormer(nn.Module):
    def __init__(self,nd,h=64,L=2,nh=2,do=0.2):
        super().__init__(); self.p=nn.Linear(nd,h); self.de=nn.Embedding(20,h)
        ly=nn.TransformerEncoderLayer(h,nh,h*4,do,'gelu',batch_first=True)
        self.tr=nn.TransformerEncoder(ly,L); self.h=Head(h,h,do)
    def _dg(self,ei,n):
        dg=torch.zeros(n,dtype=torch.long,device=ei.device)
        return dg.index_add(0,ei[0],torch.ones(ei.shape[1],dtype=torch.long,device=ei.device)).clamp(max=19)
    def forward(self,d):
        x,ei,b=d.x,d.edge_index,d.batch; h=F.relu(self.p(x)+self.de(self._dg(ei,x.shape[0])))
        hs=[]; cls=torch.zeros(1,1,h.shape[1],device=h.device)
        for g in torch.unique(b):
            m=(b==g); hg=h[m].unsqueeze(0); hs.append(self.tr(torch.cat([cls,hg],1))[:,0,:])
        return self.h(torch.cat(hs,0))

# ===== Train/Eval =====
def te(m,ld,opt,crit):
    m.train(); tl,n=0,0
    for d in ld:
        opt.zero_grad(); l=crit(m(d),d.y); l.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        tl+=l.item()*d.num_graphs; n+=d.num_graphs
    return tl/max(n,1)

@torch.no_grad()
def ev(m,ld):
    m.eval(); ps,ls=[],[]
    for d in ld: ps.extend(torch.sigmoid(m(d)).cpu().numpy()); ls.extend(d.y.cpu().numpy())
    ps,ls=np.array(ps),np.array(ls); pb=(ps>=0.5).astype(int)
    return {
        'AUC-ROC':roc_auc_score(ls,ps),'PR-AUC':average_precision_score(ls,ps),
        'Accuracy':accuracy_score(ls,pb),'Balanced Accuracy':balanced_accuracy_score(ls,pb),
        'Precision':precision_score(ls,pb,zero_division=0),'Recall':recall_score(ls,pb,zero_division=0),
        'F1-Score':f1_score(ls,pb,zero_division=0),'MCC':matthews_corrcoef(ls,pb),
    }

MK=['AUC-ROC','PR-AUC','Accuracy','Balanced Accuracy','Precision','Recall','F1-Score','MCC']

def run(name,cls,need_ed,graphs,labels,**kw):
    print(f"\n{'='*50}\n  {name}\n{'='*50}",flush=True)
    fm={m:[] for m in MK}
    sg=graphs[0]; nd=sg.x.shape[1]; ed=sg.edge_attr.shape[1] if sg.edge_attr is not None else 1
    skf=StratifiedKFold(5,shuffle=True,random_state=42)
    y=np.array(labels)

    for fold,(ti,tsi) in enumerate(skf.split(np.zeros(len(labels)),y)):
        print(f"  Fold {fold+1}/5...",end=' ',flush=True)
        tl=DataLoader([graphs[i] for i in ti],32,shuffle=True)
        tsl=DataLoader([graphs[i] for i in tsi],32)

        if need_ed: m=cls(nd=nd,ed=ed,h=64,do=0.2,**kw)
        else: m=cls(nd=nd,h=64,do=0.2,**kw)

        np_=sum(y[ti]); nn_=len(ti)-np_; pw=torch.tensor([nn_/max(np_,1)]).float()
        crit=nn.BCEWithLogitsLoss(pos_weight=pw)
        opt=torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-5)

        best,best_st,pc=0,None,0
        for ep in range(100):
            te(m,tl,opt,crit); vm=ev(m,tsl); va=vm['AUC-ROC']
            if va>best: best=va; best_st={k:v.cpu().clone() for k,v in m.state_dict().items()}; pc=0
            else: pc+=1
            if pc>=20: break

        m.load_state_dict(best_st); tm=ev(m,tsl)
        for mk in MK: fm[mk].append(tm[mk])
        print(f"AUC={tm['AUC-ROC']:.4f} F1={tm['F1-Score']:.4f} MCC={tm['MCC']:.4f}",flush=True)

    avg={}
    print(f"  {'─'*40}")
    for mk in MK:
        v=fm[mk]; avg[mk]=f'{np.mean(v):.4f} ± {np.std(v):.4f}'
        print(f"  {mk:20s}: {avg[mk]}")
    return avg,fm

# ===== MAIN =====
print("="*50,flush=True)
print("GNN Models: Stratified 5-Fold CV",flush=True)
print("="*50,flush=True)

df=pd.read_csv('/mnt/f/Dangruoyu/260727/ikkb_05_bioactivity_training_1458+362SMILES.csv')
smiles=df['canonical_smiles'].tolist(); labels=df['label'].tolist()
print(f"Data: {len(df)} ({sum(labels)} active, {len(labels)-sum(labels)} inactive)",flush=True)

print("Building graphs...",end=' ',flush=True)
graphs=[g for g in (build_graph(s,l) for s,l in zip(smiles,labels)) if g is not None]
print(f"{len(graphs)} valid",flush=True)

models=[
    ('GCN',GCN,False,{}),
    ('GAT',GAT,False,{}),
    ('MPNN',MPNN,True,{}),
    ('AttentiveFP',AttFP,False,{}),
    ('D-MPNN',DMPNN,True,{}),
    ('Graphormer',Graphormer,False,{}),
]

all_res={}; all_folds={}
for nm,cls,ne,kw in models:
    avg,folds=run(nm,cls,ne,graphs,labels,**kw)
    all_res[nm]=avg; all_folds[nm]=folds

# Merge with previous results
prev={
    'CMPNN':{'AUC-ROC':'0.9446±0.0101','PR-AUC':'0.9859±0.0046','Accuracy':'0.9322±0.0129',
        'Balanced Accuracy':'0.8945±0.0280','Precision':'0.9679±0.0118','Recall':'0.9507±0.0153',
        'F1-Score':'0.9591±0.0079','MCC':'0.7636±0.0444'},
    'Random Forest':{'AUC-ROC':'0.9636±0.0096','PR-AUC':'0.9919±0.0028','Accuracy':'0.9322±0.0171',
        'Balanced Accuracy':'0.8476±0.0339','Precision':'0.9471±0.0130','Recall':'0.9735±0.0148',
        'F1-Score':'0.9600±0.0102','MCC':'0.7427±0.0639'},
    'Gradient Boosting':{'AUC-ROC':'0.9175±0.0095','PR-AUC':'0.9756±0.0062','Accuracy':'0.9190±0.0082',
        'Balanced Accuracy':'0.8139±0.0250','Precision':'0.9357±0.0102','Recall':'0.9701±0.0162',
        'F1-Score':'0.9524±0.0050','MCC':'0.6877±0.0345'},
    'RBF SVM':{'AUC-ROC':'0.9524±0.0108','PR-AUC':'0.9888±0.0038','Accuracy':'0.9219±0.0128',
        'Balanced Accuracy':'0.8521±0.0302','Precision':'0.9511±0.0114','Recall':'0.9560±0.0081',
        'F1-Score':'0.9535±0.0076','MCC':'0.7119±0.0481'},
}
all_res.update(prev)

# ===== TABLE =====
order=['CMPNN','GCN','GAT','MPNN','AttentiveFP','D-MPNN','Graphormer','Random Forest','Gradient Boosting','RBF SVM']
print("\n"+"="*130)
print("COMPREHENSIVE TABLE: All Models (5-Fold Stratified CV)")
print("="*130)

rows=[]
for nm in order:
    if nm in all_res:
        row={'Model':nm}
        for mk in MK: row[mk]=all_res[nm].get(mk,'N/A')
        rows.append(row)

tdf=pd.DataFrame(rows)
print(tdf.to_string(index=False))
tdf.to_csv('/mnt/f/Dangruoyu/260727/table3_all_models_comprehensive.csv',index=False)

# Fold details
frows=[]
for nm in ['GCN','GAT','MPNN','AttentiveFP','D-MPNN','Graphormer']:
    if nm in all_folds:
        print(f"\n{nm} fold-level:")
        for mk in MK:
            vals=[f'{v:.4f}' for v in all_folds[nm][mk]]
            print(f"  {mk}: [{', '.join(vals)}]")
        for f in range(5):
            r={'Model':nm,'Fold':f+1}
            for mk in MK: r[mk]=round(all_folds[nm][mk][f],4)
            frows.append(r)

fdf=pd.DataFrame(frows)
fdf.to_csv('/mnt/f/Dangruoyu/260727/table_s2_gnn_fold_metrics.csv',index=False)
print(f"\n✓ Results saved!",flush=True)
