#!/usr/bin/env python3
"""
Fast GNN Model Comparison for IKKβ Inhibitor Classification
=============================================================
Implements GCN, GAT, MPNN, AttentiveFP, D-MPNN, Graphormer with:
- 3-fold scaffold-based CV
- 50 epochs, patience=15
- hidden_dim=64
- Comprehensive metrics output
"""
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import (GCNConv, GATConv, NNConv, global_mean_pool, global_max_pool)
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef)
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
import warnings
warnings.filterwarnings('ignore')

# ===== Atom Features =====
ATOM_FEATURES = {
    'atomic_num': list(range(1,119)), 'degree': [0,1,2,3,4,5,6],
    'formal_charge': [-5,-4,-3,-2,-1,0,1,2,3,4,5], 'chiral_tag': [0,1,2,3],
    'num_Hs': [0,1,2,3,4,5,6,7,8], 'hybridization': [0,1,2,3,4,5,6,7],
    'is_aromatic': [0,1], 'is_in_ring': [0,1],
}

def onehot(v, choices):
    enc = [0]*len(choices)
    if v in choices: enc[choices.index(v)] = 1
    return enc

def build_graph(smiles, label=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    mol = Chem.AddHs(mol)
    af = []
    for atom in mol.GetAtoms():
        f = []
        f += onehot(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num'])
        f += onehot(atom.GetTotalDegree(), ATOM_FEATURES['degree'])
        f += onehot(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge'])
        f += onehot(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag'])
        f += onehot(atom.GetTotalNumHs(), ATOM_FEATURES['num_Hs'])
        f += onehot(int(atom.GetHybridization()), ATOM_FEATURES['hybridization'])
        f += onehot(int(atom.GetIsAromatic()), ATOM_FEATURES['is_aromatic'])
        f += onehot(int(atom.IsInRing()), ATOM_FEATURES['is_in_ring'])
        af.append(f)
    x = torch.tensor(af, dtype=torch.float)
    ei, ea = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ei += [[i,j],[j,i]]
        bt = float(bond.GetBondTypeAsDouble())
        ea += [[bt],[bt]]
    if not ei: ei = torch.zeros((2,0), dtype=torch.long)
    else: ei = torch.tensor(ei, dtype=torch.long).t().contiguous()
    ea = torch.tensor(ea, dtype=torch.float) if ea else torch.zeros((0,1))
    y = torch.tensor([label], dtype=torch.float) if label is not None else None
    return Data(x=x, edge_index=ei, edge_attr=ea, y=y)

# ===== Scaffold Split =====
def get_scaffold(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: return smi
        s = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(s) if s else smi
    except: return smi

def scaffold_split(smiles_list, labels, test_size=0.2, val_size=0.1, seed=42):
    np.random.seed(seed)
    groups = defaultdict(list)
    for i, smi in enumerate(smiles_list): groups[get_scaffold(smi)].append(i)
    scaffolds = sorted(groups.keys(), key=lambda s: len(groups[s]), reverse=True)
    n = len(smiles_list)
    tc, vc = int(n*test_size), int(n*val_size)
    test_i, val_i, train_i = [], [], []
    for s in scaffolds:
        ids = groups[s]
        if len(test_i) < tc: test_i.extend(ids)
        elif len(val_i) < vc: val_i.extend(ids)
        else: train_i.extend(ids)
    return train_i, val_i, test_i

# ===== MLP Head =====
class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.BatchNorm1d(hidden//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden//2, 1),
        )
    def forward(self, x): return self.net(x).squeeze(-1)

# ===== GCN =====
class GCNNet(nn.Module):
    def __init__(self, node_dim, hidden=64, n_layers=3, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(node_dim, hidden)
        self.convs = nn.ModuleList([GCNConv(hidden, hidden) for _ in range(n_layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(n_layers)])
        self.do = dropout
        self.head = MLPHead(hidden, hidden, dropout)
    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.proj(x))
        for c, b in zip(self.convs, self.bns):
            x = F.relu(b(c(x, ei)))
            x = F.dropout(x, p=self.do, training=self.training)
        return self.head(global_max_pool(x, batch))

# ===== GAT =====
class GATNet(nn.Module):
    def __init__(self, node_dim, hidden=64, n_layers=3, heads=2, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(node_dim, hidden)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(n_layers):
            ind = hidden*(heads if i>0 else 1)
            self.convs.append(GATConv(ind, hidden, heads=heads, dropout=dropout))
            self.bns.append(nn.BatchNorm1d(hidden*heads))
        self.do = dropout
        self.head = MLPHead(hidden*heads, hidden*2, dropout)
    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.proj(x))
        for c, b in zip(self.convs, self.bns):
            x = F.relu(b(c(x, ei)))
            x = F.dropout(x, p=self.do, training=self.training)
        return self.head(global_max_pool(x, batch))

# ===== MPNN =====
class MPNNNet(nn.Module):
    def __init__(self, node_dim, edge_dim=1, hidden=64, n_layers=3, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(node_dim, hidden)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(n_layers):
            enet = nn.Sequential(nn.Linear(edge_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden*hidden))
            self.convs.append(NNConv(hidden, hidden, enet, aggr='mean'))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.do = dropout
        self.head = MLPHead(hidden, hidden, dropout)
    def forward(self, data):
        x, ei, ea, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.relu(self.proj(x))
        for c, b in zip(self.convs, self.bns):
            x = F.relu(b(c(x, ei, ea)))
            x = F.dropout(x, p=self.do, training=self.training)
        return self.head(global_max_pool(x, batch))

# ===== AttentiveFP =====
class AttentiveFPNet(nn.Module):
    def __init__(self, node_dim, hidden=64, n_layers=3, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(node_dim, hidden)
        self.gats = nn.ModuleList([GATConv(hidden, hidden, heads=1, dropout=dropout, concat=False) for _ in range(n_layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(n_layers)])
        self.attn = nn.MultiheadAttention(hidden, num_heads=2, dropout=dropout, batch_first=True)
        self.do = dropout
        self.head = MLPHead(hidden*2, hidden, dropout)
    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.proj(x))
        for g, b in zip(self.gats, self.bns):
            x = F.relu(b(g(x, ei)))
            x = F.dropout(x, p=self.do, training=self.training)
        x_max = global_max_pool(x, batch)
        xs = []
        for g in torch.unique(batch):
            m = (batch==g); xg = x[m].unsqueeze(0)
            xa, _ = self.attn(xg, xg, xg); xs.append(xa.squeeze(0).mean(0))
        return self.head(torch.cat([x_max, torch.stack(xs)], dim=-1))

# ===== D-MPNN =====
class DMPNNNet(nn.Module):
    def __init__(self, node_dim, edge_dim=1, hidden=64, depth=3, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(node_dim, hidden)
        self.W_i = nn.Linear(hidden, hidden, bias=False)
        self.W_h = nn.Linear(hidden, hidden, bias=False)
        self.W_o = nn.Linear(hidden*2, hidden)
        self.updates = nn.ModuleList([nn.Sequential(nn.Linear(hidden*2, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden)) for _ in range(depth)])
        self.depth = depth; self.do = dropout;
        self.head = MLPHead(hidden, hidden, dropout)
    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        h = F.relu(self.proj(x))
        row, col = ei
        hr, hc = self.W_i(h[row]), self.W_h(h[col])
        bh = F.relu(self.W_o(torch.cat([hr, hc], -1)))
        for t in range(self.depth):
            ai = torch.zeros(h.shape[0], bh.shape[1], device=h.device)
            ai = ai.index_add(0, col, bh)
            cnt = torch.zeros(h.shape[0], 1, device=h.device)
            cnt = cnt.index_add(0, col, torch.ones(len(row),1,device=h.device)).clamp(min=1)
            incoming = (ai/cnt)[row]
            bh = bh + self.updates[t](torch.cat([bh, incoming], -1))
            bh = F.dropout(bh, p=self.do, training=self.training)
        ae = torch.zeros(h.shape[0], bh.shape[1], device=h.device)
        ae = ae.index_add(0, col, bh)
        return self.head(global_max_pool(ae, batch))

# ===== Graphormer =====
class GraphormerNet(nn.Module):
    def __init__(self, node_dim, hidden=64, n_layers=2, n_heads=2, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(node_dim, hidden)
        self.deg_emb = nn.Embedding(20, hidden)
        layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=n_heads, dim_feedforward=hidden*4, dropout=dropout, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(layer, n_layers)
        self.head = MLPHead(hidden, hidden, dropout)
    def _deg(self, ei, n):
        d = torch.zeros(n, dtype=torch.long, device=ei.device)
        return d.index_add(0, ei[0], torch.ones(ei.shape[1], dtype=torch.long, device=ei.device)).clamp(max=19)
    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        h = F.relu(self.proj(x) + self.deg_emb(self._deg(ei, x.shape[0])))
        hs = []
        for g in torch.unique(batch):
            m = (batch==g); hg = h[m].unsqueeze(0)
            cls = torch.zeros(1,1,h.shape[1],device=h.device)
            hs.append(self.transformer(torch.cat([cls, hg],1))[:,0,:])
        return self.head(torch.cat(hs, 0))

# ===== Train/Eval =====
def train_epoch(m, ld, opt, crit):
    m.train(); tl, ng = 0, 0
    for d in ld:
        opt.zero_grad(); loss = crit(m(d), d.y); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        tl += loss.item()*d.num_graphs; ng += d.num_graphs
    return tl/max(ng,1)

@torch.no_grad()
def ev(m, ld):
    m.eval(); ps, ls = [], []
    for d in ld:
        ps.extend(torch.sigmoid(m(d)).cpu().numpy()); ls.extend(d.y.cpu().numpy())
    ps, ls = np.array(ps), np.array(ls); pb = (ps>=0.5).astype(int)
    return {k: fn(ls, ps if 'PR' in k or 'ROC' in k else pb) for k, fn in [
        ('AUC-ROC', roc_auc_score), ('PR-AUC', average_precision_score),
        ('Accuracy', accuracy_score), ('Balanced Accuracy', balanced_accuracy_score),
        ('Precision', lambda a,b: precision_score(a,b,zero_division=0)),
        ('Recall', lambda a,b: recall_score(a,b,zero_division=0)),
        ('F1-Score', lambda a,b: f1_score(a,b,zero_division=0)),
        ('MCC', matthews_corrcoef),
    ]}

METRICS = ['AUC-ROC','PR-AUC','Accuracy','Balanced Accuracy','Precision','Recall','F1-Score','MCC']

def run_model(model_cls, name, needs_edge, graphs, smiles, labels, **kw):
    print(f"\n{'='*55}\n  {name}\n{'='*55}", flush=True)
    fm = {m:[] for m in METRICS}
    sg = graphs[0]; nd = sg.x.shape[1]; ed = sg.edge_attr.shape[1] if sg.edge_attr is not None else 1

    for fold in range(3):
        print(f"  Fold {fold+1}/3...", end=' ', flush=True)
        ti, vi, tsti = scaffold_split(smiles, labels, 0.2, 0.1, 42+fold)
        tl = DataLoader([graphs[i] for i in ti], 32, shuffle=True)
        vl = DataLoader([graphs[i] for i in vi], 32)
        tsl = DataLoader([graphs[i] for i in tsti], 32)

        if needs_edge: m = model_cls(node_dim=nd, edge_dim=ed, hidden=64, dropout=0.2, **kw)
        else: m = model_cls(node_dim=nd, hidden=64, dropout=0.2, **kw)

        tlabels = [labels[i] for i in ti]
        np_, nn_ = sum(tlabels), len(tlabels)-sum(tlabels)
        pw = torch.tensor([nn_/max(np_,1)]).float()
        crit = nn.BCEWithLogitsLoss(pos_weight=pw)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)

        best, best_st, pc = 0, None, 0
        for ep in range(50):
            train_epoch(m, tl, opt, crit)
            vm = ev(m, vl); va = vm['AUC-ROC']
            if va > best: best = va; best_st = {k:v.cpu().clone() for k,v in m.state_dict().items()}; pc = 0
            else: pc += 1
            if pc >= 15: break

        m.load_state_dict(best_st); tm = ev(m, tsl)
        for mk in METRICS: fm[mk].append(tm[mk])
        print(f"AUC={tm['AUC-ROC']:.4f} F1={tm['F1-Score']:.4f} MCC={tm['MCC']:.4f}", flush=True)

    avg = {}
    print(f"  {'─'*40}")
    for mk in METRICS:
        v = fm[mk]; avg[mk] = f'{np.mean(v):.4f} ± {np.std(v):.4f}'
        print(f"  {mk:20s}: {avg[mk]}")
    return avg, fm

# ===== MAIN =====
print("="*55)
print("GNN Model Comparison for IKKβ Inhibitor Classification")
print("="*55, flush=True)

df = pd.read_csv('/mnt/f/Dangruoyu/260727/ikkb_05_bioactivity_training_1458+362SMILES.csv')
smiles = df['canonical_smiles'].tolist()
labels = df['label'].tolist()
print(f"Data: {len(df)} compounds ({sum(labels)} active, {len(labels)-sum(labels)} inactive)", flush=True)

print("Building graphs...", end=' ', flush=True)
graphs = [g for g in (build_graph(s,l) for s,l in zip(smiles,labels)) if g is not None]
print(f"{len(graphs)} valid", flush=True)

# Models: (name, class, needs_edge_dim, kwargs)
models = [
    ('GCN',      GCNNet,       False, {}),
    ('GAT',      GATNet,       False, {}),
    ('MPNN',     MPNNNet,      True,  {}),
    ('AttentiveFP', AttentiveFPNet, False, {}),
    ('D-MPNN',   DMPNNNet,     True,  {}),
    ('Graphormer', GraphormerNet, False, {}),
]

all_res = {}
all_folds = {}
for nm, cls, need_e, kw in models:
    avg, folds = run_model(cls, nm, need_e, graphs, smiles, labels, **kw)
    all_res[nm] = avg
    all_folds[nm] = folds

# Add previous CMPNN + baselines
all_res.update({
    'CMPNN': {'AUC-ROC':'0.9446±0.0101','PR-AUC':'0.9859±0.0046','Accuracy':'0.9322±0.0129',
              'Balanced Accuracy':'0.8945±0.0280','Precision':'0.9679±0.0118','Recall':'0.9507±0.0153',
              'F1-Score':'0.9591±0.0079','MCC':'0.7636±0.0444'},
    'Random Forest': {'AUC-ROC':'0.9636±0.0096','PR-AUC':'0.9919±0.0028','Accuracy':'0.9322±0.0171',
              'Balanced Accuracy':'0.8476±0.0339','Precision':'0.9471±0.0130','Recall':'0.9735±0.0148',
              'F1-Score':'0.9600±0.0102','MCC':'0.7427±0.0639'},
    'Gradient Boosting': {'AUC-ROC':'0.9175±0.0095','PR-AUC':'0.9756±0.0062','Accuracy':'0.9190±0.0082',
              'Balanced Accuracy':'0.8139±0.0250','Precision':'0.9357±0.0102','Recall':'0.9701±0.0162',
              'F1-Score':'0.9524±0.0050','MCC':'0.6877±0.0345'},
    'RBF SVM': {'AUC-ROC':'0.9524±0.0108','PR-AUC':'0.9888±0.0038','Accuracy':'0.9219±0.0128',
              'Balanced Accuracy':'0.8521±0.0302','Precision':'0.9511±0.0114','Recall':'0.9560±0.0081',
              'F1-Score':'0.9535±0.0076','MCC':'0.7119±0.0481'},
})

# ==== COMPREHENSIVE TABLE ====
order = ['CMPNN','GCN','GAT','MPNN','AttentiveFP','D-MPNN','Graphormer','Random Forest','Gradient Boosting','RBF SVM']

print("\n"+"="*120)
print("TABLE: Comprehensive Model Performance Comparison (3-Fold Scaffold-Based CV)")
print("="*120)

rows = []
for nm in order:
    if nm in all_res:
        row = {'Model': nm}
        for mk in METRICS: row[mk] = all_res[nm].get(mk, 'N/A')
        rows.append(row)

tdf = pd.DataFrame(rows)
print(tdf.to_string(index=False))
tdf.to_csv('/mnt/f/Dangruoyu/260727/table3_all_models_comprehensive.csv', index=False)

# Fold-level details
print("\n\nFold-Level Details for GNN Models:")
fold_rows = []
for nm in ['GCN','GAT','MPNN','AttentiveFP','D-MPNN','Graphormer']:
    if nm in all_folds:
        print(f"\n{nm}:")
        for mk in METRICS:
            vals = [f'{v:.4f}' for v in all_folds[nm][mk]]
            print(f"  {mk}: [{', '.join(vals)}]")
        for f in range(3):
            row = {'Model': nm, 'Fold': f+1}
            for mk in METRICS: row[mk] = round(all_folds[nm][mk][f], 4)
            fold_rows.append(row)

fdf = pd.DataFrame(fold_rows)
fdf.to_csv('/mnt/f/Dangruoyu/260727/table_s2_gnn_fold_metrics.csv', index=False)

print(f"\nResults saved: table3_all_models_comprehensive.csv, table_s2_gnn_fold_metrics.csv")
print("DONE!", flush=True)
