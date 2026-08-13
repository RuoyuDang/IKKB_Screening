#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CMPNN activity-discrimination model for IKKβ kinase inhibitors
===============================================================

Pipeline
--------
1. Load bioactivity dataset (canonical SMILES + active/inactive class).
2. Build attributed molecular graphs (RDKit atom/bond features).
3. Scaffold-based partitioning (Bemis–Murcko scaffolds):
     - training (70%) / validation (10%) / test (20%) hold-out split, and
     - scaffold-disjoint 5-fold cross-validation.
4. Train the Communicative Message Passing Neural Network (CMPNN) with a
   class-weighted binary cross-entropy loss and early stopping.
5. Report cross-validation performance metrics (per-fold + mean ± SD).

Outputs (written next to this script)
-------------------------------------
- cmpnn_cv_fold_metrics.csv       : per-fold metrics for the 5-fold CV
- cmpnn_cv_summary.csv            : mean ± SD across folds
- cmpnn_holdout_test_metrics.csv  : test metrics from the 70/10/20 hold-out
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import global_mean_pool, global_max_pool
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef,
)
from collections import defaultdict
import warnings

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

warnings.filterwarnings('ignore')

SEED = 42
DATA_PATH = '/mnt/f/Dangruoyu/260727/ikkb_05_bioactivity_training_1725+_SMILES.xls'

METRIC_KEYS = ['AUC-ROC', 'PR-AUC', 'Accuracy', 'Balanced Accuracy',
               'Precision', 'Recall', 'F1-Score', 'MCC']


# ============================================================================
# 1. Molecular graph construction
# ============================================================================

ATOM_FEATURES = {
    'atomic_num':    list(range(1, 119)),            # 118
    'degree':        [0, 1, 2, 3, 4, 5, 6],          # 7
    'formal_charge': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],  # 11
    'chiral_tag':    [0, 1, 2, 3],                   # 4
    'num_Hs':        [0, 1, 2, 3, 4, 5, 6, 7, 8],    # 9
    'hybridization': [0, 1, 2, 3, 4, 5, 6, 7],       # 8
    'is_aromatic':   [0, 1],                         # 2
    'is_in_ring':    [0, 1],                         # 2
}

BOND_TYPES = [1, 2, 3, 12]  # single, double, triple, aromatic
BOND_STEREO = [0, 1, 2, 3, 4, 5]


def one_hot(value, choices):
    enc = [0] * len(choices)
    if value in choices:
        enc[choices.index(value)] = 1
    return enc


def atom_features(atom):
    f = []
    f += one_hot(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num'])
    f += one_hot(atom.GetTotalDegree(), ATOM_FEATURES['degree'])
    f += one_hot(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge'])
    f += one_hot(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag'])
    f += one_hot(atom.GetTotalNumHs(), ATOM_FEATURES['num_Hs'])
    f += one_hot(int(atom.GetHybridization()), ATOM_FEATURES['hybridization'])
    f += one_hot(int(atom.GetIsAromatic()), ATOM_FEATURES['is_aromatic'])
    f += one_hot(int(atom.IsInRing()), ATOM_FEATURES['is_in_ring'])
    return f


def bond_features(bond):
    f = []
    f += one_hot(int(bond.GetBondType()), BOND_TYPES)
    f += one_hot(int(bond.GetIsConjugated()), [0, 1])
    f += one_hot(int(bond.IsInRing()), [0, 1])
    f += one_hot(int(bond.GetStereo()), BOND_STEREO)
    return f


def build_graph(smiles, label=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)

    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index += [[i, j], [j, i]]
        bf = bond_features(bond)
        edge_attr += [bf, bf]

    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float) if edge_attr else torch.zeros((0, len(BOND_TYPES) + len(BOND_STEREO) + 4))

    y = torch.tensor([label], dtype=torch.float) if label is not None else None
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


# ============================================================================
# 2. CMPNN model
# ============================================================================

class CMPNNLayer(nn.Module):
    """One communicative message-passing step: bond -> atom -> bond."""

    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        h = hidden_dim
        # bond message from (atom_u, atom_v, bond)
        self.msg_mlp = nn.Sequential(
            nn.Linear(3 * h, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h, h),
        )
        # atom update from (atom_v, aggregated incoming messages)
        self.atom_update = nn.Sequential(
            nn.Linear(2 * h, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h, h),
        )
        # communicative bond update from (updated atoms, bond)
        self.bond_update = nn.Sequential(
            nn.Linear(3 * h, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h, h),
        )

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index

        # 1) bond <- atoms
        msg_in = torch.cat([x[row], x[col], edge_attr], dim=-1)
        m = self.msg_mlp(msg_in)

        # 2) atom <- incoming bond messages (mean aggregation)
        agg = torch.zeros(x.shape[0], m.shape[1], device=x.device)
        agg = agg.index_add(0, col, m)
        deg = torch.zeros(x.shape[0], 1, device=x.device)
        deg = deg.index_add(0, col, torch.ones(m.shape[0], 1, device=x.device)).clamp(min=1)
        agg = agg / deg
        x_new = x + self.atom_update(torch.cat([x, agg], dim=-1))

        # 3) bond <- updated atoms (communicative step)
        bond_in = torch.cat([x_new[row], x_new[col], edge_attr], dim=-1)
        edge_attr_new = edge_attr + self.bond_update(bond_in)

        return x_new, edge_attr_new


class CMPNN(nn.Module):
    """Communicative Message Passing Neural Network for binary classification."""

    def __init__(self, node_dim, edge_dim, hidden_dim=128, num_layers=3, dropout=0.2):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.layers = nn.ModuleList([CMPNNLayer(hidden_dim, dropout) for _ in range(num_layers)])

        readout_dim = 2 * hidden_dim  # mean + max pooling
        self.classifier = nn.Sequential(
            nn.Linear(readout_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.relu(self.node_proj(x))
        edge_attr = F.relu(self.edge_proj(edge_attr))
        for layer in self.layers:
            x, edge_attr = layer(x, edge_index, edge_attr)
        readout = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1)
        return self.classifier(readout).squeeze(-1)


# ============================================================================
# 3. Scaffold-based partitioning
# ============================================================================

def get_scaffold(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaf) if scaf is not None else smiles
    except Exception:
        return smiles


def scaffold_partition(smiles, indices, n_test, n_val, seed=SEED):
    """Greedy scaffold fill: largest scaffolds assigned to test, then val, then train."""
    rng = np.random.RandomState(seed)
    groups = defaultdict(list)
    for i in indices:
        groups[get_scaffold(smiles[i])].append(i)
    scaffolds = sorted(groups.keys(), key=lambda s: -len(groups[s]))

    # randomise order among equal-size scaffolds for seed-sensitive variety
    if rng.rand() < 1.0:
        import itertools
        chunks = [list(g) for _, g in itertools.groupby(scaffolds, key=lambda s: len(groups[s]))]
        ordered = []
        for chunk in chunks:
            rng.shuffle(chunk)
            ordered.extend(chunk)
        scaffolds = ordered

    test, val, train = [], [], []
    for s in scaffolds:
        ids = groups[s]
        if len(test) < n_test:
            test += ids
        elif len(val) < n_val:
            val += ids
        else:
            train += ids
    return train, val, test


def holdout_split(smiles, n, test_frac=0.2, val_frac=0.1, seed=SEED):
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    train, val, test = scaffold_partition(smiles, list(range(n)), n_test, n_val, seed)
    return train, val, test


def scaffold_kfold(smiles, n_splits=5, seed=SEED):
    """Scaffold-disjoint k-fold CV: each scaffold belongs to exactly one fold."""
    rng = np.random.RandomState(seed)
    groups = defaultdict(list)
    for i, s in enumerate(smiles):
        groups[get_scaffold(s)].append(i)

    # randomise scaffold order, then assign each scaffold to the currently smallest fold
    scaffolds = list(groups.keys())
    rng.shuffle(scaffolds)
    folds = [[] for _ in range(n_splits)]
    sizes = [0] * n_splits
    for s in scaffolds:
        k = int(np.argmin(sizes))
        folds[k].extend(groups[s])
        sizes[k] += len(groups[s])

    splits = []
    n = len(smiles)
    for k in range(n_splits):
        test = folds[k]
        train_val = [i for j in range(n_splits) if j != k for i in folds[j]]
        n_val = int(round(n * 0.1))
        train, val, _ = scaffold_partition(smiles, train_val, 0, n_val, seed=seed + k)
        splits.append((train, val, test))
    return splits


# ============================================================================
# 4. Training & evaluation
# ============================================================================

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total, n = 0.0, 0
    for data in loader:
        optimizer.zero_grad()
        loss = criterion(model(data), data.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * data.num_graphs
        n += data.num_graphs
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    for data in loader:
        preds.extend(torch.sigmoid(model(data)).cpu().numpy().tolist())
        labels.extend(data.y.cpu().numpy().tolist())
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    binary = (preds >= 0.5).astype(int)
    return {
        'AUC-ROC': roc_auc_score(labels, preds),
        'PR-AUC': average_precision_score(labels, preds),
        'Accuracy': accuracy_score(labels, binary),
        'Balanced Accuracy': balanced_accuracy_score(labels, binary),
        'Precision': precision_score(labels, binary, zero_division=0),
        'Recall': recall_score(labels, binary, zero_division=0),
        'F1-Score': f1_score(labels, binary, zero_division=0),
        'MCC': matthews_corrcoef(labels, binary),
    }


def train_and_evaluate(graphs, labels, train_idx, val_idx, test_idx, device,
                       hidden_dim=128, num_layers=3, max_epochs=150, patience=20,
                       batch_size=64, lr=1e-3, fold_seed=SEED):
    torch.manual_seed(fold_seed)
    np.random.seed(fold_seed)

    train_loader = DataLoader([graphs[i] for i in train_idx], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader([graphs[i] for i in val_idx], batch_size=batch_size)
    test_loader = DataLoader([graphs[i] for i in test_idx], batch_size=batch_size)

    node_dim = graphs[0].x.shape[1]
    edge_dim = graphs[0].edge_attr.shape[1]
    model = CMPNN(node_dim, edge_dim, hidden_dim, num_layers, dropout=0.2).to(device)

    train_labels = np.asarray(labels)[train_idx]
    n_pos = int(train_labels.sum())
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_auc, best_state, patience_counter = 0.0, None, 0
    for epoch in range(max_epochs):
        train_epoch(model, train_loader, optimizer, criterion)
        val_metrics = evaluate(model, val_loader)
        if val_metrics['AUC-ROC'] > best_auc:
            best_auc = val_metrics['AUC-ROC']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    model.load_state_dict(best_state)
    return evaluate(model, test_loader), best_auc


# ============================================================================
# 5. Main
# ============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # ---- load data ---------------------------------------------------------
    df = pd.read_excel(DATA_PATH, sheet_name=0)
    print('Raw columns:', list(df.columns), '| rows:', len(df))

    # locate SMILES / class columns robustly
    smi_col = next(c for c in df.columns if 'smiles' in c.lower())
    cls_col = next(c for c in df.columns if c.lower() in ('class', 'label', 'activity'))
    smiles = df[smi_col].astype(str).tolist()

    def to_label(c):
        c = str(c).strip().lower()
        return 1 if c in ('active', 'hit') else 0

    labels = [to_label(c) for c in df[cls_col]]
    n = len(smiles)
    n_active = sum(labels)
    n_inactive = n - n_active
    print(f'Dataset: n={n}  (active={n_active}, inactive={n_inactive}, ratio={n_active/max(n_inactive,1):.1f}:1)')

    # ---- build graphs ------------------------------------------------------
    print('Building molecular graphs...')
    graphs, valid_smiles, valid_labels = [], [], []
    for s, l in zip(smiles, labels):
        g = build_graph(s, l)
        if g is not None:
            graphs.append(g)
            valid_smiles.append(s)
            valid_labels.append(l)
    print(f'  valid graphs: {len(graphs)}/{n}')

    smiles = valid_smiles
    labels = valid_labels
    n = len(smiles)
    n_active = sum(labels)
    n_inactive = n - n_active

    # ---- 70 / 10 / 20 scaffold hold-out ------------------------------------
    train_idx, val_idx, test_idx = holdout_split(smiles, n, 0.2, 0.1, SEED)
    print(f'\nScaffold hold-out 70/10/20: '
          f'train={len(train_idx)} ({len(train_idx)/n:.2f}), '
          f'val={len(val_idx)} ({len(val_idx)/n:.2f}), '
          f'test={len(test_idx)} ({len(test_idx)/n:.2f})')

    holdout_metrics, _ = train_and_evaluate(
        graphs, labels, train_idx, val_idx, test_idx, device,
        hidden_dim=128, num_layers=3, fold_seed=SEED)
    holdout_df = pd.DataFrame([{'Fold': 'Hold-out', **holdout_metrics}])
    holdout_df.to_csv('/mnt/f/Dangruoyu/260727/cmpnn_holdout_test_metrics.csv', index=False)
    print('Hold-out test metrics:', {k: round(v, 4) for k, v in holdout_metrics.items()})

    # ---- scaffold-disjoint 5-fold CV --------------------------------------
    print('\nRunning scaffold-disjoint 5-fold cross-validation...')
    splits = scaffold_kfold(smiles, n_splits=5, seed=SEED)

    fold_rows = []
    accum = {k: [] for k in METRIC_KEYS}
    for fold, (tr, va, te) in enumerate(splits, 1):
        print(f'  Fold {fold}/5: train={len(tr)} val={len(va)} test={len(te)}', flush=True)
        metrics, best_auc = train_and_evaluate(
            graphs, labels, tr, va, te, device,
            hidden_dim=128, num_layers=3, fold_seed=SEED + fold)
        fold_rows.append({'Fold': fold, **{k: round(metrics[k], 4) for k in METRIC_KEYS}})
        for k in METRIC_KEYS:
            accum[k].append(metrics[k])
        print(f'    AUC-ROC={metrics["AUC-ROC"]:.4f}  F1={metrics["F1-Score"]:.4f}  '
              f'MCC={metrics["MCC"]:.4f}', flush=True)

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv('/mnt/f/Dangruoyu/260727/cmpnn_cv_fold_metrics.csv', index=False)

    summary = {'Fold': 'Mean ± SD'}
    for k in METRIC_KEYS:
        summary[k] = f'{np.mean(accum[k]):.4f} ± {np.std(accum[k]):.4f}'
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv('/mnt/f/Dangruoyu/260727/cmpnn_cv_summary.csv', index=False)

    # ---- display -----------------------------------------------------------
    print('\n' + '=' * 100)
    print('CROSS-VALIDATION PERFORMANCE (CMPNN, scaffold-based 5-fold CV)')
    print('=' * 100)
    disp = fold_df.copy()
    disp.loc[len(disp)] = summary
    print(disp.to_string(index=False))
    print('=' * 100)
    print('Saved:')
    print('  - cmpnn_cv_fold_metrics.csv')
    print('  - cmpnn_cv_summary.csv')
    print('  - cmpnn_holdout_test_metrics.csv')


if __name__ == '__main__':
    main()
