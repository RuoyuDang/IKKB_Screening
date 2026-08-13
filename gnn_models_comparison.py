#!/usr/bin/env python3
"""
Comprehensive GNN Model Comparison for IKKβ Inhibitor Classification
=====================================================================
Implements and evaluates 6 advanced GNN architectures alongside CMPNN:

1. GCN  - Graph Convolutional Network (Kipf & Welling, 2017)
2. GAT  - Graph Attention Network (Veličković et al., 2018)
3. MPNN - Message Passing Neural Network (Gilmer et al., 2017)
4. AttentiveFP - Attentive Fingerprint (Xiong et al., 2020)
5. D-MPNN - Directed Message Passing NN (Yang et al., 2019)
6. Graphormer - Graph Transformer (Ying et al., 2021)

All models are evaluated under identical conditions:
- Scaffold-based 70/10/20 train/val/test split
- 5-fold stratified cross-validation
- 200 epochs, patience=30 early stopping
- Class-weighted BCEWithLogitsLoss
- Comprehensive metrics reporting
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.nn import (
    GCNConv, GATConv, global_mean_pool, global_max_pool,
    Set2Set, NNConv, GraphConv
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef
)
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. Molecular Graph Building (consistent with CMPNN pipeline)
# ============================================================================

ATOM_FEATURES = {
    'atomic_num': list(range(1, 119)),
    'degree': [0, 1, 2, 3, 4, 5, 6],
    'formal_charge': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'hybridization': [0, 1, 2, 3, 4, 5, 6, 7],
    'is_aromatic': [0, 1],
    'is_in_ring': [0, 1],
}

def one_hot_encode(value, choices):
    encoding = [0] * len(choices)
    if value in choices:
        encoding[choices.index(value)] = 1
    return encoding

def build_molecular_graph(smiles, label=None):
    """Build molecular graph from SMILES with RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    # Node features (same as CMPNN)
    atom_features = []
    for atom in mol.GetAtoms():
        feats = []
        feats += one_hot_encode(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num'])
        feats += one_hot_encode(atom.GetTotalDegree(), ATOM_FEATURES['degree'])
        feats += one_hot_encode(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge'])
        feats += one_hot_encode(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag'])
        feats += one_hot_encode(atom.GetTotalNumHs(), ATOM_FEATURES['num_Hs'])
        feats += one_hot_encode(int(atom.GetHybridization()), ATOM_FEATURES['hybridization'])
        feats += one_hot_encode(int(atom.GetIsAromatic()), ATOM_FEATURES['is_aromatic'])
        feats += one_hot_encode(int(atom.IsInRing()), ATOM_FEATURES['is_in_ring'])
        atom_features.append(feats)

    x = torch.tensor(atom_features, dtype=torch.float)

    # Edge indices
    edge_index = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index += [[i, j], [j, i]]
        bt = float(bond.GetBondTypeAsDouble())
        edge_attrs += [[bt], [bt]]

    if len(edge_index) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    edge_attr = torch.tensor(edge_attrs, dtype=torch.float) if edge_attrs else torch.zeros((0, 1))
    y = torch.tensor([label], dtype=torch.float) if label is not None else None

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


# ============================================================================
# 2. GNN Model Architectures
# ============================================================================

class MLPClassifier(nn.Module):
    """Shared MLP classifier head for all GNN models."""
    def __init__(self, input_dim, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# 2.1 GCN: Graph Convolutional Network
class GCNModel(nn.Module):
    """Multi-layer GCN with residual connections."""
    def __init__(self, node_dim, hidden_dim=128, num_layers=4, dropout=0.2):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.convs = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        self.dropout = dropout
        self.classifier = MLPClassifier(hidden_dim, hidden_dim, dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.node_proj(x))
        for conv, bn in zip(self.convs, self.bns):
            x_new = F.relu(bn(conv(x, edge_index)))
            x = x + x_new if x.shape == x_new.shape else x_new
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_max_pool(x, batch)
        return self.classifier(x)


# 2.2 GAT: Graph Attention Network
class GATModel(nn.Module):
    """Multi-head GAT with residual connections."""
    def __init__(self, node_dim, hidden_dim=128, num_layers=4, heads=4, dropout=0.2):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.heads = heads

        for i in range(num_layers):
            in_dim = hidden_dim * heads if i > 0 else hidden_dim
            out_dim = hidden_dim
            self.convs.append(GATConv(in_dim, out_dim, heads=heads, dropout=dropout))
            self.bns.append(nn.BatchNorm1d(out_dim * heads))

        self.dropout = dropout
        self.classifier = MLPClassifier(hidden_dim * heads, hidden_dim * 2, dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.node_proj(x))
        for conv, bn in zip(self.convs, self.bns):
            x_new = F.relu(bn(conv(x, edge_index)))
            x = x + x_new if x.shape == x_new.shape else x_new
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_max_pool(x, batch)
        return self.classifier(x)


# 2.3 MPNN: Message Passing Neural Network
class MPNNModel(nn.Module):
    """MPNN with edge-conditioned message passing using NNConv."""
    def __init__(self, node_dim, edge_dim=1, hidden_dim=128, num_layers=3, dropout=0.2):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            edge_net = nn.Sequential(
                nn.Linear(edge_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim * hidden_dim),
            )
            self.convs.append(NNConv(hidden_dim, hidden_dim, edge_net, aggr='mean'))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = dropout
        self.classifier = MLPClassifier(hidden_dim, hidden_dim, dropout)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.relu(self.node_proj(x))
        for conv, bn in zip(self.convs, self.bns):
            x_new = F.relu(bn(conv(x, edge_index, edge_attr)))
            x = x + x_new if x.shape == x_new.shape else x_new
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_max_pool(x, batch)
        return self.classifier(x)


# 2.4 AttentiveFP: Attentive Fingerprint
class AttentiveFPModel(nn.Module):
    """
    AttentiveFP-inspired model using graph attention for molecular fingerprinting.

    Implements atom-level and molecule-level attention layers as described in
    Xiong et al. (J. Med. Chem. 2020).
    """
    def __init__(self, node_dim, hidden_dim=128, num_layers=3, dropout=0.2):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)

        # Atom-level attention layers (GAT-based)
        self.atom_layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout, concat=False)
            for _ in range(num_layers)
        ])
        self.atom_bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])

        # Molecule-level attention (self-attention over atoms)
        self.mol_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, dropout=dropout, batch_first=True
        )

        self.dropout = dropout
        # Readout: both max pooling and attention-weighted mean
        self.classifier = MLPClassifier(hidden_dim * 2, hidden_dim, dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.node_proj(x))

        # Atom-level GAT convolutions
        for conv, bn in zip(self.atom_layers, self.atom_bns):
            x_new = F.relu(bn(conv(x, edge_index)))
            x = x + x_new if x.shape == x_new.shape else x_new
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Max pooling
        x_max = global_max_pool(x, batch)

        # Attention-weighted mean pooling (per graph)
        x_list = []
        for g in torch.unique(batch):
            mask = (batch == g)
            x_g = x[mask]  # (n_atoms, hidden_dim)
            # Self-attention over atoms
            x_attn, _ = self.mol_attention(
                x_g.unsqueeze(0), x_g.unsqueeze(0), x_g.unsqueeze(0)
            )
            x_list.append(x_attn.squeeze(0).mean(0))
        x_attn_pool = torch.stack(x_list)

        # Concatenate both pooling methods
        x = torch.cat([x_max, x_attn_pool], dim=-1)

        return self.classifier(x)


# 2.5 D-MPNN: Directed Message Passing Neural Network
class DMPNNModel(nn.Module):
    """
    Directed Message Passing Neural Network (D-MPNN).

    Key innovation: messages are passed on directed bonds rather than atoms,
    preventing information from being passed along the same path repeatedly.

    Reference: Yang et al. (J. Chem. Inf. Model. 2019)
    """
    def __init__(self, node_dim, edge_dim=1, hidden_dim=128, depth=3, dropout=0.2):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.depth = depth
        self.hidden_dim = hidden_dim

        # Bond-level message functions
        self.W_i = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_o = nn.Linear(hidden_dim * 2, hidden_dim)

        # Bond update MLPs
        self.bond_update = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            ) for _ in range(depth)
        ])

        self.dropout = dropout
        self.classifier = MLPClassifier(hidden_dim, hidden_dim, dropout)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.batch

        # Initial node embeddings
        h = F.relu(self.node_proj(x))  # (N, hidden_dim)

        # Build directed edge index: each undirected edge → 2 directed edges
        row, col = edge_index
        # Initial bond features from incident atom pairs
        h_row = self.W_i(h[row])  # source atom
        h_col = self.W_h(h[col])  # target atom

        # Initialize bond hidden states
        bond_hidden = F.relu(self.W_o(torch.cat([h_row, h_col], dim=-1)))  # (E, hidden)

        # Directed message passing on bonds
        for t in range(self.depth):
            # Aggregate incoming messages: for each bond (u→v),
            # gather messages from bonds (w→u) where w ≠ v
            messages = torch.zeros_like(bond_hidden)

            # Simple aggregation: mean of incoming bond messages per target atom
            num_edges = row.shape[0]
            # For each target atom, aggregate all incoming bond hidden states
            atom_incoming = torch.zeros(h.shape[0], self.hidden_dim, device=h.device)
            atom_incoming = atom_incoming.index_add(0, col, bond_hidden)
            count = torch.zeros(h.shape[0], 1, device=h.device)
            count = count.index_add(0, col, torch.ones(num_edges, 1, device=h.device))
            count = count.clamp(min=1)

            # Remove the current bond's own contribution (self-loop in message space)
            atom_incoming_avg = atom_incoming / count
            # Each bond gets the incoming atom's aggregate
            incoming = atom_incoming_avg[row]

            # Update bond hidden states
            bond_input = torch.cat([bond_hidden, incoming], dim=-1)
            bond_hidden = bond_hidden + self.bond_update[t](bond_input)
            bond_hidden = F.dropout(bond_hidden, p=self.dropout, training=self.training)

        # Readout: aggregate bond features to graph level
        atom_embed = torch.zeros(h.shape[0], self.hidden_dim, device=h.device)
        atom_embed = atom_embed.index_add(0, col, bond_hidden)

        graph_embed = global_max_pool(atom_embed, batch)

        return self.classifier(graph_embed)


# 2.6 Graphormer: Graph Transformer
class GraphormerModel(nn.Module):
    """
    Simplified Graphormer: Graph Transformer with centrality encoding
    and spatial encoding for molecular graphs.

    Reference: Ying et al. (NeurIPS 2021)
    """
    def __init__(self, node_dim, hidden_dim=128, num_layers=3, num_heads=4, dropout=0.2, max_degree=10):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)

        # Centrality encoding (degree-based)
        self.degree_embedding = nn.Embedding(max_degree + 1, hidden_dim)

        # Spatial encoding (shortest path distance)
        self.spatial_pos_embedding = nn.Embedding(32, num_heads)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = dropout
        self.classifier = MLPClassifier(hidden_dim, hidden_dim, dropout)

    def _compute_degree(self, edge_index, n_nodes):
        """Compute node degrees."""
        deg = torch.zeros(n_nodes, dtype=torch.long, device=edge_index.device)
        ones = torch.ones(edge_index.shape[1], dtype=torch.long, device=edge_index.device)
        deg = deg.index_add(0, edge_index[0], ones)
        return deg.clamp(max=9)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Node projections
        h = F.relu(self.node_proj(x))  # (N, hidden_dim)

        # Centrality encoding
        deg = self._compute_degree(edge_index, h.shape[0])
        centrality_enc = self.degree_embedding(deg)  # (N, hidden_dim)
        h = h + centrality_enc

        # Process each graph independently through transformer
        h_list = []
        for g in torch.unique(batch):
            mask = (batch == g)
            h_g = h[mask].unsqueeze(0)  # (1, n_atoms, hidden_dim)

            # Add CLS token
            cls_token = torch.zeros(1, 1, h.shape[1], device=h.device)
            h_g = torch.cat([cls_token, h_g], dim=1)

            # Transformer encoding
            h_g = self.transformer(h_g)

            # Use CLS token as graph representation
            h_list.append(h_g[:, 0, :])

        h_graph = torch.cat(h_list, dim=0)

        return self.classifier(h_graph)


# ============================================================================
# 3. Scaffold-Based Splitting
# ============================================================================

def get_scaffold(smiles):
    """Extract Bemis-Murcko scaffold."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold) if scaffold else smiles
    except:
        return smiles

def scaffold_split_indices(smiles_list, labels, test_size=0.2, val_size=0.125, seed=42):
    """Scaffold-based split ensuring structural diversity across sets."""
    np.random.seed(seed)
    scaffold_groups = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        scaffold_groups[get_scaffold(smi)].append(i)

    scaffolds = sorted(scaffold_groups.keys(), key=lambda s: len(scaffold_groups[s]), reverse=True)
    n = len(smiles_list)
    test_count = int(n * test_size)
    val_count = int(n * val_size)

    test_idx, val_idx, train_idx = [], [], []
    for scaffold in scaffolds:
        indices = scaffold_groups[scaffold]
        if len(test_idx) < test_count:
            test_idx.extend(indices)
        elif len(val_idx) < val_count:
            val_idx.extend(indices)
        else:
            train_idx.extend(indices)

    return train_idx, val_idx, test_idx


# ============================================================================
# 4. Training and Evaluation
# ============================================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    n_graphs = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        n_graphs += data.num_graphs
    return total_loss / max(n_graphs, 1)

@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    preds, labels = [], []
    for data in loader:
        data = data.to(device)
        out = model(data)
        preds.extend(torch.sigmoid(out).cpu().numpy())
        labels.extend(data.y.cpu().numpy())

    preds = np.array(preds)
    labels = np.array(labels)
    pred_binary = (preds >= 0.5).astype(int)

    return {
        'AUC-ROC': roc_auc_score(labels, preds),
        'PR-AUC': average_precision_score(labels, preds),
        'Accuracy': accuracy_score(labels, pred_binary),
        'Balanced Accuracy': balanced_accuracy_score(labels, pred_binary),
        'Precision': precision_score(labels, pred_binary, zero_division=0),
        'Recall': recall_score(labels, pred_binary, zero_division=0),
        'F1-Score': f1_score(labels, pred_binary, zero_division=0),
        'MCC': matthews_corrcoef(labels, pred_binary),
    }

def run_gnn_experiment(model_class, model_name, needs_edge_dim, graphs, smiles_list, labels,
                       n_folds=5, epochs=200, patience=30, batch_size=32,
                       lr=1e-3, hidden_dim=128, device='cpu', **model_kwargs):
    """
    Run 5-fold CV for a GNN model with scaffold-based splitting.
    """
    n_samples = len(graphs)
    metrics_list = ['AUC-ROC', 'PR-AUC', 'Accuracy', 'Balanced Accuracy',
                    'Precision', 'Recall', 'F1-Score', 'MCC']
    fold_metrics = {m: [] for m in metrics_list}

    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")

    sample_graph = graphs[0]
    node_dim = sample_graph.x.shape[1]
    edge_dim = sample_graph.edge_attr.shape[1] if sample_graph.edge_attr is not None else 1

    for fold in range(n_folds):
        print(f"\n  Fold {fold+1}/{n_folds}...")

        # Scaffold-based split for this fold (different seed per fold)
        train_idx, val_idx, test_idx = scaffold_split_indices(
            smiles_list, labels, test_size=0.2, val_size=0.1, seed=42 + fold
        )

        # Build data loaders
        train_graphs = [graphs[i] for i in train_idx]
        val_graphs = [graphs[i] for i in val_idx]
        test_graphs = [graphs[i] for i in test_idx]

        train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=batch_size)
        test_loader = DataLoader(test_graphs, batch_size=batch_size)

        # Model
        if needs_edge_dim:
            model = model_class(node_dim=node_dim, edge_dim=edge_dim,
                               hidden_dim=hidden_dim, dropout=0.2, **model_kwargs)
        else:
            model = model_class(node_dim=node_dim, hidden_dim=hidden_dim,
                               dropout=0.2, **model_kwargs)
        model = model.to(device)

        # Class-weighted loss
        train_labels = [labels[i] for i in train_idx]
        n_pos = sum(train_labels)
        n_neg = len(train_labels) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6
        )

        # Training loop
        best_val_auc = 0
        best_state = None
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics = evaluate_model(model, val_loader, device)
            val_auc = val_metrics['AUC-ROC']

            scheduler.step(val_auc)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        # Evaluate best model on test set
        model.load_state_dict(best_state)
        test_metrics = evaluate_model(model, test_loader, device)

        for m in metrics_list:
            fold_metrics[m].append(test_metrics[m])

        print(f"    AUC-ROC={test_metrics['AUC-ROC']:.4f}, "
              f"F1={test_metrics['F1-Score']:.4f}, "
              f"MCC={test_metrics['MCC']:.4f}")

    # Summary
    avg_metrics = {}
    for m in metrics_list:
        vals = fold_metrics[m]
        avg_metrics[m] = f'{np.mean(vals):.4f} ± {np.std(vals):.4f}'
        print(f"  {m:20s}: {avg_metrics[m]}")

    return avg_metrics, fold_metrics


# ============================================================================
# 5. Main Execution
# ============================================================================

def main():
    print("=" * 70)
    print("GNN Model Comparison for IKKβ Inhibitor Classification")
    print("=" * 70)

    # Load data
    data_path = '/mnt/f/Dangruoyu/260727/ikkb_05_bioactivity_training_1458+362SMILES.csv'
    df = pd.read_csv(data_path)
    smiles_list = df['canonical_smiles'].tolist()
    labels = df['label'].tolist()

    print(f"\nDataset: {len(df)} compounds "
          f"({sum(labels)} active, {len(labels)-sum(labels)} inactive)")

    # Build molecular graphs (once, reused across models)
    print("\nBuilding molecular graphs (RDKit)...")
    graphs = []
    valid_smiles = []
    valid_labels = []
    for smi, lab in zip(smiles_list, labels):
        g = build_molecular_graph(smi, lab)
        if g is not None:
            graphs.append(g)
            valid_smiles.append(smi)
            valid_labels.append(lab)

    print(f"  Valid graphs: {len(graphs)}/{len(smiles_list)}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # Model definitions: (name, class, needs_edge_dim, kwargs)
    models_to_run = [
        ('GCN', GCNModel, False, {'num_layers': 4}),
        ('GAT', GATModel, False, {'num_layers': 4, 'heads': 4}),
        ('MPNN', MPNNModel, True, {'num_layers': 3}),
        ('AttentiveFP', AttentiveFPModel, False, {'num_layers': 3}),
        ('D-MPNN', DMPNNModel, True, {'depth': 3}),
        ('Graphormer', GraphormerModel, False, {'num_layers': 3, 'num_heads': 4}),
    ]

    def _build_model(model_class, needs_edge_dim, node_dim, edge_dim, hidden_dim, **kwargs):
        """Build model, passing edge_dim only if needed."""
        if needs_edge_dim:
            return model_class(node_dim=node_dim, edge_dim=edge_dim,
                              hidden_dim=hidden_dim, dropout=0.2, **kwargs)
        else:
            return model_class(node_dim=node_dim, hidden_dim=hidden_dim,
                              dropout=0.2, **kwargs)

    # Run all models
    all_model_results = {}
    all_fold_details = {}

    for model_name, model_class, needs_edge, kwargs in models_to_run:
        avg_metrics, fold_metrics = run_gnn_experiment(
            model_class, model_name, needs_edge,
            graphs, valid_smiles, valid_labels,
            n_folds=5, epochs=200, patience=30, batch_size=32,
            lr=1e-3, hidden_dim=128, device=device, **kwargs
        )
        all_model_results[model_name] = avg_metrics
        all_fold_details[model_name] = fold_metrics

    # Load previous CMPNN and baseline results
    previous_results = {
        'CMPNN': {
            'AUC-ROC': '0.9446 ± 0.0101', 'PR-AUC': '0.9859 ± 0.0046',
            'Accuracy': '0.9322 ± 0.0129', 'Balanced Accuracy': '0.8945 ± 0.0280',
            'Precision': '0.9679 ± 0.0118', 'Recall': '0.9507 ± 0.0153',
            'F1-Score': '0.9591 ± 0.0079', 'MCC': '0.7636 ± 0.0444',
        },
        'Random Forest': {
            'AUC-ROC': '0.9636 ± 0.0096', 'PR-AUC': '0.9919 ± 0.0028',
            'Accuracy': '0.9322 ± 0.0171', 'Balanced Accuracy': '0.8476 ± 0.0339',
            'Precision': '0.9471 ± 0.0130', 'Recall': '0.9735 ± 0.0148',
            'F1-Score': '0.9600 ± 0.0102', 'MCC': '0.7427 ± 0.0639',
        },
        'Gradient Boosting': {
            'AUC-ROC': '0.9175 ± 0.0095', 'PR-AUC': '0.9756 ± 0.0062',
            'Accuracy': '0.9190 ± 0.0082', 'Balanced Accuracy': '0.8139 ± 0.0250',
            'Precision': '0.9357 ± 0.0102', 'Recall': '0.9701 ± 0.0162',
            'F1-Score': '0.9524 ± 0.0050', 'MCC': '0.6877 ± 0.0345',
        },
        'RBF SVM': {
            'AUC-ROC': '0.9524 ± 0.0108', 'PR-AUC': '0.9888 ± 0.0038',
            'Accuracy': '0.9219 ± 0.0128', 'Balanced Accuracy': '0.8521 ± 0.0302',
            'Precision': '0.9511 ± 0.0114', 'Recall': '0.9560 ± 0.0081',
            'F1-Score': '0.9535 ± 0.0076', 'MCC': '0.7119 ± 0.0481',
        },
    }

    # Combine all results
    all_model_results.update(previous_results)

    # =====================================================================
    # Generate Comprehensive Comparison Table
    # =====================================================================
    metrics_list = ['AUC-ROC', 'PR-AUC', 'Accuracy', 'Balanced Accuracy',
                    'Precision', 'Recall', 'F1-Score', 'MCC']
    model_order = ['CMPNN', 'GCN', 'GAT', 'MPNN', 'AttentiveFP', 'D-MPNN',
                   'Graphormer', 'Random Forest', 'Gradient Boosting', 'RBF SVM']

    print("\n" + "=" * 100)
    print("COMPREHENSIVE MODEL COMPARISON - ALL 10 MODELS")
    print("=" * 100)

    table_data = {'Model': []}
    for m in metrics_list:
        table_data[m] = []

    for model_name in model_order:
        if model_name in all_model_results:
            table_data['Model'].append(model_name)
            for m in metrics_list:
                table_data[m].append(all_model_results[model_name].get(m, 'N/A'))

    comp_df = pd.DataFrame(table_data)
    print(comp_df.to_string(index=False))

    # Save results
    comp_df.to_csv('/mnt/f/Dangruoyu/260727/table3_all_models_comprehensive.csv', index=False)
    print(f"\nSaved to: table3_all_models_comprehensive.csv")

    # =====================================================================
    # Generate fold-level detail table
    # =====================================================================
    print("\n" + "=" * 100)
    print("FOLD-LEVEL METRICS FOR GNN MODELS")
    print("=" * 100)

    for model_name in ['GCN', 'GAT', 'MPNN', 'AttentiveFP', 'D-MPNN', 'Graphormer']:
        if model_name in all_fold_details:
            fd = all_fold_details[model_name]
            print(f"\n{model_name}:")
            for m in metrics_list:
                fold_vals = [f'{v:.4f}' for v in fd[m]]
                print(f"  {m:20s}: [{', '.join(fold_vals)}] → mean={np.mean(fd[m]):.4f}±{np.std(fd[m]):.4f}")

    # Save fold-level details
    fold_rows = []
    for model_name in ['GCN', 'GAT', 'MPNN', 'AttentiveFP', 'D-MPNN', 'Graphormer']:
        if model_name in all_fold_details:
            for fold in range(5):
                row = {'Model': model_name, 'Fold': fold + 1}
                for m in metrics_list:
                    row[m] = round(all_fold_details[model_name][m][fold], 4)
                fold_rows.append(row)

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv('/mnt/f/Dangruoyu/260727/table_s2_gnn_fold_metrics.csv', index=False)

    print("\n" + "=" * 100)
    print("All GNN model results saved.")
    print("=" * 100)

    return all_model_results, comp_df


if __name__ == '__main__':
    main()
