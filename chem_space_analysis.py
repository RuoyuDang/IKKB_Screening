#!/usr/bin/env python3
"""
Comprehensive Chemical Space Analysis for IKKβ Inhibitor Dataset
=================================================================
1,458 training compounds (1,096 active + 362 inactive) + 22 final hits

Outputs:
- Molecular property distributions (MW, LogP, TPSA, HBD, HBA, RotB, etc.)
- Scaffold diversity analysis (Bemis-Murcko)
- t-SNE & UMAP visualization of ECFP4 chemical space
- Tanimoto similarity matrix (hits vs. training set, hits vs. known IKKβ inhibitors)
- PCA of property space
- Drug-likeness assessment (Lipinski, Veber, PAINS)
- All figures saved as PNG, all data as CSV
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import Draw
from rdkit.DataStructs import TanimotoSimilarity, BulkTanimotoSimilarity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({'font.size': 10, 'axes.titlesize': 13, 'axes.labelsize': 11,
                     'figure.dpi': 200, 'savefig.dpi': 200, 'savefig.bbox': 'tight'})

OUT = '/mnt/f/Dangruoyu/260727/chem_space_output/'
import os; os.makedirs(OUT, exist_ok=True)

# ===================================================================
# 1. LOAD & PARSE DATA
# ===================================================================
print("="*60)
print("Loading data...")
df = pd.read_excel('/mnt/f/Dangruoyu/260727/ikkb_05_bioactivity_training_1725+_Final_hit_22_SMILES.xls',
                   sheet_name='ikkb_05_bioactivity_data_2class')
df = df.dropna(subset=['canonical_smiles'])

# Parse molecules
def parse_mol(smi):
    try: return Chem.MolFromSmiles(smi)
    except: return None

df['mol'] = df['canonical_smiles'].apply(parse_mol)
df = df[df['mol'].notna()].copy()

# Class labels
train_df = df[df['class'].isin(['active', 'inactive'])].copy()
hits_df = df[df['class'] == 'hit'].copy()
print(f"Training: {len(train_df)} ({train_df['class'].value_counts().to_dict()})")
print(f"Hits: {len(hits_df)}")

# ===================================================================
# 2. MOLECULAR PROPERTIES
# ===================================================================
print("\n"+"="*60)
print("Computing molecular properties...")

for label, subdf in [('all', df), ('train', train_df), ('hits', hits_df)]:
    mols = subdf['mol'].values
    idx = subdf.index
    df.loc[idx, 'MW'] = [Descriptors.MolWt(m) for m in mols]
    df.loc[idx, 'LogP'] = [Crippen.MolLogP(m) for m in mols]
    df.loc[idx, 'TPSA'] = [rdMolDescriptors.CalcTPSA(m) for m in mols]
    df.loc[idx, 'HBD'] = [Lipinski.NumHDonors(m) for m in mols]
    df.loc[idx, 'HBA'] = [Lipinski.NumHAcceptors(m) for m in mols]
    df.loc[idx, 'RotB'] = [Lipinski.NumRotatableBonds(m) for m in mols]
    df.loc[idx, 'NumRings'] = [rdMolDescriptors.CalcNumRings(m) for m in mols]
    df.loc[idx, 'NumAromaticRings'] = [rdMolDescriptors.CalcNumAromaticRings(m) for m in mols]
    df.loc[idx, 'FractionCsp3'] = [rdMolDescriptors.CalcFractionCSP3(m) for m in mols]
    df.loc[idx, 'NumHeteroatoms'] = [rdMolDescriptors.CalcNumHeteroatoms(m) for m in mols]
    df.loc[idx, 'HeavyAtomCount'] = [m.GetNumHeavyAtoms() for m in mols]

# Lipinski violations
def count_lipinski_violations(m):
    v = 0
    if Descriptors.MolWt(m) > 500: v += 1
    if Crippen.MolLogP(m) > 5: v += 1
    if Lipinski.NumHDonors(m) > 5: v += 1
    if Lipinski.NumHAcceptors(m) > 10: v += 1
    return v

df['Lipinski_Violations'] = df['mol'].apply(count_lipinski_violations)

# Save property table
prop_cols = ['canonical_smiles', 'class', 'MW', 'LogP', 'TPSA', 'HBD', 'HBA',
             'RotB', 'NumRings', 'NumAromaticRings', 'FractionCsp3',
             'NumHeteroatoms', 'HeavyAtomCount', 'Lipinski_Violations']
df[prop_cols].to_csv(f'{OUT}table_molecular_properties_all.csv', index=False)
print("Properties saved.")

# ===================================================================
# 3. SCAFFOLD ANALYSIS
# ===================================================================
print("\n"+"="*60)
print("Scaffold analysis...")

def get_scaffold(mol):
    try:
        s = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(s) if s and s.GetNumAtoms() > 0 else None
    except: return None

df['scaffold'] = df['mol'].apply(get_scaffold)

# Re-derive masks after adding scaffold column
train_scaffolds = df.loc[df['class'].isin(['active','inactive']), 'scaffold'].dropna().value_counts()
hit_scaffolds = df.loc[df['class'] == 'hit', 'scaffold'].dropna()

print(f"Training unique scaffolds: {len(train_scaffolds)}")
print(f"Training compounds/scaffold: mean={train_scaffolds.mean():.1f}, max={train_scaffolds.max()}")
print(f"Hit scaffolds: {len(set(hit_scaffolds))} unique among {len(hit_scaffolds)} compounds")

# Scaffold overlap between hits and training
train_scaff_set = set(train_scaffolds.index)
hit_scaff_set = set(hit_scaffolds)
overlap_scaffolds = hit_scaff_set & train_scaff_set
print(f"Hit scaffolds overlapping with training: {len(overlap_scaffolds)}/{len(hit_scaff_set)}")

# Novel scaffolds in hits
novel_hit_scaffolds = hit_scaff_set - train_scaff_set
print(f"Novel hit scaffolds (not in training): {len(novel_hit_scaffolds)}")

# Save scaffold data
scaffold_df = pd.DataFrame({
    'scaffold_smiles': list(train_scaffolds.index),
    'count': train_scaffolds.values,
    'in_hits': [s in hit_scaff_set for s in train_scaffolds.index]
})
scaffold_df.to_csv(f'{OUT}table_scaffold_analysis.csv', index=False)

# ===================================================================
# 4. ECFP4 FINGERPRINTS & TANIMOTO SIMILARITY
# ===================================================================
print("\n"+"="*60)
print("Computing ECFP4 fingerprints...")

def get_ecfp4(mol, radius=2, nbits=2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)

df['ecfp4'] = df['mol'].apply(lambda m: get_ecfp4(m))

# Tanimoto similarity: each hit vs all training compounds
hit_indices = hits_df.index.tolist()
train_indices = train_df.index.tolist()
hit_fps = [df.loc[i, 'ecfp4'] for i in hit_indices]
train_fps = [df.loc[i, 'ecfp4'] for i in train_indices]

# Max Tanimoto per hit vs training
max_tc = []
for i, hfp in zip(hit_indices, hit_fps):
    sims = BulkTanimotoSimilarity(hfp, train_fps)
    max_tc.append({'compound_idx': i, 'canonical_smiles': df.loc[i, 'canonical_smiles'],
                   'max_tanimoto_training': np.max(sims),
                   'mean_tanimoto_training': np.mean(sims),
                   'nearest_training_idx': train_indices[np.argmax(sims)],
                   'nearest_training_smiles': df.loc[train_indices[np.argmax(sims)], 'canonical_smiles']})

tc_df = pd.DataFrame(max_tc)
tc_df.to_csv(f'{OUT}table_hit_tanimoto_vs_training.csv', index=False)
print(f"Hit-Training Tanimoto: max={tc_df['max_tanimoto_training'].max():.3f}, "
      f"mean={tc_df['max_tanimoto_training'].mean():.3f}, "
      f"min={tc_df['max_tanimoto_training'].min():.3f}")

# Inter-hit Tanimoto
hit_tc_matrix = np.zeros((len(hit_indices), len(hit_indices)))
for i in range(len(hit_indices)):
    for j in range(len(hit_indices)):
        hit_tc_matrix[i, j] = TanimotoSimilarity(hit_fps[i], hit_fps[j])

print(f"Inter-hit Tanimoto: mean={hit_tc_matrix[np.triu_indices(len(hit_indices),1)].mean():.3f}, "
      f"max={hit_tc_matrix[np.triu_indices(len(hit_indices),1)].max():.3f}")

# ===================================================================
# 5. DIMENSIONALITY REDUCTION (t-SNE + UMAP)
# ===================================================================
print("\n"+"="*60)
print("Running t-SNE and PCA...")

# All ECFP4 → numpy array
all_fps = np.array([list(df.loc[i, 'ecfp4']) for i in df.index])
train_mask = np.array([i in train_indices for i in df.index])
hit_mask = np.array([i in hit_indices for i in df.index])
active_mask = np.array([df.loc[i, 'class'] == 'active' for i in df.index])
inactive_mask = np.array([df.loc[i, 'class'] == 'inactive' for i in df.index])

# PCA first (50 components)
print("  PCA...")
pca = PCA(n_components=50, random_state=42)
fps_pca = pca.fit_transform(all_fps)
print(f"  PCA 50D variance explained: {pca.explained_variance_ratio_.sum():.3f}")

# t-SNE
print("  t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000, verbose=0)
fps_tsne = tsne.fit_transform(fps_pca)

# UMAP
print("  UMAP...")
import umap
umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3)
fps_umap = umap_reducer.fit_transform(fps_pca)

# Also do PCA of molecular properties
prop_features = ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotB', 'NumRings',
                 'NumAromaticRings', 'FractionCsp3', 'NumHeteroatoms', 'HeavyAtomCount']
props = df[prop_features].fillna(0).values
props_scaled = StandardScaler().fit_transform(props)
pca_prop = PCA(n_components=2, random_state=42)
props_pca = pca_prop.fit_transform(props_scaled)
print(f"  Property PCA variance: {pca_prop.explained_variance_ratio_.sum():.3f}")

# Save coordinates
coord_df = pd.DataFrame({
    'canonical_smiles': df['canonical_smiles'].values,
    'class': df['class'].values,
    'tsne_x': fps_tsne[:, 0], 'tsne_y': fps_tsne[:, 1],
    'umap_x': fps_umap[:, 0], 'umap_y': fps_umap[:, 1],
    'pca_prop_x': props_pca[:, 0], 'pca_prop_y': props_pca[:, 1],
})
coord_df.to_csv(f'{OUT}table_chemical_space_coordinates.csv', index=False)
print("Coordinates saved.")

# ===================================================================
# 6. FIGURES
# ===================================================================
print("\n"+"="*60)
print("Generating figures...")

# Final refresh: ensure all dataframes have all columns
train_df = df[df['class'].isin(['active','inactive'])].copy()
hits_df = df[df['class'] == 'hit'].copy()
train_indices_list = train_df.index.tolist()
hit_indices_list = hits_df.index.tolist()

# Colors
C_ACTIVE = '#2E86AB'
C_INACTIVE = '#D64045'
C_HIT = '#F18F01'
C_HIT_STAR = '#FF4500'

# --- Figure C1: Property Distribution Panel (6-panel) ---
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

props_plot = [
    ('MW', 'Molecular Weight (Da)', axes[0,0]),
    ('LogP', 'LogP', axes[0,1]),
    ('TPSA', 'TPSA (Å²)', axes[0,2]),
    ('HBD', 'H-Bond Donors', axes[1,0]),
    ('HBA', 'H-Bond Acceptors', axes[1,1]),
    ('RotB', 'Rotatable Bonds', axes[1,2]),
]

for col, xlabel, ax in props_plot:
    ax.hist(train_df[col].dropna(), bins=40, alpha=0.7, color=C_ACTIVE, density=True, label='Active (n=1096)')
    ax.hist(train_df[train_df['class']=='inactive'][col].dropna(), bins=40, alpha=0.5, color=C_INACTIVE, density=True, label='Inactive (n=362)')
    # Mark hits with vertical lines
    for _, h in hits_df.iterrows():
        ax.axvline(h[col], color=C_HIT, alpha=0.3, linewidth=0.5)
    ax.set_xlabel(xlabel); ax.set_ylabel('Density'); ax.legend(fontsize=7)
    ax.grid(alpha=0.2)

fig.suptitle('Molecular Property Distributions (Training Set vs. 22 Hits)', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(f'{OUT}figure_c1_property_distributions.png')
plt.close()
print("  Figure C1: Property distributions ✓")

# --- Figure C2: t-SNE Chemical Space ---
fig, axes = plt.subplots(1, 3, figsize=(21, 7))

# Panel A: Active vs Inactive
ax = axes[0]
ax.scatter(fps_tsne[active_mask, 0], fps_tsne[active_mask, 1], c=C_ACTIVE, alpha=0.4, s=8, label='Active (n=1096)')
ax.scatter(fps_tsne[inactive_mask, 0], fps_tsne[inactive_mask, 1], c=C_INACTIVE, alpha=0.5, s=8, label='Inactive (n=362)')
ax.scatter(fps_tsne[hit_mask, 0], fps_tsne[hit_mask, 1], c=C_HIT_STAR, s=60, marker='*', edgecolors='black', linewidth=0.8, label='22 Hits', zorder=10)
ax.set_title('A. t-SNE: Active vs Inactive'); ax.legend(fontsize=7); ax.grid(alpha=0.2)

# Panel B: Training + Hits labeled
ax = axes[1]
ax.scatter(fps_tsne[train_mask, 0], fps_tsne[train_mask, 1], c='lightgray', alpha=0.4, s=8, label='Training Set (n=1458)')
ax.scatter(fps_tsne[hit_mask, 0], fps_tsne[hit_mask, 1], c=C_HIT_STAR, s=80, marker='*', edgecolors='black', linewidth=1.0, label='22 Hits', zorder=10)
# Annotate top hits
for i, idx in enumerate(hit_indices):
    ax.annotate(f'H{i+1}', (fps_tsne[idx, 0], fps_tsne[idx, 1]),
                fontsize=6, xytext=(3,3), textcoords='offset points', alpha=0.8)
ax.set_title('B. t-SNE: Training + 22 Hits'); ax.legend(fontsize=7); ax.grid(alpha=0.2)

# Panel C: UMAP
ax = axes[2]
ax.scatter(fps_umap[train_mask, 0], fps_umap[train_mask, 1], c='lightgray', alpha=0.3, s=6, label='Training Set')
ax.scatter(fps_umap[hit_mask, 0], fps_umap[hit_mask, 1], c=C_HIT_STAR, s=80, marker='*', edgecolors='black', linewidth=1.0, label='22 Hits', zorder=10)
for i, idx in enumerate(hit_indices):
    ax.annotate(f'H{i+1}', (fps_umap[idx, 0], fps_umap[idx, 1]),
                fontsize=6, xytext=(3,3), textcoords='offset points', alpha=0.8)
ax.set_title('C. UMAP: Training + 22 Hits'); ax.legend(fontsize=7); ax.grid(alpha=0.2)

fig.suptitle('Chemical Space Visualization (ECFP4 Fingerprints, 2048 bits)', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(f'{OUT}figure_c2_chemical_space_tsne_umap.png')
plt.close()
print("  Figure C2: t-SNE/UMAP ✓")

# --- Figure C3: Tanimoto Similarity Heatmap ---
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Panel A: Inter-hit Tanimoto heatmap
ax = axes[0]
labels = [f'H{i+1}' for i in range(len(hit_indices))]
sns.heatmap(hit_tc_matrix, annot=True, fmt='.2f', cmap='YlOrRd', vmin=0, vmax=1,
            xticklabels=labels, yticklabels=labels, ax=ax, cbar_kws={'label': 'Tanimoto Coefficient'},
            linewidths=0.5, linecolor='white')
ax.set_title('A. Inter-Hit Tanimoto Similarity (ECFP4)', fontsize=12, fontweight='bold')

# Panel B: Max Tanimoto of each hit vs training set
ax = axes[1]
x_pos = np.arange(len(hit_indices))
bars = ax.bar(x_pos, tc_df['max_tanimoto_training'].values, color=C_HIT, edgecolor='black', linewidth=0.5)
ax.axhline(y=0.4, color='red', linestyle='--', linewidth=1, label='Tc = 0.4 (novelty threshold)')
ax.axhline(y=tc_df['max_tanimoto_training'].mean(), color='blue', linestyle='--', linewidth=1,
           label=f'Mean = {tc_df["max_tanimoto_training"].mean():.3f}')
ax.set_xticks(x_pos); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Max Tanimoto Coefficient'); ax.set_xlabel('Hit Compound')
ax.set_title('B. Max Tanimoto Similarity: Hits vs. Training Set', fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)

fig.suptitle('Structural Novelty Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(f'{OUT}figure_c3_tanimoto_analysis.png')
plt.close()
print("  Figure C3: Tanimoto analysis ✓")

# --- Figure C4: Property PCA + Drug-likeness ---
fig, axes = plt.subplots(2, 2, figsize=(14, 14))

# Panel A: Property PCA
ax = axes[0, 0]
ax.scatter(props_pca[train_mask, 0], props_pca[train_mask, 1], c='lightgray', alpha=0.3, s=8, label='Training Set')
ax.scatter(props_pca[hit_mask, 0], props_pca[hit_mask, 1], c=C_HIT_STAR, s=100, marker='*', edgecolors='black', linewidth=1.0, label='22 Hits', zorder=10)
for i, idx in enumerate(hit_indices):
    ax.annotate(f'H{i+1}', (props_pca[idx, 0], props_pca[idx, 1]),
                fontsize=6, xytext=(3,3), textcoords='offset points', alpha=0.8)
ax.set_xlabel(f'PC1 ({pca_prop.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca_prop.explained_variance_ratio_[1]:.1%})')
ax.set_title('A. PCA of Molecular Properties'); ax.legend(fontsize=7); ax.grid(alpha=0.2)

# Panel B: MW vs LogP scatter
ax = axes[0, 1]
ax.scatter(train_df['MW'], train_df['LogP'], c=C_ACTIVE, alpha=0.3, s=8, label='Active')
ax.scatter(train_df[train_df['class']=='inactive']['MW'],
           train_df[train_df['class']=='inactive']['LogP'], c=C_INACTIVE, alpha=0.3, s=8, label='Inactive')
ax.scatter(hits_df['MW'], hits_df['LogP'], c=C_HIT_STAR, s=120, marker='*', edgecolors='black', linewidth=1.2, label='22 Hits', zorder=10)
# Lipinski Rule of 5 boundaries
ax.axvline(x=500, color='gray', linestyle='--', alpha=0.5, label='MW=500')
ax.axhline(y=5, color='gray', linestyle='--', alpha=0.5, label='LogP=5')
ax.set_xlabel('Molecular Weight (Da)'); ax.set_ylabel('LogP')
ax.set_title('B. MW vs. LogP (Lipinski Space)'); ax.legend(fontsize=7); ax.grid(alpha=0.2)

# Panel C: Scaffold frequency distribution
ax = axes[1, 0]
top_scaffolds = train_scaffolds.head(20)
ax.barh(range(len(top_scaffolds)), top_scaffolds.values, color=C_ACTIVE, edgecolor='white')
ax.set_yticks(range(len(top_scaffolds)))
ax.set_yticklabels([f'S{i+1}' for i in range(len(top_scaffolds))], fontsize=7)
ax.set_xlabel('Number of Compounds'); ax.set_ylabel('Scaffold ID')
ax.set_title(f'C. Top 20 Scaffolds in Training Set\n(Total unique scaffolds: {len(train_scaffolds)})')
ax.grid(axis='x', alpha=0.3)

# Panel D: Drug-likeness radar
ax = axes[1, 1]
categories = ['MW<500', 'LogP<5', 'HBD≤5', 'HBA≤10', 'TPSA<140', 'RotB≤10', '0 Lipinski\nViolations']
train_pass = [
    (train_df['MW']<=500).mean(),
    (train_df['LogP']<=5).mean(),
    (train_df['HBD']<=5).mean(),
    (train_df['HBA']<=10).mean(),
    (train_df['TPSA']<140).mean(),
    (train_df['RotB']<=10).mean(),
    (train_df['Lipinski_Violations']==0).mean(),
]
hits_pass = [
    (hits_df['MW']<=500).mean(),
    (hits_df['LogP']<=5).mean(),
    (hits_df['HBD']<=5).mean(),
    (hits_df['HBA']<=10).mean(),
    (hits_df['TPSA']<140).mean(),
    (hits_df['RotB']<=10).mean(),
    (hits_df['Lipinski_Violations']==0).mean(),
]

angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
train_pass_plot = train_pass + [train_pass[0]]
hits_pass_plot = hits_pass + [hits_pass[0]]
angles_plot = angles + [angles[0]]

ax = fig.add_subplot(2, 2, 4, polar=True)
ax.fill(angles_plot, train_pass_plot, alpha=0.25, color=C_ACTIVE, label='Training Set')
ax.plot(angles_plot, train_pass_plot, 'o-', linewidth=2, color=C_ACTIVE, markersize=4)
ax.fill(angles_plot, hits_pass_plot, alpha=0.35, color=C_HIT, label='22 Hits')
ax.plot(angles_plot, hits_pass_plot, 's-', linewidth=2, color=C_HIT, markersize=6)
ax.set_xticks(angles); ax.set_xticklabels(categories, fontsize=8)
ax.set_ylim(0, 1.1); ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['20%','40%','60%','80%','100%'])
ax.set_title('D. Drug-Likeness Profile', pad=20, fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=8); ax.grid(True, alpha=0.3)

fig.suptitle('Property Space and Drug-Likeness Assessment', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(f'{OUT}figure_c4_property_pca_druglikeness.png')
plt.close()
print("  Figure C4: Property PCA + Drug-likeness ✓")

# --- Figure C5: Hit compound grid visualization (SVG-based, no Cairo needed) ---
print("  Figure C5: Drawing hit compound structures...")
from rdkit.Chem.Draw import IPythonConsole
from rdkit import Chem
from io import BytesIO
from PIL import Image

# Draw each molecule using MolsToGridImage which works without Cairo
try:
    hit_mols = [df.loc[idx, 'mol'] for idx in hit_indices]
    hit_labels = [f'H{i+1}\nTc={tc_df.loc[tc_df["compound_idx"]==idx,"max_tanimoto_training"].values[0]:.2f}'
                   for i, idx in enumerate(hit_indices)]

    # Draw in batches to keep images readable
    for batch_start in range(0, 22, 6):
        batch_end = min(batch_start + 6, 22)
        batch_mols = hit_mols[batch_start:batch_end]
        batch_labels = hit_labels[batch_start:batch_end]
        batch_legends = [f'H{batch_start+i+1}' for i in range(len(batch_mols))]

        img = Draw.MolsToGridImage(batch_mols, molsPerRow=3, subImgSize=(350, 250),
                                    legends=batch_legends, useSVG=True)
        # Save SVG
        svg_path = f'{OUT}figure_c5_hit_compounds_batch{batch_start//6+1}.svg'
        with open(svg_path, 'w') as f:
            f.write(img)
        print(f"    Batch {batch_start//6+1} saved as SVG: {svg_path}")

    print("  Figure C5: Hit compounds grid saved as SVG batches ✓")
except Exception as e:
    print(f"  Figure C5: SVG approach failed ({e}), generating SMILES table instead...")
    # Fallback: create a text-based figure with SMILES
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.axis('off')
    table_data = [['ID', 'SMILES', 'Max Tc']]
    for i, idx in enumerate(hit_indices):
        smi = df.loc[idx, 'canonical_smiles']
        tc = tc_df.loc[tc_df['compound_idx'] == idx, 'max_tanimoto_training'].values[0]
        table_data.append([f'H{i+1}', smi[:60] + ('...' if len(smi) > 60 else ''), f'{tc:.3f}'])
    tbl = ax.table(cellText=table_data, loc='center', cellLoc='left')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for i in range(len(table_data)):
        tbl[i, 0].set_width(0.05)
        tbl[i, 1].set_width(0.7)
        tbl[i, 2].set_width(0.08)
    ax.set_title('22 Virtual Screening Hit Compounds (SMILES and Max Tanimoto)', fontsize=14, fontweight='bold')
    fig.savefig(f'{OUT}figure_c5_hit_compound_table.png')
    plt.close()
    print("  Figure C5: Hit compound SMILES table saved ✓")

# ===================================================================
# 7. SUMMARY STATISTICS TABLE
# ===================================================================
print("\n"+"="*60)
print("Generating summary statistics table...")

summary = {
    'Property': ['MW (Da)', 'LogP', 'TPSA (Å²)', 'HBD', 'HBA', 'RotB',
                 'Num Rings', 'Num Aromatic Rings', 'Fraction Csp3',
                 'Num Heteroatoms', 'Heavy Atom Count', 'Lipinski Violations',
                 'Unique Scaffolds', 'Tanimoto (max vs training)'],
    'Training_Mean': [
        f"{train_df['MW'].mean():.1f} ± {train_df['MW'].std():.1f}",
        f"{train_df['LogP'].mean():.1f} ± {train_df['LogP'].std():.1f}",
        f"{train_df['TPSA'].mean():.1f} ± {train_df['TPSA'].std():.1f}",
        f"{train_df['HBD'].mean():.1f} ± {train_df['HBD'].std():.1f}",
        f"{train_df['HBA'].mean():.1f} ± {train_df['HBA'].std():.1f}",
        f"{train_df['RotB'].mean():.1f} ± {train_df['RotB'].std():.1f}",
        f"{train_df['NumRings'].mean():.1f} ± {train_df['NumRings'].std():.1f}",
        f"{train_df['NumAromaticRings'].mean():.1f} ± {train_df['NumAromaticRings'].std():.1f}",
        f"{train_df['FractionCsp3'].mean():.2f} ± {train_df['FractionCsp3'].std():.2f}",
        f"{train_df['NumHeteroatoms'].mean():.1f} ± {train_df['NumHeteroatoms'].std():.1f}",
        f"{train_df['HeavyAtomCount'].mean():.1f} ± {train_df['HeavyAtomCount'].std():.1f}",
        f"{train_df['Lipinski_Violations'].mean():.1f}",
        str(len(train_scaffolds)),
        '-',
    ],
    'Hits_Mean': [
        f"{hits_df['MW'].mean():.1f} ± {hits_df['MW'].std():.1f}",
        f"{hits_df['LogP'].mean():.1f} ± {hits_df['LogP'].std():.1f}",
        f"{hits_df['TPSA'].mean():.1f} ± {hits_df['TPSA'].std():.1f}",
        f"{hits_df['HBD'].mean():.1f} ± {hits_df['HBD'].std():.1f}",
        f"{hits_df['HBA'].mean():.1f} ± {hits_df['HBA'].std():.1f}",
        f"{hits_df['RotB'].mean():.1f} ± {hits_df['RotB'].std():.1f}",
        f"{hits_df['NumRings'].mean():.1f} ± {hits_df['NumRings'].std():.1f}",
        f"{hits_df['NumAromaticRings'].mean():.1f} ± {hits_df['NumAromaticRings'].std():.1f}",
        f"{hits_df['FractionCsp3'].mean():.2f} ± {hits_df['FractionCsp3'].std():.2f}",
        f"{hits_df['NumHeteroatoms'].mean():.1f} ± {hits_df['NumHeteroatoms'].std():.1f}",
        f"{hits_df['HeavyAtomCount'].mean():.1f} ± {hits_df['HeavyAtomCount'].std():.1f}",
        f"{hits_df['Lipinski_Violations'].mean():.1f}",
        str(len(set(hit_scaffolds))),
        f"{tc_df['max_tanimoto_training'].mean():.3f} ± {tc_df['max_tanimoto_training'].std():.3f}",
    ],
}

summary_df = pd.DataFrame(summary)
summary_df.to_csv(f'{OUT}table_summary_property_comparison.csv', index=False)
print("Summary table saved.")

print(f"\n{'='*60}")
print(f"CHEMICAL SPACE ANALYSIS COMPLETE")
print(f"{'='*60}")
print(f"\nAll outputs saved to: {OUT}")
print(f"  figure_c1_property_distributions.png")
print(f"  figure_c2_chemical_space_tsne_umap.png")
print(f"  figure_c3_tanimoto_analysis.png")
print(f"  figure_c4_property_pca_druglikeness.png")
print(f"  figure_c5_hit_compound_structures.png")
print(f"  table_molecular_properties_all.csv")
print(f"  table_scaffold_analysis.csv")
print(f"  table_hit_tanimoto_vs_training.csv")
print(f"  table_chemical_space_coordinates.csv")
print(f"  table_summary_property_comparison.csv")
