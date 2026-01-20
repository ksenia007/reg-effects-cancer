# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

cosmic_genes = pd.read_csv('resources/cosmic.07142025.21_22_13.csv')
cosmic_genes_tier1 = cosmic_genes[cosmic_genes['Tier'] == 1]['Gene Symbol'].unique().tolist()
cosmic_genes_tier2 = cosmic_genes[cosmic_genes['Tier'] == 2]['Gene Symbol'].unique().tolist()
cosmic_hallmark = cosmic_genes[cosmic_genes['Hallmark'] == 'Yes']['Gene Symbol'].unique().tolist()

print(f"Number of COSMIC Tier 1 genes: {len(cosmic_genes_tier1)}")
print(f"Number of COSMIC Tier 2 genes: {len(cosmic_genes_tier2)}")
print(f"Number of COSMIC Hallmark genes: {len(cosmic_hallmark)}")

non_cosmic = []
hallmark = []

LOC_FILES_GENE   = 'protein_coding.variant_to_gene.deid.tsv'
LOC_PREDS        = 'sorted.protein_coding_20kb.all_chunks.sequence_class_scores.tsv'
PYCLONE_COMBINED = 'pyclone_vi_all_results.filtered_assignm_prob.tsv'
INFO_USED = 'SUPP.surv_df_cosmic_hallmark.csv'
DIST = 20000
SEQ_CLASSES_ALL = ['PC1 Polycomb / Heterochromatin', 'L1 Low signal', 'TN1 Transcription',
       'TN2 Transcription', 'L2 Low signal', 'E1 Stem cell', 'E2 Multi-tissue',
       'E3 Brain / Melanocyte', 'L3 Low signal', 'E4 Multi-tissue',
       'TF1 NANOG / FOXA1', 'HET1 Heterochromatin', 'E5 B-cell-like',
       'E6 Weak epithelial', 'TF2 CEBPB', 'PC2 Weak Polycomb',
       'E7 Monocyte / Macrophage', 'E8 Weak multi-tissue', 'L4 Low signal',
       'TF3 FOXA1 / AR / ESR1', 'PC3 Polycomb', 'TN3 Transcription',
       'L5 Low signal', 'HET2 Heterochromatin', 'L6 Low signal', 'P Promoter',
       'E9 Liver / Intestine', 'CTCF CTCF-Cohesin', 'TN4 Transcription',
       'HET3 Heterochromatin', 'E10 Brain', 'TF4 OTX2', 'HET4 Heterochromatin',
       'L7 Low signal', 'PC4 Polycomb / Bivalent stem cell Enh',
       'HET5 Centromere', 'E11 T-cell', 'TF5 AR', 'E12 Erythroblast-like',
       'HET6 Centromere']

NORM_STD  = np.load('resources/1000G.EUR.std.npy')[:len(SEQ_CLASSES_ALL)]

# Variant -> gene mapping
var_gene = pd.read_csv(LOC_FILES_GENE, sep='\t')
var_gene['loc'] = var_gene['variant_id'].str.split('_', expand=True)[1].astype(int)
# Keep only what we need from variant to gene mapping
var_gene['dist'] = np.abs(var_gene['window_start'].astype(int) + 20000 - var_gene['loc'])
print(f"Before filtering, max distance is {var_gene['dist'].max()}")
var_gene = var_gene[var_gene['dist'] <= DIST]
var_gene = var_gene[['variant_id', 'gene_name']]

path = "/pph/controlled/dbGaP/GDC_TCGA/input/genetic/snv/processed_sets/pyclone_vi_all_results.filtered_assignm_prob.tsv"

col0 = pd.read_csv(
    path,
    sep="\t",
    usecols=['mutation_id', 'sample_id'],          
    header=0,          
    dtype={0: "string"},  
    engine="c",
)
print(f"Loaded {col0.shape[0]} rows from {path}")
print(f"Unique samples: {col0['sample_id'].nunique()}")
used = pd.read_csv(INFO_USED, sep=',')['sample_id'].unique().tolist()
col0 = col0[col0['sample_id'].isin(used)]
print(f"Filtered to {col0.shape[0]} rows after keeping only used samples")
print(f"Unique samples after filtering: {col0['sample_id'].nunique()}")

# as a Series:
s = col0.iloc[:, 0]
var_gene = var_gene[var_gene['variant_id'].isin(s)]
preds = pd.read_csv(LOC_PREDS, sep='\t')

# Normalize SEQ_CLASSES_ALL and take abs 
preds[SEQ_CLASSES_ALL] = np.abs((preds[SEQ_CLASSES_ALL]) / NORM_STD) 
preds['MAX_ALL'] = preds[SEQ_CLASSES_ALL].max(axis=1)
# rename 'name' column to 'variant_id' for merging
preds.rename(columns={'name': 'variant_id'}, inplace=True)

# downcast to save RAM
float_cols = SEQ_CLASSES_ALL + ['MAX_ALL']
print(preds[float_cols].dtypes)
print(f"Downcasting {len(float_cols)} float columns, currently using {preds[float_cols].memory_usage(deep=True).sum() / 1e6:.1f} MB")
preds[float_cols] = preds[float_cols].astype('float16')
print(f"Downcasted float columns now using {preds[float_cols].memory_usage(deep=True).sum() / 1e6:.1f} MB")

print(f"preds columns: {preds.columns.tolist()}")

# Variant -> gene + SEI
vg = var_gene.merge(
    preds,
    on='variant_id',
    how='inner'
)

print(f"Merged to {len(vg)} variant to gene+SEI+metadata rows")
print(f"using {vg.memory_usage(deep=True).sum() / 1e6:.1f} MB of RAM")
max_across_hallmark = vg[vg['gene_name'].isin(cosmic_hallmark)]['MAX_ALL'].values
print(f"Hallmark genes max SEI across all variants: mean={max_across_hallmark.mean():.3f}, median={np.median(max_across_hallmark):.3f}")
max_across_non_cosmic = vg[~vg['gene_name'].isin(cosmic_genes_tier1 + cosmic_genes_tier2 + cosmic_hallmark)]['MAX_ALL'].values
print(f"Non-COSMIC genes max SEI across all variants: mean={max_across_non_cosmic.mean():.3f}, median={np.median(max_across_non_cosmic):.3f}")
value_use = 'effect'                  

# ---- PREPARE CATEGORIES ----
# take preds only for genes w/ non-zero NVAR_ALL
preds_hallmark   = max_across_hallmark
preds_non_cosmic = max_across_non_cosmic
# Combined DataFrame for both score and NVAR
vals = np.concatenate([preds_hallmark, preds_non_cosmic])
use_df = pd.DataFrame({
    value_use: vals,
    "Category": (["Hallmark"] * len(preds_hallmark)) + (["Non-COSMIC"] * len(preds_non_cosmic)),
})
category_order = ['Non-COSMIC', 'Hallmark']
use_df['Category'] = pd.Categorical(
    use_df['Category'], categories=category_order, ordered=True
)
pairs = [
    ('Non-COSMIC', 'Hallmark'),
]
use_df['score_norm'] = use_df[value_use]
score_ylabel = f"{value_use}"

for i, (cat1, cat2) in enumerate(pairs):
    v1 = use_df.loc[use_df['Category'] == cat1, value_use]
    v2 = use_df.loc[use_df['Category'] == cat2, value_use]

    # keep directional MW test 
    stat, pval = mannwhitneyu(v1, v2, alternative='less')
    print(f'[SCORE] {cat1} vs {cat2}: '
          f", p={pval:.4e}")

# %%
