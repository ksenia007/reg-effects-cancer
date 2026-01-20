import numpy as np
import pandas as pd

# ---------------------------------------------------
# Paths and constants
# ---------------------------------------------------
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

LOC_FILES_GENE   = 'protein_coding.variant_to_gene.deid.tsv'
LOC_PREDS        = 'sorted.protein_coding_20kb.all_chunks.sequence_class_scores.tsv'
PYCLONE_COMBINED = 'pyclone_vi_all_results.filtered_assignm_prob.tsv'
NORM_MEAN = np.load('resources/1000G.EUR.mean.npy')[:len(SEQ_CLASSES_ALL)]
NORM_STD  = np.load('resources/1000G.EUR.std.npy')[:len(SEQ_CLASSES_ALL)]
SAMPLE_COL = 'sample_id'          
CP_COL     = 'cellular_prevalence'
CHUNKSIZE  = 1_000_000
DIST      = 20000

# Variant -> gene mapping
# ---------------------------------------------------

var_gene = pd.read_csv(LOC_FILES_GENE, sep='\t')
var_gene['loc'] = var_gene['variant_id'].str.split('_', expand=True)[1].astype(int)
var_gene['dist'] = np.abs(var_gene['window_start'].astype(int) + 20000 - var_gene['loc'])
print(f"Before filtering, max distance is {var_gene['dist'].max()}")
var_gene = var_gene[var_gene['dist'] <= DIST]
# Keep only what we need from variant to gene mapping
var_gene = var_gene[['variant_id', 'gene_name']]
print(f"Filered to distance of {DIST} bp, {len(var_gene)} variant to gene rows")

# SEI predictions: normalize and downcast
# ---------------------------------------------------

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

# Chunked PyClone reader 
# ---------------------------------------------------

pyclone_reader = pd.read_csv(
    PYCLONE_COMBINED,
    sep='\t',
    chunksize=CHUNKSIZE,
    usecols=lambda c: c in ('sample_id', 'mutation_id', CP_COL)
)

# Per-bin chunk aggregates
sum_all_chunks        = []
max_all_chunks        = []
count_all_chunks      = []

sum_clonal_chunks     = []
max_clonal_chunks     = []
count_clonal_chunks   = []

sum_sub_chunks        = []
max_sub_chunks        = []
count_sub_chunks      = []

sum_early_chunks      = []
max_early_chunks      = []
count_early_chunks    = []

sum_late_chunks       = []
max_late_chunks       = []
count_late_chunks     = []


for pchunk in pyclone_reader:
    print('Processing PyClone chunk with', len(pchunk), 'rows')

    # Rename mutation_id -> variant_id to match vg
    pchunk = pchunk.rename(columns={'mutation_id': 'variant_id'})

    # Drop rows without CP, just in case
    pchunk = pchunk[pchunk[CP_COL].notna()]
    if pchunk.empty:
        continue

    # Merge per-sample PyClone chunk with variant->gene+SEI scores
    merged = pchunk.merge(
        vg,
        on='variant_id',
        how='inner'
    )
    if merged.empty:
        continue

    cp = merged[CP_COL]

    # ---- all variants ----
    m_all = merged
    grp_all = m_all.groupby([SAMPLE_COL, 'gene_name'], observed=True)
    sum_all = grp_all[float_cols].sum().reset_index()
    max_all = grp_all[float_cols].max().reset_index()
    n_all   = grp_all.size().reset_index(name='NVAR_ALL')
    sum_all_chunks.append(sum_all)
    max_all_chunks.append(max_all)
    count_all_chunks.append(n_all)   

    # ---- clonal: cp >= 0.9 ----
    m_clonal = merged[cp >= 0.9]
    if not m_clonal.empty:
        grp_cl = m_clonal.groupby([SAMPLE_COL, 'gene_name'], observed=True)
        sum_cl = grp_cl[float_cols].sum().reset_index()
        max_cl = grp_cl[float_cols].max().reset_index()
        n_cl   = grp_cl.size().reset_index(name='NVAR_CLONAL')

        sum_clonal_chunks.append(sum_cl)
        max_clonal_chunks.append(max_cl)
        count_clonal_chunks.append(n_cl)



    # ---- subclonal (all): cp < 0.9 ----
    m_sub = merged[cp < 0.9]
    if not m_sub.empty:
        grp_sub = m_sub.groupby([SAMPLE_COL, 'gene_name'], observed=True)
        sum_sub = grp_sub[float_cols].sum().reset_index()
        max_sub = grp_sub[float_cols].max().reset_index()
        n_sub   = grp_sub.size().reset_index(name='NVAR_SUBCLONAL')

        sum_sub_chunks.append(sum_sub)
        max_sub_chunks.append(max_sub)
        count_sub_chunks.append(n_sub)


# Combine partial aggregates across all chunks
# ---------------------------------------------------

print("Combining chunked aggregates...")

def _combine_sum(dfs, prefix):
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    df = df.groupby([SAMPLE_COL, 'gene_name'], as_index=False).sum()
    df.rename(columns={c: f'{prefix}{c}' for c in float_cols}, inplace=True)
    return df

def _combine_max(dfs, prefix):
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    df = df.groupby([SAMPLE_COL, 'gene_name'], as_index=False).max()
    df.rename(columns={c: f'{prefix}{c}' for c in float_cols}, inplace=True)
    return df

def _combine_count(dfs, col_name):
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    # multiple chunks may have the same sample x gene, so we sum counts
    df = df.groupby([SAMPLE_COL, 'gene_name'], as_index=False)[col_name].sum()
    return df

all_sum    = _combine_sum(sum_all_chunks,    'SUM_ALL_')
all_max    = _combine_max(max_all_chunks,    'MAX_ALL_')
all_count  = _combine_count(count_all_chunks, 'NVAR_ALL')

clonal_sum   = _combine_sum(sum_clonal_chunks,  'SUM_CLONAL_')
clonal_max   = _combine_max(max_clonal_chunks,  'MAX_CLONAL_')
clonal_count = _combine_count(count_clonal_chunks, 'NVAR_CLONAL')

sub_sum    = _combine_sum(sum_sub_chunks,   'SUM_SUBCLONAL_')
sub_max    = _combine_max(max_sub_chunks,   'MAX_SUBCLONAL_')
sub_count  = _combine_count(count_sub_chunks, 'NVAR_SUBCLONAL')

early_sum  = _combine_sum(sum_early_chunks, 'SUM_EARLY_SUB_')
early_max  = _combine_max(max_early_chunks, 'MAX_EARLY_SUB_')
early_count= _combine_count(count_early_chunks, 'NVAR_EARLY_SUB')

late_sum   = _combine_sum(sum_late_chunks,  'SUM_LATE_SUB_')
late_max   = _combine_max(max_late_chunks,  'MAX_LATE_SUB_')
late_count = _combine_count(count_late_chunks, 'NVAR_LATE_SUB')



# Merge all aggregates into final table
# ---------------------------------------------------

dfs = [
    all_sum, all_max, all_count,
    clonal_sum, clonal_max, clonal_count,
    sub_sum, sub_max, sub_count,
    early_sum, early_max, early_count,
    late_sum, late_max, late_count,
]

final = None
for df in dfs:
    if df is None:
        continue
    if final is None:
        final = df
    else:
        final = final.merge(df, on=[SAMPLE_COL, 'gene_name'], how='outer')

print(f"Final aggregated table has {len(final)} gene x sample rows")
print(final.head())

# Write out the result
# ---------------------------------------------------

OUT_FILE = f'output/set_20kbp.protein_coding.gene_level_aggregated.dist_{DIST}.tsv'
final.to_csv(OUT_FILE, sep='\t', index=False)
print(f"Wrote {len(final)} gene x sample rows to {OUT_FILE}")
