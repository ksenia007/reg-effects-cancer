# %%
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
grey_red = LinearSegmentedColormap.from_list("grey_red", ["#bdbdbd", "#d7301f"])
import matplotlib as mpl
import matplotlib.ticker as mticker

from utils import compute_surv_df_for_gene_list, plot_km_by_group_pretty, compute_surv_df_for_gene_list, \
    add_group_from_continuous, fit_cox, gene_dropout_stability, sample_dropout_stability
    
from visualization_utils import plot_corr_heatmap, p_to_stars


use_id = "sample_id"
DATA = 'data/set_20kbp.protein_coding.gene_level_aggregated.dist_20000.tsv'
base_patient = 'data/base_patient_df.filtered_hyperMut20.filtered_PFIproj.tsv'
base_patient_df = pd.read_csv(base_patient, sep='\t')
print(f"Loaded base patient data with {len(base_patient_df)} rows")
df = pd.read_csv(DATA, sep='\t')
print(f"Loaded aggregated data with {len(df)} rows")
df['MAIN_REG_SCORE'] = df['MAX_ALL_MAX_ALL'] / df['NVAR_ALL'].replace(0, 1)
cosmic_genes = pd.read_csv('data/cosmic.07142025.21_22_13.csv')
cosmic_hallmark = cosmic_genes[cosmic_genes['Hallmark'] == 'Yes']['Gene Symbol'].unique().tolist()
cosmic_sets = {
    'cosmic_hallmark': cosmic_hallmark,
}
# save cosmic hallmark genes as a list txt
with open('results/used_cosmic_hallmark_genes.txt', 'w') as f:
    for gene in cosmic_hallmark:
        f.write(f"{gene}\n")
        
# print summary of cosmic sets in N genes
for set_name, genes in cosmic_sets.items():
    print(f"{set_name}: {len(genes)} genes")

hallmark_genes = set(cosmic_hallmark)
df['is_hallmark'] = df['gene_name'].isin(hallmark_genes)
print("Hallmark rows:", df['is_hallmark'].sum())

### PLOT: hallmark genes recurrence vs effect
hm = df[df['is_hallmark']].copy()
gene_hit = hm.groupby('gene_name').agg(
    n_samples_hit=('sample_id', 'nunique'),
    total_vars=('NVAR_ALL', 'sum'),
    mean_effect=('MAIN_REG_SCORE', 'mean'),
).reset_index()
gene_hit['mean_vars_per_hit_sample'] = gene_hit['total_vars'] / gene_hit['n_samples_hit'].clip(lower=1)
# size = # cancer types hit 
sample_to_ct = base_patient_df.set_index('sample_id')['cancer_type']
tmp = hm[['sample_id','gene_name']].drop_duplicates().copy()
tmp['cancer_type'] = tmp['sample_id'].map(sample_to_ct)
ct_counts = tmp.groupby('gene_name')['cancer_type'].nunique().rename('n_cancer_types_hit')
gene_hit = gene_hit.merge(ct_counts, on='gene_name', how='left').fillna({'n_cancer_types_hit': 0})

topK = len(gene_hit) # plot all
gene_hit = gene_hit.sort_values('n_samples_hit', ascending=False).head(topK)

YCOL = 'mean_effect'  
color_vals = np.log1p(gene_hit['mean_vars_per_hit_sample'])
size_base = 18
sizes = (gene_hit['n_cancer_types_hit'].clip(lower=1) ** 1.3) * size_base
ylabel_map = {
    'mean_effect': "Mean per-sample effect",
}
fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=250)

fig.subplots_adjust(right=0.78)  

base = mpl.cm.get_cmap('YlOrRd')
cmap_dark = LinearSegmentedColormap.from_list('YlOrRd_dark', base(np.linspace(0.25, 1.0, 256)))

sc = ax.scatter(
    gene_hit['n_samples_hit'],
    gene_hit[YCOL],
    s=sizes,
    c=color_vals,
    alpha=0.75,
    zorder=3,  # points above everything
    cmap=cmap_dark
)

ax.set_axisbelow(True)          # grid behind points
ax.grid(alpha=0.2, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_xlabel("# samples with ≥1 noncoding hit", fontsize=13)
ax.set_ylabel(ylabel_map.get(YCOL, YCOL), fontsize=13)

# colorbar in reserved margin (won’t overlap plot)
cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
cbar.set_label("log(1 + mean # variants per hit-sample)", fontsize=13)
cbar.ax.tick_params(labelsize=12) 

# --- labels: extremes only (keeps plot readable) ---
to_label = pd.concat([
    gene_hit.nlargest(4, 'n_samples_hit'),
    gene_hit.nlargest(2, YCOL),
    gene_hit.nlargest(2, 'n_cancer_types_hit') 
]).drop_duplicates(subset=['gene_name']).head(16)

# labels
for _, r in to_label.iterrows():
    ax.text(r['n_samples_hit'] - 5, r[YCOL]+0.15, r['gene_name'], fontsize=11, zorder=4)

# size legend (place under colorbar in reserved margin)
ref = [2, 5, 10, 20]
ref = [v for v in ref if v <= gene_hit['n_cancer_types_hit'].max()]
if not ref:
    ref = [max(1, int(gene_hit['n_cancer_types_hit'].max()))]

handles = [
    ax.scatter([], [], s=(max(v,1) ** 1.3) * size_base, color='grey', alpha=0.6)
    for v in ref
]
labels = [f"{v} cancer types" for v in ref]

ax.legend(
    handles, labels,
    loc='upper left',
    bbox_to_anchor=(0.6, 0.95),
    frameon=False,
    fontsize=12, 
)
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
plt.show()

# MAIN: Cox analysis 
results = {}
for set_name, genes in cosmic_sets.items():
    surv_df = compute_surv_df_for_gene_list(df, base_patient_df, genes, set_name, 
                                            zscore=True)
    surv_df.to_csv(f'results/surv_df_{set_name}.csv', sep=',', index=False)
    # plot correlation heatmap
    fig, ax, corr = plot_corr_heatmap(
        surv_df,
        method="pearson",
        title="" 
    )
    plt.show()
    _, cph = fit_cox(surv_df)
    cph.print_summary()
    cph.plot()
    print(f"HR for effect_score in {set_name}: {cph.hazard_ratios_['effect_score']:.3f}, CI=[{cph.summary.loc['effect_score', 'exp(coef) lower 95%']:.3f}, {cph.summary.loc['effect_score', 'exp(coef) upper 95%']:.3f}], p={cph.summary.loc['effect_score', 'p']:.3e}")
    cph.check_assumptions(surv_df, p_value_threshold=0.05, show_plots=True)
    cph.summary.to_csv(f'results/cox_summary_{set_name}.FINAL.tsv', sep='\t')

# PLOT FOREST PLOT PRETTY
summary = cph.summary.copy().reset_index().rename(columns={'covariate': 'variable'})
order = [
    'effect_score',
    'log_fga',
    'age_at_diagnosis',
    'nc_mutation_burden',
    'non_silent_per_Mb'
]
summary = summary.set_index('variable')
summary = summary.loc[[v for v in order if v in summary.index]].reset_index()
base_labels = {
    'effect_score': 'Regulatory score',
    'nc_mutation_burden': 'Regulatory\nvariant count',
    'non_silent_per_Mb': 'Non-silent\nmutations (per Mb)',
    'age_at_diagnosis': 'Age at diagnosis',
    'log_fga': 'log FGA',
}
base_labels = [base_labels.get(v, v) for v in summary['variable'].values]
colors = ['tomato'] + ['grey']*(len(summary)-1)
stars = [p_to_stars(p) for p in summary['p'].values]
pretty_labels = [f"{lab} {star}".rstrip() for lab, star in zip(base_labels, stars)]
# log(HR) and log-CIs
hr = summary['exp(coef)'].values
ci_low = summary['exp(coef) lower 95%'].values
ci_hi  = summary['exp(coef) upper 95%'].values
y_pos = np.arange(len(pretty_labels))
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
fig, ax = plt.subplots(figsize=(6.2, 3.9), dpi=500)
ax.axvline(1.0, color='lightgray', linestyle='--', linewidth=1)
for i, (h, lo, hi, col) in enumerate(zip(hr, ci_low, ci_hi, colors)):
    ax.errorbar(
        h, i,
        xerr=[[h - lo], [hi - h]],
        fmt='o',
        markersize=6,
        capsize=3,
        linewidth=1.8,
        color=col,
        ecolor=col,
        zorder=3
    )

ax.set_yticks(y_pos)
ax.set_yticklabels(pretty_labels, fontsize=11)
ax.invert_yaxis()
# axis limits + nice symmetric ticks
xmin = 2-float(np.nanmax(ci_hi))
xmax = float(np.nanmax(ci_hi))
pad = 0.08 * (xmax - xmin + 1e-9)
ax.set_xlim(xmin - pad, xmax + pad)
ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
ax.xaxis.set_minor_locator(mticker.NullLocator())
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2g'))
ax.set_xlabel('Hazard ratio (HR) (95% CI)', fontsize=12)
ax.set_axisbelow(True)
ax.grid(axis='x', alpha=0.15, zorder=0)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
ax.tick_params(axis='y', length=0)
plt.tight_layout()
plt.show()

########  KAPLAN MEIER
km_df, info = add_group_from_continuous(surv_df, col="effect_score")
plot_km_by_group_pretty(km_df, title=f"")


########  STABILITY
all_reps_gene = {}
leave_fraqs = [0.01, 0.05, 0.1] 
set_name = 'cosmic_hallmark' 
genes = cosmic_sets[set_name]
print(f"\n=== {set_name} ===")
for frac in leave_fraqs:
    print(f"\n--- Drop fraction: {frac} ---")
    reps = gene_dropout_stability(
        df=df,
        base_patient_df=base_patient_df,
        tier_genes=genes,
        set_name=set_name,
        drop_frac=frac,
        n_reps=50,
        seed=42,
    )
    all_reps_gene[frac] = reps

# ---- extract baseline (shared across fracs) ----
first_frac = next(iter(all_reps_gene.keys()))
baseline_coef = all_reps_gene[first_frac]["baseline"]["coef"]

# ---- collect replicate coefs per fraction ----
fractions = sorted(all_reps_gene.keys())
coef_groups = []
for frac in fractions:
    reps = all_reps_gene[frac]
    rep_keys = sorted([k for k in reps.keys() if isinstance(k, (int, np.integer))])
    # if any of the reps non significant - all 0s
    max_p = max([reps[k]["p_value"] for k in rep_keys])
    if max_p >= 0.05:
        print("At drop fraction", frac, "at least one replicate non-significant (max p =", max_p, "); setting coefs to 0.0")
        coef_groups.append([0.0 for k in rep_keys])
    else:
        coef_groups.append([reps[k]["coef"] for k in rep_keys if reps[k]["p_value"] < 0.05])

# remove empty groups - all 0s
coef_groups_filtered = []
fractions_filtered = []
for frac, coefs in zip(fractions, coef_groups):
    if len(coefs) > 0 and not all(c == 0.0 for c in coefs):
        coef_groups_filtered.append(coefs)
        fractions_filtered.append(frac)
coef_groups = coef_groups_filtered
fractions = fractions_filtered

# ---- plot ---
fig, ax = plt.subplots(figsize=(4.1, 5.4), dpi=250)
ax.set_axisbelow(True)
ax.grid(alpha=0.2, zorder=0)
bp = ax.boxplot(
    coef_groups,
    labels=[f"{int(frac*100)}%" for frac in fractions],
    patch_artist=True,     # allow filling boxes
    showfliers=True,
    widths=0.55,
    medianprops=dict(linewidth=2),
    whiskerprops=dict(linewidth=1.5),
    capprops=dict(linewidth=1.5),
)
# --- style boxes/lines (clean grey) ---
for box in bp["boxes"]:
    box.set_alpha(0.25)      # light fill
    box.set_linewidth(1.5)
for k in ["whiskers", "caps", "medians", "fliers"]:
    for obj in bp[k]:
        obj.set_zorder(2)
# make lines a consistent neutral grey
for element in ["boxes", "whiskers", "caps", "medians"]:
    for obj in bp[element]:
        obj.set_color("grey")

# fliers subtle
for fl in bp["fliers"]:
    fl.set_marker("o")
    fl.set_markersize(4)
    fl.set_alpha(0.5)
    fl.set_markeredgecolor("grey")
    fl.set_markerfacecolor("grey")

# baseline dashed line (tomato), above grid/boxes
ax.axhline(baseline_coef, linestyle="--", linewidth=2, color="tomato", zorder=3)

# labels/ticks to match your style
ax.set_xlabel("Fraction of genes dropped", fontsize=18)
ax.set_ylabel("Hazard ratio (HR)", fontsize=18)
ax.tick_params(axis="x", labelsize=15)
ax.tick_params(axis="y", labelsize=15)

# optional title (kept minimal)
ax.set_ylim(bottom=min(1, ax.get_ylim()[0]))

ax.set_axisbelow(True)          
ax.grid(alpha=0.2, zorder=0)
ax.set_axisbelow(True)

# clean spines like boxplots
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.show()



all_reps_sample = {}
leave_fracs = [0.01, 0.05, 0.1] 
set_name = "cosmic_hallmark"
genes = cosmic_sets[set_name]

print(f"\n=== {set_name} ===")
for frac in leave_fracs:
    print(f"\n--- Drop sample fraction: {frac} ---")
    reps = sample_dropout_stability(
        df=df,
        base_patient_df=base_patient_df,
        tier_genes=genes,
        set_name=set_name,
        drop_frac=frac,
        n_reps=50,
        seed=42,
        use_id="sample_id",
        recompute_surv=True,  
        return_dropped=False,
    )
    all_reps_sample[frac] = reps

# ---- extract baseline (shared across fracs) ----
first_frac = next(iter(all_reps_sample.keys()))
baseline_coef = all_reps_sample[first_frac]["baseline"]["coef"]

# ---- collect replicate coefs per fraction ----
fractions = sorted(all_reps_sample.keys())
coef_groups = []
for frac in fractions:
    reps = all_reps_sample[frac]
    rep_keys = sorted([k for k in reps.keys() if isinstance(k, (int, np.integer))])
    # if any of the reps non significant - all 0s
    max_p = max([reps[k]["p_value"] for k in rep_keys])
    if max_p >= 0.05:
        coef_groups.append([0.0 for k in rep_keys])
    else:
        coef_groups.append([reps[k]["coef"] for k in rep_keys if reps[k]["p_value"] < 0.05])

# remove empty groups - all 0s
coef_groups_filtered = []
fractions_filtered = []
for frac, coefs in zip(fractions, coef_groups):
    if len(coefs) > 0 and not all(c == 0.0 for c in coefs):
        coef_groups_filtered.append(coefs)
        fractions_filtered.append(frac)
coef_groups = coef_groups_filtered
fractions = fractions_filtered

# ---- plot ---
fig, ax = plt.subplots(figsize=(3.1, 3.8), dpi=250)

ax.set_axisbelow(True)
ax.grid(alpha=0.2, zorder=0)

bp = ax.boxplot(
    coef_groups,
    labels=[f"{int(frac*100)}%" for frac in fractions],
    patch_artist=True,     # allow filling boxes
    showfliers=True,
    widths=0.55,
    medianprops=dict(linewidth=2),
    whiskerprops=dict(linewidth=1.5),
    capprops=dict(linewidth=1.5),
)

# --- style boxes/lines (clean grey) ---
for box in bp["boxes"]:
    box.set_alpha(0.25)      # light fill
    box.set_linewidth(1.5)
for k in ["whiskers", "caps", "medians", "fliers"]:
    for obj in bp[k]:
        obj.set_zorder(2)

# make lines a consistent neutral grey
for element in ["boxes", "whiskers", "caps", "medians"]:
    for obj in bp[element]:
        obj.set_color("grey")

# fliers subtle
for fl in bp["fliers"]:
    fl.set_marker("o")
    fl.set_markersize(4)
    fl.set_alpha(0.5)
    fl.set_markeredgecolor("grey")
    fl.set_markerfacecolor("grey")

# baseline dashed line (tomato), above grid/boxes
ax.axhline(baseline_coef, linestyle="--", linewidth=2, color="tomato", zorder=3)

# labels/ticks to match your style
ax.set_xlabel("Fraction samples dropped", fontsize=14)
ax.set_ylabel("Hazard ratio (HR)", fontsize=14)
ax.tick_params(axis="x", labelsize=13)
ax.tick_params(axis="y", labelsize=13)

# optional title (kept minimal)
ax.set_ylim(bottom=min(1, ax.get_ylim()[0]))

ax.set_axisbelow(True)          
ax.grid(alpha=0.2, zorder=0)
ax.set_axisbelow(True)

# clean spines like boxplots
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.show()


## LOCO cancer
min_events = 50
rows = []
# Baseline (all cancers)
baseline_fit, baseline_cph = fit_cox(surv_df)
baseline_HR = baseline_fit["coef"]         
baseline_lo = baseline_fit["coef_lower"]
baseline_hi = baseline_fit["coef_upper"]

# LOCO
for ct in sorted(surv_df["cancer_type"].dropna().unique()):
    d2 = surv_df[surv_df["cancer_type"] != ct].copy() # no need to recompute

    # keep strata meaningful 
    if d2["cancer_type"].nunique() < 2:
        continue
    n_events = int(d2["event"].sum())
    if n_events < min_events:
        continue

    rep_fit, _ = fit_cox(d2)

    hr = rep_fit["coef"]           # HR
    lo = rep_fit["coef_lower"]     # HR lower 95%
    hi = rep_fit["coef_upper"]     # HR upper 95%

    if rep_fit['p_value'] >= 0.05:
        raise ValueError(f"LOCO {ct}: non-significant replicate (p={rep_fit['p_value']:.3e}); cannot include in LOCO plot")
    rows.append({
        "left_out": ct,
        "HR": hr,
        "HR_lo": lo,
        "HR_hi": hi,
        "p": rep_fit["p_value"],
        "n": int(len(d2)),
        "events": n_events,
        "delta_logHR": float(np.log(hr) - np.log(baseline_HR)),
    })

loco_df = pd.DataFrame(rows).sort_values("HR").reset_index(drop=True)

results, cph = fit_cox(surv_df)
coef_full = results['coef']
se_full = cph.summary.loc['effect_score','se(coef)']

HR_lo_full = np.exp(coef_full - 1.96 * se_full)
HR_hi_full = np.exp(coef_full + 1.96 * se_full)

# ---- full model HR + CI ----
coef_full = float(cph.summary.loc['effect_score', 'coef'])
se_full   = float(cph.summary.loc['effect_score', 'se(coef)'])

HR_full    = np.exp(coef_full)
HR_lo_full = np.exp(coef_full - 1.96 * se_full)
HR_hi_full = np.exp(coef_full + 1.96 * se_full)

# ---- order cancers ----
loco_plot = loco_df.sort_values('HR').reset_index(drop=True)

x = np.arange(len(loco_plot))
y = loco_plot['HR'].values
yerr = np.vstack([
    y - loco_plot['HR_lo'].values,
    loco_plot['HR_hi'].values - y
])

# ---- figure ----
fig, ax = plt.subplots(
    figsize=(max(5.4, 0.17 * len(x)), 4.6),
    dpi=300
)

# ---- full-model CI band (grey) ----
ax.axhspan(
    HR_lo_full,
    HR_hi_full,
    color='lightgray',
    alpha=0.25,
    zorder=1
)

# ---- LOCO points + CI (grey) ----
ax.errorbar(
    x,
    y,
    yerr=yerr,
    fmt='o',
    markersize=4.8,
    capsize=2.5,
    linewidth=1.4,
    color='grey',
    ecolor='grey',
    alpha=0.85,
    zorder=3
)

# ---- full-model point + CI (tomato, separated) ----
x_full = -1.4
ax.errorbar(
    x_full,
    HR_full,
    yerr=[[HR_full - HR_lo_full], [HR_hi_full - HR_full]],
    fmt='o',
    markersize=7.5,
    capsize=4,
    linewidth=2.2,
    color='tomato',
    ecolor='tomato',
    zorder=4
)

# ---- axes / labels ----
ax.set_xticks(x)
ax.set_xticklabels(loco_plot['left_out'], rotation=90, fontsize=9.8)
ax.set_xlim(x_full - 0.6, len(x) - 0.5)

ax.set_ylabel("Hazard ratio (HR)", fontsize=14)
ax.set_xlabel("Project left out", fontsize=14)

# ---- clean style ----
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)
ax.set_axisbelow(True)

fig.tight_layout()
plt.show()


# %%
