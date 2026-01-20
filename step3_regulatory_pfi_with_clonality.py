# %%
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from lifelines import CoxPHFitter
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker
grey_red = LinearSegmentedColormap.from_list("grey_red", ["#bdbdbd", "#d7301f"])
base = mpl.cm.get_cmap('YlOrRd')
cmap_dark = LinearSegmentedColormap.from_list('YlOrRd_dark', base(np.linspace(0.25, 1.0, 256)))

from utils import z_by_cancer, l2_norm
from visualization_utils import plot_hallmark_clonal_subclonal_box, p_to_stars
use_id = "sample_id"


def plot_corr_heatmap(
    surv_df: pd.DataFrame,
    cols=None,
    method: str = "pearson",   
    title: str = None,
    annotate: bool = True,
    figsize=(7, 6),
    vmin: float = -0.1,
    vmax: float =  1.0,
):
    """
    Plot a correlation matrix heatmap for selected columns in surv_df.
    Uses matplotlib only (no seaborn).
    """
    if cols is None:
        cols = [
            "effect_score_clonal",
            "effect_score_subclonal",
            "nc_mutation_burden_clonal",
            "nc_mutation_burden_subclonal",
            "non_silent_per_Mb",
            "log_fga",
            "age_at_diagnosis",
        ]

    # Keep only existing columns (avoids crashes if you renamed something)
    cols = [c for c in cols if c in surv_df.columns]
    if len(cols) < 2:
        raise ValueError(f"Need at least 2 valid columns to correlate; got: {cols}")

    d = surv_df[cols].copy()

    # Compute correlation matrix (pairwise complete observations)
    corr = d.corr(method=method)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr.values, vmin=vmin, vmax=vmax, cmap=cmap_dark)

    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"{method.capitalize()} correlation", fontsize=13)

    # Optional cell annotations
    if annotate:
        for i in range(len(cols)):
            for j in range(len(cols)):
                val = corr.values[i, j]
                if np.isnan(val):
                    txt = "NA"
                else:
                    txt = f"{val:.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=13)

    # Minor aesthetics: gridlines between cells
    label_map = {
    "effect_score_clonal": "Clonal regulatory score",
    "effect_score_subclonal": "Subclonal regulatory score",
    "nc_mutation_burden_clonal": "Clonal regulatory\nvariant count",
    "nc_mutation_burden_subclonal": "Subclonal regulatory\nvariant count",
    "non_silent_per_Mb": "Non-silent\nmutations (per Mb)",
    "age_at_diagnosis": "Age at diagnosis",
    "log_fga": "log FGA",
    }

    pretty = [label_map.get(c, c) for c in corr.columns]

    ax.set_xticklabels(pretty, rotation=45, ha="right", fontsize=14)
    ax.set_yticklabels(pretty, fontsize=14)
    ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(cols), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.tight_layout()
    return fig, ax, corr

def compute_surv_df_for_gene_list(
    df, base_patient_df, tier_genes, set_name,
    f=l2_norm, use_id="sample_id", 
    zscore=True, 
    var_count_col1 = "NVAR_CLONAL", 
    var_count_col2 = "NVAR_SUBCLONAL"
):
    """
    Recomputes effect_score + covariates (z-by-cancer) and returns the final surv_df
    exactly in the shape you fit Cox on.
    """
    tier_genes = list(set(tier_genes))
    N = len(tier_genes)
    if N == 0:
        raise ValueError(f"{set_name}: tier_genes empty")

    tier_subset = df[df["gene_name"].isin(tier_genes)].copy()

    tier_effects = (
        tier_subset.groupby(use_id)
        .agg({
            "REG_MAIN_CLONAL": lambda x: f(x, N),
            "REG_MAIN_SUBCLONAL": lambda x: f(x, N),
            var_count_col1: "sum",
            var_count_col2: "sum",
        })
        .rename(columns={
            "REG_MAIN_CLONAL": "effect_score_clonal",
            "REG_MAIN_SUBCLONAL": "effect_score_subclonal",
            var_count_col1: "nc_mutation_burden_clonal",
            var_count_col2: "nc_mutation_burden_subclonal",
        })
        .reset_index()
        .drop_duplicates()
    )

    surv_df = base_patient_df.merge(tier_effects, left_on="sample_id", right_on=use_id, how="left")

    surv_df = surv_df.drop_duplicates(subset=["case_id"])

    surv_df['effect_score_clonal'] = surv_df['effect_score_clonal'].fillna(0.0)
    surv_df['effect_score_subclonal'] = surv_df['effect_score_subclonal'].fillna(0.0)
    surv_df['nc_mutation_burden_clonal'] = surv_df['nc_mutation_burden_clonal'].fillna(0.0)
    surv_df['nc_mutation_burden_subclonal'] = surv_df['nc_mutation_burden_subclonal'].fillna(0.0)
        
    # z-score within cancer
    if zscore:
        surv_df["effect_score_clonal"] = z_by_cancer(surv_df, "effect_score_clonal")
        surv_df["effect_score_subclonal"] = z_by_cancer(surv_df, "effect_score_subclonal")
        surv_df["log_fga"] = z_by_cancer(surv_df, "log_fga")
        surv_df["non_silent_per_Mb"] = z_by_cancer(surv_df, "non_silent_per_Mb")
        surv_df["nc_mutation_burden_clonal"] = z_by_cancer(surv_df, "nc_mutation_burden_clonal")
        surv_df["nc_mutation_burden_subclonal"] = z_by_cancer(surv_df, "nc_mutation_burden_subclonal")
        surv_df["age_at_diagnosis"] = z_by_cancer(surv_df, "age_at_diagnosis")
                
    surv_df = (
        surv_df.dropna(subset=[
            "time", "event",
            "non_silent_per_Mb", "log_fga", "age_at_diagnosis",
            "effect_score_clonal", "effect_score_subclonal", 
            "nc_mutation_burden_clonal", "nc_mutation_burden_subclonal",
            "cancer_type",
        ])
        .drop_duplicates()
    )

    return surv_df

def fit_cox(surv_df):
    formula = """
        non_silent_per_Mb +
        log_fga +
        age_at_diagnosis +
        nc_mutation_burden_clonal +
        nc_mutation_burden_subclonal +
        effect_score_clonal +
        effect_score_subclonal    """
    print('Number of samples for Cox PH fit:', len(surv_df))
    cph = CoxPHFitter() 
    cph.fit(
        surv_df,
        duration_col="time",
        event_col="event",
        formula=formula,
        strata=["cancer_type"],
    )
    coef = float(cph.summary.loc["effect_score_clonal", "exp(coef)"])
    pval = float(cph.summary.loc["effect_score_clonal", "p"])
    coef_lower = float(cph.summary.loc["effect_score_clonal", "exp(coef) lower 95%"])
    coef_upper = float(cph.summary.loc["effect_score_clonal", "exp(coef) upper 95%"])
    coef_sub = float(cph.summary.loc["effect_score_subclonal", "exp(coef)"])
    pval_sub = float(cph.summary.loc["effect_score_subclonal", "p"])
    coef_lower_sub = float(cph.summary.loc["effect_score_subclonal", "exp(coef) lower 95%"])
    coef_upper_sub = float(cph.summary.loc["effect_score_subclonal", "exp(coef) upper 95%"])
    
    return {"coef": coef, "p_value": pval, "coef_lower": coef_lower, "coef_upper": coef_upper, 
            "coef_sub": coef_sub, "p_value_sub": pval_sub, "coef_lower_sub": coef_lower_sub, "coef_upper_sub": coef_upper_sub}, cph


mpl.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.linewidth": 1.0,
})

DATA = 'output/set_20kbp.protein_coding.gene_level_aggregated.dist_20000.tsv'
base_patient = 'base_patient_df.filtered_hyperMut20.filtered_PFIproj.tsv'
base_patient_df = pd.read_csv(base_patient, sep='\t')
print(f"Loaded base patient data with {len(base_patient_df)} rows")
cosmic_genes = pd.read_csv('/data/sokolova/cancer_lucaria/SUPP.used_cosmic_hallmark_genes.txt', header=None)[0].values
cosmic_sets = {
    'cosmic_hallmark': cosmic_genes,
}
print(f"Loaded {len(cosmic_genes)} COSMIC hallmark genes")
df = pd.read_csv(DATA, sep='\t')
print(f"Loaded aggregated data with {len(df)} rows")
df['REG_MAIN_CLONAL'] = df['MAX_CLONAL_MAX_ALL'] / df['NVAR_CLONAL'].replace(0, 1)
df['REG_MAIN_SUBCLONAL'] = df['MAX_SUBCLONAL_MAX_ALL'] / df['NVAR_SUBCLONAL'].replace(0, 1)
hallmark_genes = set(cosmic_genes)
df['is_hallmark'] = df['gene_name'].isin(hallmark_genes)
print("Hallmark rows:", df['is_hallmark'].sum())
plot_hallmark_clonal_subclonal_box(df)

use_id = 'sample_id'
f = l2_norm

results = {}
for set_name, genes in cosmic_sets.items():
    surv_df = compute_surv_df_for_gene_list(df, base_patient_df, genes, set_name, 
                                            zscore=True, 
                                            )
    # plot correlation heatmap
    fig, ax, corr = plot_corr_heatmap(
        surv_df,
        method="pearson",
        title=f"{set_name}: covariate correlations (z-scored within cancer)"
    )
    plt.show()
    _, cph = fit_cox(surv_df)
    # save cph summary 
    cph.summary.to_csv(f'SUPP.coxph_summary_clonal_subclonal.FINAL.tsv', sep='\t')
    cph.print_summary()
    cph.plot()
    print(f"HR for effect_score_clonal in {set_name}: {cph.hazard_ratios_['effect_score_clonal']:.3f}")
    print(f"HR for effect_score_subclonal in {set_name}: {cph.hazard_ratios_['effect_score_subclonal']:.3f}")
    cph.check_assumptions(surv_df, p_value_threshold=0.05, show_plots=True)

# PLOT NICE FOREST PLOT
summary = cph.summary.copy().reset_index().rename(columns={'covariate': 'variable'})
order = [
    'effect_score_clonal',
    'effect_score_subclonal',
    'nc_mutation_burden_clonal',
    'nc_mutation_burden_subclonal',
    'log_fga',
    'age_at_diagnosis',
    'non_silent_per_Mb'
]

summary = summary.set_index('variable')
summary = summary.loc[[v for v in order if v in summary.index]].reset_index()

base_labels = {
    'effect_score_clonal': 'Clonal regulatory score',
    'effect_score_subclonal': 'Subclonal regulatory score',
    'nc_mutation_burden_clonal': 'Clonal regulatory\nvariant count',
    'nc_mutation_burden_subclonal': 'Subclonal regulatory\nvariant count',
    'non_silent_per_Mb': 'Non-silent\nmutations (per Mb)',
    'age_at_diagnosis': 'Age at diagnosis',
    'log_fga': 'log FGA',
}
base_labels = [base_labels.get(v, v) for v in summary['variable'].values]

colors = ["#F4A261", "#FF8C00"] + ['grey']*(len(summary)-2)

stars = [p_to_stars(p) for p in summary['p'].values]
pretty_labels = [f"{lab} {star}".rstrip() for lab, star in zip(base_labels, stars)]

# log(HR) and log-CIs
hr = summary['exp(coef)'].values
ci_low = summary['exp(coef) lower 95%'].values
ci_hi  = summary['exp(coef) upper 95%'].values

y_pos = np.arange(len(pretty_labels))

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

fig, ax = plt.subplots(figsize=(4.1, 3.5), dpi=300)

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
xmin = float(np.nanmin(ci_low))
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

# print p-values in e-3 
for var, p in zip(summary['variable'], summary['p']):
    print(f"{var}: p = {p:.2e}")

# %%
