import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.stats import mannwhitneyu
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
grey_red = LinearSegmentedColormap.from_list("grey_red", ["#bdbdbd", "#d7301f"])
base = mpl.cm.get_cmap('YlOrRd')
cmap_dark = LinearSegmentedColormap.from_list('YlOrRd_dark', base(np.linspace(0.25, 1.0, 256)))
from utils import cliffs_delta


def p_to_stars(p):
    if p < 1e-4: return "****"
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 5e-2: return "*"
    return "ns"


def plot_hallmark_clonal_subclonal_box(
    df,
    is_hallmark_col="is_hallmark",
    sample_col="sample_id",
    gene_col="gene_name",
    value_cols=("REG_MAIN_CLONAL", "REG_MAIN_SUBCLONAL"),
    y_label="Regulatory score per gene",
    figsize=(4.1, 4.2),
    dpi=300,
    base_fontsize=14,
    showfliers=False,
    box_width=0.55,
    show_p_text=False,   # keep plot clean by default
):
    # -----------------------------
    # Filter + counts (printed)
    # -----------------------------
    d0 = df[df[is_hallmark_col]].copy()
    n_samples = int(d0[sample_col].nunique()) if sample_col in d0.columns else None
    n_genes = int(d0[gene_col].nunique()) if gene_col in d0.columns else None

    # -----------------------------
    # Long-form
    # -----------------------------
    long = d0.melt(
        id_vars=[c for c in [sample_col, gene_col] if c in d0.columns],
        value_vars=list(value_cols),
        var_name="Clonality",
        value_name=y_label,
    ).dropna(subset=[y_label])

    mapping = {
        value_cols[0]: "Clonal",
        value_cols[1]: "Subclonal",
        "REG_MAIN_CLONAL": "Clonal",
        "REG_MAIN_SUBCLONAL": "Subclonal",
    }
    long["Clonality"] = long["Clonality"].map(lambda x: mapping.get(x, str(x)))
    order = ["Clonal", "Subclonal"]
    order = [o for o in order if o in set(long["Clonality"])]

    # -----------------------------
    # Stats (printed)
    # -----------------------------
    a = long.loc[long["Clonality"] == "Clonal", y_label].values
    b = long.loc[long["Clonality"] == "Subclonal", y_label].values
    n_a, n_b = len(a), len(b)

    pval = np.nan
    if n_a > 0 and n_b > 0:
        _, pval = mannwhitneyu(a, b, alternative="two-sided")

    delta = cliffs_delta(a, b)
    med_a = np.nanmedian(a) if n_a else np.nan
    med_b = np.nanmedian(b) if n_b else np.nan

    if np.isnan(med_a) or np.isnan(med_b):
        direction = "NA"
    elif med_a > med_b:
        direction = "Clonal > Subclonal"
    elif med_b > med_a:
        direction = "Subclonal > Clonal"
    else:
        direction = "Equal medians"

    stars = p_to_stars(pval) if not np.isnan(pval) else "NA"

    def fmt_p(p):
        if np.isnan(p): return "NA"
        if p < 1e-4: return "<1e-4"
        return f"{p:.2e}"

    # Print summary (NOT on plot)
    print("=== Hallmark clonal vs subclonal severity (unpaired) ===")
    if n_samples is not None: print(f"N samples: {n_samples}")
    if n_genes is not None:   print(f"N genes:   {n_genes}")
    print(f"Observations: clonal n={n_a}, subclonal n={n_b}")
    print(f"Median: clonal={med_a:.4g}, subclonal={med_b:.4g}  -> {direction}")
    print(f"Mann–Whitney U p={fmt_p(pval)}   ({stars})")
    print(f"Raw p-value: {pval}")
    print(f"Cliff's delta (clonal - subclonal): {delta:+.3f}")

    # -----------------------------
    # Plot
    # -----------------------------
    palette = {"Clonal": "#F4A261", "Subclonal": "#FF8C00"}  # orangeade, darkorange

    sns.set_theme(style="white", context="paper")
    plt.rcParams.update({
        "font.size": base_fontsize,
        "axes.labelsize": base_fontsize + 1,
        "xtick.labelsize": base_fontsize,
        "ytick.labelsize": base_fontsize,
    })

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    sns.boxplot(
        data=long,
        x="Clonality",
        y=y_label,
        order=order,
        palette=palette,
        showfliers=showfliers,
        width=box_width,
        linewidth=1.2,
        ax=ax,
    )

    # remove legend created by hue
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()

    # Clean spines + no title
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel(y_label)

    # -----------------------------
    # Significance bar + stars + Cliff's delta 
    # -----------------------------
    if len(order) == 2 and n_a > 0 and n_b > 0:
        y_max = 5
        y_min = -0.5
        y_range = (y_max - y_min) if np.isfinite(y_max - y_min) and (y_max > y_min) else 1.0

        bar_y = y_max + 0.06 * y_range
        bar_h = 0.02 * y_range
        x1, x2 = 0, 1

        ax.plot([x1, x1, x2, x2], [bar_y, bar_y + bar_h, bar_y + bar_h, bar_y],
                lw=1.4, c="black")

        # stars
        ax.text((x1 + x2) / 2, bar_y + bar_h + 0.01 * y_range, stars,
                ha="center", va="bottom", fontsize=base_fontsize + 1)

        # Cliff's delta near the bar
        delta_txt = "Cliff’s δ = NA" if np.isnan(delta) else f"Cliff’s δ = {delta:+.2f}"
        ax.text((x1 + x2) / 2, bar_y + bar_h - 0.08 * y_range, delta_txt,
                ha="center", va="bottom", fontsize=base_fontsize - 1)

        # optional p text (off by default)
        if show_p_text:
            ax.text((x1 + x2) / 2, bar_y + bar_h + 0.085 * y_range, f"p={fmt_p(pval)}",
                    ha="center", va="bottom", fontsize=base_fontsize - 2)

        ax.set_ylim(y_min, bar_y + bar_h + (0.14 * y_range))

    # dashed horizontal line at 0
    ax.axhline(0, color="grey", linestyle="--", linewidth=1.0, alpha=0.7)
    plt.tight_layout()
    plt.show()

    return {
        "p_value": None if np.isnan(pval) else float(pval),
        "stars": stars,
        "cliffs_delta": None if np.isnan(delta) else float(delta),
        "direction_by_median": direction,
        "median_clonal": None if np.isnan(med_a) else float(med_a),
        "median_subclonal": None if np.isnan(med_b) else float(med_b),
        "n_samples": n_samples,
        "n_genes": n_genes,
        "n_obs_clonal": int(n_a),
        "n_obs_subclonal": int(n_b),
    }
    


def plot_corr_heatmap_clonal(
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



def plot_corr_heatmap(
    surv_df: pd.DataFrame,
    cols=None,
    method: str = "pearson",   
    title: str = None,
    annotate: bool = True,
    figsize=(7, 6),
    vmin: float = 0,
    vmax: float =  1.0,
):
    """
    Plot a correlation matrix heatmap for selected columns in surv_df.
    Uses matplotlib only (no seaborn).
    """
    if cols is None:
        cols = [
            "effect_score",
            "nc_mutation_burden",
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
    im = ax.imshow(corr.values, vmin=vmin, vmax=vmax, cmap='Reds')

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
    "effect_score": "Regulatory score",
    "nc_mutation_burden": "Regulatory\nvariant count",
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


