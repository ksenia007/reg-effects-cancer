# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import CoxPHFitter
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kruskal, mannwhitneyu
from matplotlib.gridspec import GridSpec
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from pandas import Series
from scipy.cluster.hierarchy import leaves_list

from utils import z_by_cancer, l2_norm, bh_fdr, make_safe_column_map, pretty_sei_name


def plot_event_rate(cluster_df: pd.DataFrame):
    event_summary = (
        cluster_df.groupby("cluster")
        .agg(
            n=("event", "size"),
            events=("event", "sum"),
            event_rate=("event", "mean"),
            median_time=("time", "median"),
        )
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.bar(event_summary.index.astype(str), event_summary["event_rate"])
    ax.set_xlabel("Regulatory cluster")
    ax.set_ylabel("Progression event rate")
    ax.set_ylim(0, event_summary["event_rate"].max() * 1.2)
    ax.grid(axis="y", alpha=0.2)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.show()
    return event_summary

# Compute sample-level scores per Sei class across hallmark genes
def compute_surv_df_for_sei_classes_list(
    df_gene: pd.DataFrame,
    base_patient_df: pd.DataFrame,
    tier_genes: list,
    sei_cols: list,
    var_count_column = 'NVAR_ALL',
    f=l2_norm,
    use_id="sample_id",
    prefix="effect_score__",
    drop_zero=True,
):
    tier_genes = list(set(tier_genes))
    N = len(tier_genes)
    if N == 0:
        raise ValueError("tier_genes empty")

    tier_subset = df_gene.loc[df_gene["gene_name"].isin(tier_genes)].copy()

    burden = tier_subset.groupby(use_id)[var_count_column].sum().rename("nc_mutation_burden")
    g = tier_subset.groupby(use_id)

    score_series = []
    for col in sei_cols:
        s = g[col].apply(lambda x: f(x.fillna(0.0).values, N))
        s.name = f"{prefix}{col}"
        score_series.append(s)

    effects = pd.concat(score_series + [burden], axis=1).reset_index().drop_duplicates()

    surv_df = base_patient_df.merge(effects, left_on="sample_id", right_on=use_id, how="left")

    if "case_id" in surv_df.columns:
        surv_df = surv_df.drop_duplicates(subset=["case_id"])

    score_cols = [c for c in surv_df.columns if c.startswith(prefix)]
    for c in score_cols:
        surv_df[c] = surv_df[c].fillna(0.0)
    surv_df["nc_mutation_burden"] = surv_df["nc_mutation_burden"].fillna(0.0)


    if drop_zero:
        surv_df = surv_df.loc[surv_df["nc_mutation_burden"] > 0.0].copy()
    
    # z-score within cancer
    for c in score_cols:
        surv_df[c] = z_by_cancer(surv_df, c)

    surv_df = surv_df.drop_duplicates()
    return surv_df, score_cols

def bootstrap_optimal_K(X, K_range=range(2, 11), n_boot=50, frac=0.8, seed=42):
    """
    Bootstrap to find optimal K by stability (ARI between bootstrap and full data)
    
    Returns DataFrame with mean/std for each metric at each K.
    """
    np.random.seed(seed)
    n = X.shape[0]
    
    results = {K: {"stability_ari": []} for K in K_range}
    
    # Full-data clustering for each K (reference for stability)
    full_labels = {}
    for K in K_range:
        link = linkage(pdist(X.values, metric="correlation"), method="average")
        full_labels[K] = fcluster(link, K, criterion="maxclust")
    
    for b in range(n_boot):
        # Bootstrap sample
        idx = np.random.choice(n, size=int(n * frac), replace=False)
        X_boot = X.iloc[idx]
        
        # Skip if too few samples
        if len(X_boot) < max(K_range) * 2:
            continue
        
        # Compute distance matrix once per bootstrap
        try:
            D_boot = pdist(X_boot.values, metric="correlation")
            if np.isnan(D_boot).any() or np.isinf(D_boot).any():
                continue
        except:
            continue
        
        link_boot = linkage(D_boot, method="average")
        
        for K in K_range:
            labels_boot = fcluster(link_boot, K, criterion="maxclust")
            full_labels_subset = full_labels[K][idx]
            ari = adjusted_rand_score(full_labels_subset, labels_boot)
            results[K]["stability_ari"].append(ari)
    
    # Summarize
    summary = []
    for K in K_range:
        row = {"K": K}
        for metric in ["stability_ari"]:
            vals = results[K][metric]
            if len(vals) > 0:
                row[f"{metric}_mean"] = np.mean(vals)
                row[f"{metric}_std"] = np.std(vals)
            else:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_std"] = np.nan
        summary.append(row)
    
    return pd.DataFrame(summary)


def relabel_clusters_by_size(clusters: np.ndarray, descending: bool = True) -> np.ndarray:
    """
    Re-label cluster IDs by cluster size.
    If descending=True: 1 = largest cluster.
    """
    vc = Series(clusters).value_counts(sort=False)          
    order = vc.sort_values(ascending=not descending).index  
    mapping = {old: new for new, old in enumerate(order, start=1)}
    return np.array([mapping[c] for c in clusters], dtype=int), mapping

def make_clusters_and_means(X: pd.DataFrame, surv_sei_safe: pd.DataFrame, K: int = 6):
    X_use = X.copy()
    row_link = linkage(pdist(X_use.values, metric="correlation"), method="average")
    clusters = fcluster(row_link, K, criterion="maxclust")

    clusters, size_map = relabel_clusters_by_size(clusters, descending=True)

    Xc = X_use.copy()
    Xc["cluster"] = clusters
    cluster_mean = Xc.groupby("cluster").mean()

    cluster_mean_z = (cluster_mean - cluster_mean.mean(axis=0)) / cluster_mean.std(axis=0).replace(0, np.nan)
    cluster_mean_z = cluster_mean_z.fillna(0.0)

    cluster_meta = (
        surv_sei_safe.loc[X_use.index, ["cancer_type"]]
        .assign(cluster=clusters)
        .groupby(["cluster", "cancer_type"])
        .size()
        .unstack(fill_value=0)
    )
    cluster_frac = cluster_meta.div(cluster_meta.sum(axis=1), axis=0)

    cluster_df = surv_sei_safe.loc[X_use.index, ["time", "event", "cancer_type", "sample_id"]].copy()
    cluster_df["cluster"] = clusters

    return clusters, cluster_mean_z, cluster_meta, cluster_frac, cluster_df, X_use, size_map


# Enrichment: KW then MWU cell stars
def enrichment_kw_then_mwu_cells(X_use: pd.DataFrame, clusters: np.ndarray, alpha: float = 0.05, tail: str = "greater"):
    Xmat = X_use.copy()
    Xmat["cluster"] = clusters

    classes = list(X_use.columns)
    clus_ids = np.sort(Xmat["cluster"].unique())

    p_kw = []
    for c in classes:
        groups = [Xmat.loc[Xmat["cluster"] == k, c].values for k in clus_ids]
        if np.allclose(np.var(np.concatenate(groups)), 0):
            p = 1.0
        else:
            _, p = kruskal(*groups)
        p_kw.append(p)

    p_kw = np.asarray(p_kw)
    q_kw = bh_fdr(p_kw)

    class_stats = pd.DataFrame({"seq_class": classes, "p_kw": p_kw, "q_kw": q_kw}).sort_values("q_kw")
    sig = class_stats.loc[class_stats["q_kw"] < alpha, "seq_class"].tolist()

    cell_p = pd.DataFrame(index=clus_ids, columns=sig, dtype=float)
    cell_eff = pd.DataFrame(index=clus_ids, columns=sig, dtype=float)

    for c in sig:
        for k in clus_ids:
            in_k = Xmat.loc[Xmat["cluster"] == k, c].values
            out_k = Xmat.loc[Xmat["cluster"] != k, c].values
            eff = np.nanmedian(in_k) - np.nanmedian(out_k)
            cell_eff.loc[k, c] = eff

            if len(in_k) < 5 or len(out_k) < 5 or (np.allclose(np.var(in_k), 0) and np.allclose(np.var(out_k), 0)):
                p = 1.0
            else:
                alt = "greater" if tail == "greater" else ("less" if tail == "less" else "two-sided")
                p = mannwhitneyu(in_k, out_k, alternative=alt).pvalue

            cell_p.loc[k, c] = p

    flat_q = bh_fdr(cell_p.values.flatten())
    cell_q = pd.DataFrame(flat_q.reshape(cell_p.shape), index=cell_p.index, columns=cell_p.columns)

    if tail == "greater":
        star = (cell_q < alpha) & (cell_eff > 0)
    elif tail == "less":
        star = (cell_q < alpha) & (cell_eff < 0)
    else:
        star = (cell_q < alpha)

    return class_stats, cell_q, star

def get_col_order_hierarchical(cluster_mean_z):
    """
    Order columns by hierarchical clustering of their profiles.
    """
    M = cluster_mean_z.copy()
    
    # Cluster columns (transpose so columns become rows)
    col_link = linkage(pdist(M.T.values, metric="correlation"), method="average")
    col_order_idx = leaves_list(col_link)
    
    return [M.columns[i] for i in col_order_idx]


# Heatmap + event fraction bar
def plot_cluster_heatmap_with_event_bar(
    cluster_mean_z: pd.DataFrame,
    cluster_df: pd.DataFrame,
    star_mask: pd.DataFrame | None = None,
    cmap: str = "RdYlBu_r",
    figsize=(13, 4),
):
    M = cluster_mean_z.copy().iloc[::-1]
    #col_order = sorted(M.columns.tolist(), key=pretty_sei_name)
    col_order = get_col_order_hierarchical(cluster_mean_z)
    if len(col_order) != 40:
        raise ValueError("Column order length mismatch")
    M = M[col_order]

    M_plot = M.copy()
    M_plot.columns = [pretty_sei_name(c) for c in M_plot.columns]

    v = np.nanmax(np.abs(M_plot.values))
    if not np.isfinite(v) or v == 0:
        v = 1.0

    ev = (
        cluster_df.groupby("cluster")["event"]
        .mean()
        .reindex(M.index)
        .fillna(0.0)
        .values
    )

    fig = plt.figure(figsize=figsize, dpi=250)
    gs = GridSpec(nrows=1, ncols=3, width_ratios=[0.02, 1.0, 0.18], wspace=0.15)

    ax_cbar = fig.add_subplot(gs[0, 0])
    ax_hm = fig.add_subplot(gs[0, 1])
    ax_event = fig.add_subplot(gs[0, 2], sharey=ax_hm)

    im = ax_hm.imshow(M_plot.values, aspect="auto", cmap=cmap, vmin=-v, vmax=v)

    ax_hm.set_yticks(np.arange(M_plot.shape[0]))
    ax_hm.set_yticklabels([str(i) for i in M_plot.index], rotation=90)
    ax_hm.set_ylabel("Regulatory cluster")

    ax_hm.set_xticks(np.arange(M_plot.shape[1]))
    ax_hm.set_xticklabels(M_plot.columns, rotation=90, ha="center")
    ax_hm.set_xlabel("")

    if star_mask is not None and star_mask.size > 0:
        star_aligned = star_mask.reindex(index=M.index, columns=M.columns).fillna(False)
        for i, cl in enumerate(M.index):
            for j, col in enumerate(M.columns):
                if bool(star_aligned.loc[cl, col]):
                    ax_hm.text(j, i, "*", ha="center", va="center", fontsize=12, color="black")

    cb = fig.colorbar(im, cax=ax_cbar)
    cb.set_label("z (across clusters)")
    ax_cbar.yaxis.set_ticks_position("left")
    ax_cbar.yaxis.set_label_position("left")
    # rotate colorbar ticks
    ax_cbar.set_yticklabels([f"{x:.1f}" for x in ax_cbar.get_yticks()], rotation=90)

    y = np.arange(len(M_plot.index))
    ax_event.barh(y, ev, color="darkred", edgecolor="none", height=0.9)
    ax_event.barh(y, 1 - ev, left=ev, color="lightgray", edgecolor="none", height=0.9)
    ax_event.set_xlim(1, 0)  
    ax_event.set_xticks([1.0, 0.5, 0.0])  
    ax_event.set_xlabel("Event fraction", rotation=90, labelpad=15)
    ax_event.tick_params(axis="y", left=False, labelleft=False)
    # rotate x ticks
    ax_event.set_xticklabels([f"{x:.1f}" for x in ax_event.get_xticks()], rotation=90)

    for ax in [ax_hm, ax_event]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.30, top=0.98)
    plt.show()

# Cancer composition plots
def plot_cluster_cancer_composition_topN(cluster_meta: pd.DataFrame, topN: int = 10, title: str | None = None):
    totals = cluster_meta.sum(axis=0).sort_values(ascending=False)
    keep = totals.head(topN).index.tolist()

    meta_keep = cluster_meta[keep].copy()
    meta_other = cluster_meta.drop(columns=keep).sum(axis=1).rename("Other")
    meta_plot = pd.concat([meta_keep, meta_other], axis=1)

    frac_plot = meta_plot.div(meta_plot.sum(axis=1), axis=0).fillna(0.0)
    frac_plot = frac_plot.rename(columns=lambda s: s.replace("TCGA-", ""))

    n_per_cluster = meta_plot.sum(axis=1).astype(int)

    fig, ax = plt.subplots(figsize=(6.2, 3.2), dpi=250)
    frac_plot.plot(kind="bar", stacked=True, ax=ax, colormap="tab20", width=0.85)

    ax.set_xlabel("Regulatory cluster")
    ax.set_ylabel("Fraction of samples")
    if title:
        ax.set_title(title)

    ymax = 1.0
    for i, (cl, n) in enumerate(n_per_cluster.items()):
        ax.text(i, ymax + 0.02, f"n={n}", ha="center", va="bottom", fontsize=11, clip_on=False)

    ax.set_ylim(0, 1.08)

    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
        title="Cancer type",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.15)
    fig.subplots_adjust(left=0.10, right=0.78, bottom=0.20, top=0.93)

    plt.show()

def gene_cluster_alignment_from_safe_centroid(
    df_gene,          # gene-level df (df_hm), with raw sei cols
    cluster_df,       # must have sample_id + cluster
    cluster_mean_z,   # cluster x SAFE score cols
    cluster_id,
    safe_score_cols,  # list of SAFE cols you want for centroid (e.g. sei_score_cols_safe)
    safe_map,         # original -> safe map (same one used to rename surv_sei)
):
    inv_safe_map = {v: k for k, v in safe_map.items()}  # safe -> original

    # map safe score cols -> raw gene-level sei cols
    orig_score_cols = [inv_safe_map[c] for c in safe_score_cols]  # "effect_score__SUM_ALL_*"
    gene_sei_cols   = [c.replace("effect_score__", "") for c in orig_score_cols]

    # centroid vector in the SAME order
    centroid = cluster_mean_z.loc[int(cluster_id), safe_score_cols].values

    in_samples = cluster_df.loc[cluster_df["cluster"] == int(cluster_id), "sample_id"]

    rows = []
    for gene, dfg in df_gene.groupby("gene_name"):
        g = dfg[dfg["sample_id"].isin(in_samples)]
        if len(g) == 0:
            continue

        v = g[gene_sei_cols].mean().values  # raw sei space, aligned to centroid

        if np.std(v) == 0 or np.std(centroid) == 0:
            corr = np.nan
        else:
            corr = float(np.corrcoef(v, centroid)[0, 1])

        rows.append({"gene_name": gene, "corr_to_cluster": corr})

    out = pd.DataFrame(rows).dropna().sort_values("corr_to_cluster", ascending=False)
    return out

def get_driver_gene_set(cluster_corr, cl, topN=100, corr_min=None):
    gt = cluster_corr[int(cl)].copy()

    if corr_min is not None:
        genes = gt.loc[gt["corr_to_cluster"] >= corr_min, "gene_name"]
    else:
        genes = gt.sort_values("corr_to_cluster", ascending=False).head(topN)["gene_name"]

    return set(genes.astype(str).tolist())

def plot_overlap_heatmap(cluster_corr, clusters=None, topN=80, corr_min=None, metric="jaccard"):
    if clusters is None:
        clusters = sorted(cluster_corr.keys())

    sets = {cl: get_driver_gene_set(cluster_corr, cl, topN=topN, corr_min=corr_min) for cl in clusters}

    M = pd.DataFrame(index=clusters, columns=clusters, dtype=float)

    for i in clusters:
        for j in clusters:
            A, B = sets[i], sets[j]
            inter = len(A & B)
            if metric == "jaccard":
                denom = len(A | B) if len(A | B) else 1
                val = inter / denom
            elif metric == "overlap_coef":
                denom = min(len(A), len(B)) if min(len(A), len(B)) else 1
                val = inter / denom
            else:  # raw intersection size
                val = inter
            M.loc[i, j] = val
    return M

def plot_cluster_overlap_heatmap(
    M: pd.DataFrame,                     # square DataFrame (cluster x cluster)
    title: str = None,
    metric_label: str = "Jaccard",
    cmap: str = "YlOrRd",
    vmin: float = 0.0,
    vmax: float = None,
    annotate: bool = True,
    fmt: str = ".2f",
    figsize=(5.2, 4.6),
    tick_fontsize: int = 18,
    annot_fontsize: int = 14,
    grid_color: str = "white",
    grid_lw: float = 1.0,
    cluster_label_map: dict | None = None,  # {cluster_id: "C1"} etc
):
    """
    M must have identical index/columns in same order.
    """

    # ensure numeric matrix
    corr = M.copy().astype(float)

    # set vmax if not provided
    if vmax is None:
        vmax = float(np.nanmax(corr.values)) if np.isfinite(np.nanmax(corr.values)) else 1.0

    cols = list(corr.columns)

    fig, ax = plt.subplots(figsize=figsize, dpi=250)
    im = ax.imshow(corr.values, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")

    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    if title:
        ax.set_title(title)

    # colorbar like your style
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric_label, fontsize=13)
    # increase colorbar tick fontsize
    cbar.ax.tick_params(labelsize=12)

    # pretty cluster labels
    if cluster_label_map is None:
        pretty = [str(c) for c in cols]
    else:
        pretty = [cluster_label_map.get(c, str(c)) for c in cols]

    ax.set_xticklabels(pretty, rotation=0, ha="right", fontsize=tick_fontsize)
    ax.set_yticklabels(pretty, fontsize=tick_fontsize)

    # Optional cell annotations
    if annotate:
        for i in range(len(cols)):
            for j in range(len(cols)):
                val = corr.values[i, j]
                if np.isnan(val):
                    txt = "NA"
                else:
                    txt = format(val, fmt)
                ax.text(j, i, txt, ha="center", va="center", fontsize=annot_fontsize)

    # gridlines between cells (minor ticks)
    ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(cols), 1), minor=True)
    ax.grid(which="minor", color=grid_color, linestyle="-", linewidth=grid_lw)
    ax.tick_params(which="minor", bottom=False, left=False)
    
    # x and y axis labels
    ax.set_xlabel("Cluster", fontsize=17)
    ax.set_ylabel("Cluster", fontsize=17)

    # minimal spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    plt.show()

####### ------------------

DATA = 'output/set_20kbp.protein_coding.gene_level_aggregated.dist_20000.tsv'
BASE_PATIENT = "base_patient_df.filtered_hyperMut20.filtered_PFIproj.tsv"
COSMIC_CSV = "resources/cosmic.07142025.21_22_13.csv"
MODE = 'all'

base_patient_df = pd.read_csv(BASE_PATIENT, sep="\t").rename(columns={"sample_id_x": "sample_id"})
print(f"Loaded base patient data with {len(base_patient_df)} rows")
df_raw = pd.read_csv(DATA, sep="\t")
print(f"Loaded aggregated data with {len(df_raw)} rows")

hallmark_genes = pd.read_csv('used_cosmic_hallmark_genes.txt', header=None)[0].values
cosmic_sets = {
    'cosmic_hallmark': hallmark_genes,
}
print(f"Loaded {len(hallmark_genes)} COSMIC hallmark genes")

# -----------------------------
# Hallmark-only gene-level table + normalize Sei cols ONCE
# -----------------------------
df_hm = df_raw.copy()
df_hm["is_hallmark"] = df_hm["gene_name"].isin(hallmark_genes)
df_hm = df_hm.loc[df_hm["is_hallmark"]].copy()
print("Hallmark rows:", int(df_hm.shape[0]))

if MODE == 'all':
    sei_cols = df_hm.columns[43:83]  
    print(f"SEI columns: {sei_cols.tolist()}") 
    if 'MAX' not in sei_cols[0]:
        raise ValueError("Expected SEI columns to start with 'MAX_ALL_'")
    df_hm[sei_cols] = df_hm[sei_cols].div(df_hm["NVAR_ALL"].replace(0, np.nan), axis=0).fillna(0.0)
else: 
    raise ValueError(f"Unknown MODE: {MODE}")

surv_sei, sei_score_cols = compute_surv_df_for_sei_classes_list(
    df_gene=df_hm,
    base_patient_df=base_patient_df,
    tier_genes=hallmark_genes,
    sei_cols=sei_cols,
    drop_zero=True,
)

sei_safe_map = make_safe_column_map(sei_score_cols)
safe_map = {**sei_safe_map}
surv_sei_safe = surv_sei.rename(columns=safe_map).copy()
sei_score_cols_safe = [safe_map[c] for c in sei_score_cols]
print(f"Number of samples with non-zero mutations in hallmark genes: {len(surv_sei_safe)}")
print(f"Number of Sei score columns: {len(sei_score_cols_safe)}")

# X matrix for clustering
X0 = surv_sei_safe[sei_score_cols_safe].copy()

col_order = sorted(sei_score_cols_safe, key=pretty_sei_name)
X0 = X0[col_order]
X0 = X0.fillna(0.0)

print(f"Shape of X0 for clustering: {X0.shape}")
X0 = (X0 - X0.mean(axis=1).values[:, None]) / (
    X0.std(axis=1).replace(0, np.nan).values[:, None]
)
X0 = X0.fillna(0.0)
print(f"After row-normalization, X0 mean: {X0.values.mean():.4f}, std: {X0.values.std():.4f}")

# check for NaN
if np.isnan(X0.values).any():
    raise ValueError("X0 contains NaN values after normalization")  

zero_rows = (X0.abs().sum(axis=1) == 0)
print(f"Rows that are all zeros: {zero_rows.sum()}")
# remove zero rows
X0 = X0.loc[~zero_rows].copy()
print(f"Shape of X0 after removing zero rows: {X0.shape}")


# -----------------------------
# Run bootstrap
print("Running bootstrap K selection...")
boot_df = bootstrap_optimal_K(X0, K_range=range(2, 8), n_boot=100, frac=0.8)
print(boot_df)

# Plot results
fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=250)
ax.errorbar(boot_df["K"], boot_df["stability_ari_mean"], yerr=boot_df["stability_ari_std"], 
            marker="o", capsize=3, color="tomato")
ax.set_xlabel("Number of clusters (K)")
ax.set_ylabel("Stability (ARI)")
ax.set_title("Bootstrap stability (higher = more reproducible)")
ax.grid(alpha=0.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.show()

K = 6
clusters_assigned, cluster_mean_z, cluster_meta, cluster_frac, cluster_df, X_use, size_map = make_clusters_and_means(
    X0, surv_sei_safe, K=6
)
print("Old->new label mapping:", size_map)
print("Cluster sizes (new labels):")
print(cluster_df["cluster"].value_counts().sort_index())
cluster_df.to_csv('regulatory_clusters_samples.FINAL_MAX_6cl.tsv', sep="\t", index=False)

alpha = 0.05
class_stats, cell_q, star_mask = enrichment_kw_then_mwu_cells(X_use, clusters_assigned, alpha=alpha, tail="greater")
print("Top enriched classes by overall KW q-value:")
print(class_stats.head(10))

plot_cluster_heatmap_with_event_bar(
    cluster_mean_z=cluster_mean_z,
    cluster_df=cluster_df,
    star_mask=star_mask,
    cmap="RdYlBu_r",
    figsize=(13, 4),
)

plot_cluster_cancer_composition_topN(cluster_meta, topN=10)
clusters = cluster_df["cluster"].unique()
cluster_corr = {}
for cl in cluster_df["cluster"].unique():
    safe_cols = star_mask.columns[star_mask.loc[cl]].values
    if len(safe_cols) == 0:
        continue

    gt = gene_cluster_alignment_from_safe_centroid(
        df_gene=df_hm,
        cluster_df=cluster_df,
        cluster_mean_z=cluster_mean_z,
        cluster_id=cl,
        safe_score_cols=safe_cols,
        safe_map=safe_map
    )

    gt = gt.assign(cluster=int(cl), n_star_classes=len(safe_cols))
    cluster_corr[int(cl)] = gt



M = plot_overlap_heatmap(cluster_corr, topN=50, metric="jaccard")
plot_cluster_overlap_heatmap(
    M,
    title="",
    metric_label="Jaccard overlap",
    cmap="YlOrRd",
    annotate=True,
    fmt=".2f",
    figsize=(6.0, 5.2),
)

# for each of the cluster, make top 50 into a dataframe (one column is cluster id, one column is gene list)
cluster_top = {}
for cl in cluster_corr.keys():
    genes = get_driver_gene_set(cluster_corr, cl, topN=50, corr_min=None)
    cluster_top[cl] = ','.join(genes)

cluster_top = pd.DataFrame.from_dict(cluster_top, orient='index', columns=['top50_genes'])
cluster_top.to_csv('results/cluster_top50_genes.MAX.FINAL.csv')

# %%
