import re
import numpy as np
from lifelines import CoxPHFitter
from matplotlib.colors import LinearSegmentedColormap
grey_red = LinearSegmentedColormap.from_list("grey_red", ["#bdbdbd", "#d7301f"])
import matplotlib as mpl
import matplotlib.ticker as mticker
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, rankdata


def z_by_cancer(df, col):
    return df.groupby('cancer_type')[col] .transform(lambda x: (x - x.mean()) /( x.std()+1e-10) ) 

def cliffs_delta(x, y):
    """
    Efficient Cliff's delta using ranks.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n1 = x.size
    n2 = y.size
    all_vals = np.concatenate([x, y])
    ranks = rankdata(all_vals)
    r1 = ranks[:n1].sum()
    U1 = r1 - n1 * (n1 + 1) / 2.0
    delta = (2 * U1 / (n1 * n2)) - 1
    return delta


def l2_norm(x, N):
    return np.sqrt(np.sum(x**2) / N)

def compute_surv_df_for_gene_list(
    df, base_patient_df, tier_genes, set_name,
    f=l2_norm, use_id="sample_id", 
    zscore=True
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
            "MAIN_REG_SCORE": lambda x: f(x, N),
            "NVAR_ALL": "sum",
        })
        .rename(columns={
            "MAIN_REG_SCORE": "effect_score",
            "NVAR_ALL": "nc_mutation_burden",
        })
        .reset_index()
        .drop_duplicates()
    )

    surv_df = base_patient_df.merge(tier_effects, left_on="sample_id", right_on=use_id, how="left")

    surv_df = surv_df.drop_duplicates(subset=["case_id"])

    surv_df["effect_score"] = surv_df["effect_score"].fillna(0.0)
    surv_df["nc_mutation_burden"] = surv_df["nc_mutation_burden"].fillna(0.0)

    # z-score within cancer
    if zscore:
        surv_df["effect_score"] = z_by_cancer(surv_df, "effect_score")
        surv_df["log_fga"] = z_by_cancer(surv_df, "log_fga")
        surv_df["non_silent_per_Mb"] = z_by_cancer(surv_df, "non_silent_per_Mb")
        surv_df["nc_mutation_burden"] = z_by_cancer(surv_df, "nc_mutation_burden")
        surv_df["age_at_diagnosis"] = z_by_cancer(surv_df, "age_at_diagnosis")
        

    surv_df = (
        surv_df.dropna(subset=[
            "time", "event",
            "non_silent_per_Mb", "log_fga", "age_at_diagnosis",
            "effect_score", "nc_mutation_burden", 
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
        nc_mutation_burden +
        effect_score
    """
    cph = CoxPHFitter() 
    cph.fit(
        surv_df,
        duration_col="time",
        event_col="event",
        formula=formula,
        strata=["cancer_type"],
    )
    coef = float(cph.summary.loc["effect_score", "exp(coef)"])
    pval = float(cph.summary.loc["effect_score", "p"])
    coef_lower = float(cph.summary.loc["effect_score", "exp(coef) lower 95%"])
    coef_upper = float(cph.summary.loc["effect_score", "exp(coef) upper 95%"])
    return {"coef": coef, "p_value": pval, "coef_lower": coef_lower, "coef_upper": coef_upper}, cph


def add_group_from_continuous(surv_df, col):
    s = surv_df[col].astype(float)
    thr_low = float(s.quantile(0.05))
    thr_high = float(s.quantile(0.95))
    print(f"Cutoff at quantiles: thresholds={thr_low:.3g}/{thr_high:.3g}")
    grp = np.where(s <= thr_low, "Low", np.where(s >= thr_high, "High", "Mid"))
    surv_df_new = surv_df.assign(group=grp)
    surv_df_new = surv_df_new[surv_df_new['group'] != "Mid"].copy()
    return surv_df_new, {"thresholds": (thr_low, thr_high)}



def plot_km_by_group_pretty(
    surv_df,
    group_col="group",
    title=None,
    x_max=4000,
    base_fontsize=18,
    legend_title=None,
):
    # ---------- p-value ----------
    groups = surv_df[group_col].astype(str)

    if groups.nunique() == 2:
        g1, g2 = groups.unique()
        d1 = surv_df[groups == g1]
        d2 = surv_df[groups == g2]
        print(f"Number in group {g1}: {len(d1)}, group {g2}: {len(d2)}")
        print(f"Events in group {g1}: {d1['event'].sum()}, group {g2}: {d2['event'].sum()}")
        res = logrank_test(
            d1["time"], d2["time"],
            event_observed_A=d1["event"],
            event_observed_B=d2["event"],
        )
        logrank_p = float(res.p_value)
    else:
        raise ValueError("plot_km_by_group_pretty only supports 2 groups for logrank test")

    def p_text(p):
        return f"Log-rank p = {p:.2e}"

    # ---------- group order ----------
    def try_float(x):
        try:
            return float(x)
        except Exception:
            return None

    vals = {g: try_float(g) for g in groups.unique()}
    if all(v is not None for v in vals.values()):
        ordered_groups = sorted(vals, key=lambda g: vals[g])
    else:
        ordered_groups = sorted(groups.unique())
        print(ordered_groups)

    # ---------- colors: grey -> dark red ----------
    cmap = LinearSegmentedColormap.from_list(
        "grey_to_darkred", ["#BDBDBD", "#7F0000"]
    ).reversed()

    colors = {
        g: cmap(i / max(len(ordered_groups) - 1, 1))
        for i, g in enumerate(ordered_groups)
    }

    # ---------- plot ----------
    plt.rcParams.update({
        "font.size": base_fontsize,
        "axes.titlesize": base_fontsize + 2,
        "axes.labelsize": base_fontsize + 1,
        "legend.fontsize": base_fontsize,
    })

    fig, ax = plt.subplots(figsize=(7.6, 6.6), dpi=180)

    for g in ordered_groups:
        d = surv_df[groups == g]
        kmf = KaplanMeierFitter()
        kmf.fit(d["time"], event_observed=d["event"], label=str(g))
        kmf.plot_survival_function(
            ax=ax,
            ci_show=True,
            linewidth=2.3,
            color=colors[g],
        )

    # axes styling
    ax.set_xlim(0, x_max)
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability")

    if title:
        ax.set_title(title, pad=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=base_fontsize)

    # legend
    ax.legend(
        title=legend_title if legend_title else group_col,
        frameon=False,
        loc="lower left",
    )

    # p-value annotation
    ax.text(
        0.98, 0.78,
        p_text(logrank_p),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=base_fontsize,
    )

    fig.tight_layout()
    plt.show()

    return {"logrank_p": logrank_p, "n_groups": groups.nunique()}


# -------- LEAVE GENES out ----------
def gene_dropout_stability(
    df, base_patient_df, tier_genes, set_name,
    drop_frac=0.10, n_reps=200, seed=0,
):
    rng = np.random.default_rng(seed + int(drop_frac * 100))  # different seed per frac

    tier_genes = list(set(tier_genes))
    N = len(tier_genes)
    if N < 10:
        raise ValueError(f"{set_name}: too few genes ({N}) for dropout={drop_frac}")

    # --- Full baseline
    full_surv = compute_surv_df_for_gene_list(df, base_patient_df, tier_genes, set_name)
    full_fit, _ = fit_cox(full_surv)
    # --- Replicates
    k_drop = int(np.round(drop_frac * N))
    rep_rows = {}
    rep_rows['baseline'] = full_fit

    for r in range(n_reps):
        drop_genes = set(rng.choice(tier_genes, size=k_drop, replace=False))
        keep_genes = [g for g in tier_genes if g not in drop_genes]
        rep_surv = compute_surv_df_for_gene_list(df, base_patient_df, keep_genes, set_name)
        rep_fit, _ = fit_cox(rep_surv)
        rep_rows[r] = rep_fit
        
    return rep_rows


# -------- LEAVE RANDOM SAMPLES out ----------
def sample_dropout_stability(
    df, base_patient_df, tier_genes, set_name,
    drop_frac=0.10, n_reps=10, seed=0,
    use_id="sample_id",
    recompute_surv=True,   
    return_dropped=False,  
):
    
    rng = np.random.default_rng(seed + int(drop_frac * 100))  # different seed per frac

    # cohort = unique sample IDs available in the patient table
    all_ids = np.array(sorted(base_patient_df[use_id].dropna().unique().tolist()))
    n_all = len(all_ids)
    if n_all < 10:
        raise ValueError(f"{set_name}: too few samples ({n_all}) for dropout={drop_frac}")

    k_drop = int(np.round(drop_frac * n_all))
    if k_drop <= 0:
        k_drop = 1
    if k_drop >= n_all:
        raise ValueError(f"{set_name}: drop_frac too large (would drop {k_drop}/{n_all} samples)")

    # --- Full baseline
    full_surv = compute_surv_df_for_gene_list(df, base_patient_df, tier_genes, set_name, use_id=use_id)
    full_fit, _ = fit_cox(full_surv)

    rep_rows = {"baseline": full_fit}
    dropped_ids = {}  # optional

    # If you *don't* want to recompute effect_score/scaling each time, precompute once and just subset
    if not recompute_surv:
        raise NotImplementedError("Currently only recompute_surv=True is implemented")

    # --- Recompute surv_df per replicate (safer if compute_surv_df... does z-scaling, etc.)
    for r in range(n_reps):
        drop = set(rng.choice(all_ids, size=k_drop, replace=False))

        base_rep = base_patient_df[~base_patient_df[use_id].isin(drop)].copy()
        df_rep = df[~df[use_id].isin(drop)].copy()  # assumes df has the same use_id column

        rep_surv = compute_surv_df_for_gene_list(df_rep, base_rep, tier_genes, set_name, use_id=use_id)
        rep_fit, _ = fit_cox(rep_surv)
        rep_rows[r] = rep_fit
        if return_dropped:
            dropped_ids[r] = sorted(drop)

    return (rep_rows, dropped_ids) if return_dropped else rep_rows



def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    n = pvals.size
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return out

def make_safe_column_map(cols):
    safe = []
    used = set()
    for c in cols:
        s = re.sub(r"[^0-9a-zA-Z_]+", "_", c)
        s = re.sub(r"_+", "_", s).strip("_")
        if re.match(r"^[0-9]", s):
            s = "c_" + s
        base = s
        k = 1
        while s in used:
            k += 1
            s = f"{base}_{k}"
        used.add(s)
        safe.append(s)
    return dict(zip(cols, safe))

def pretty_sei_name(col: str) -> str:
    s = col
    s = s.replace("effect_score__", "")
    s = s.replace("effect_score_", "")
    s = s.replace("SUM_ALL_", "")
    s = s.replace("MAX_ALL_", "")
    s = s.replace("_", " ").strip()
    return s

