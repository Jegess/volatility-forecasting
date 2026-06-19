from pathlib import Path

import numpy as np
import polars as pl
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from scipy.stats import gaussian_kde

ETFS = {"SPY", "QQQ", "IWM", "GLD", "TLT"}


def load_returns_wide(
    underlying_dir: str | Path = "data/raw/underlying",
    min_overlap_days: int = 252,
) -> tuple[pl.DataFrame, list[str]]:
    """Load all symbol parquets, compute daily log returns, pivot to wide.

    Returns
    -------
    wide : polars DataFrame with column `date` and one float column per symbol
    symbols : list of symbols retained (those with >= min_overlap_days observations)
    """
    underlying_dir = Path(underlying_dir)
    files = sorted(underlying_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {underlying_dir}")

    frames = []
    for f in files:
        df = pl.read_parquet(f).select(["symbol", "date", "underlying_price"])
        df = df.sort("date").with_columns(
            (pl.col("underlying_price").log() - pl.col("underlying_price").shift(1).log())
            .alias("ret")
        ).drop_nulls("ret")
        df = df.filter(pl.col("ret").is_finite())
        df = df.filter(pl.col("ret").abs() < 0.5)
        df = df.select(["symbol", "date", "ret"])
        frames.append(df)

    long = pl.concat(frames)
    wide = long.pivot(values="ret", index="date", on="symbol").sort("date")

    symbols = [c for c in wide.columns if c != "date"]
    counts = {s: wide[s].drop_nulls().len() for s in symbols}
    kept = [s for s in symbols if counts[s] >= min_overlap_days]
    wide = wide.select(["date"] + kept)
    return wide, kept


def compute_correlation(
    wide: pl.DataFrame,
    symbols: list[str],
    min_pairwise: int = 252,
) -> tuple[np.ndarray, list[str]]:
    """Pairwise Pearson correlation with minimum-overlap filter.

    Pairs with fewer than `min_pairwise` overlapping observations get NaN,
    then any symbol with NaN rows is dropped to produce a clean square matrix.
    """
    arr = wide.select(symbols).to_numpy()  # shape (T, N)
    n = arr.shape[1]
    corr = np.full((n, n), np.nan, dtype=np.float64)

    valid = ~np.isnan(arr)
    for i in range(n):
        xi = arr[:, i]
        for j in range(i, n):
            xj = arr[:, j]
            mask = valid[:, i] & valid[:, j]
            k = mask.sum()
            if k < min_pairwise:
                continue
            a = xi[mask]
            b = xj[mask]
            a = a - a.mean()
            b = b - b.mean()
            denom = np.sqrt((a * a).sum() * (b * b).sum())
            if denom == 0:
                continue
            c = (a * b).sum() / denom
            corr[i, j] = c
            corr[j, i] = c

    np.fill_diagonal(corr, 1.0)
    keep = np.ones(n, dtype=bool)
    while True:
        sub = corr[np.ix_(keep, keep)]
        nan_count = np.isnan(sub).sum(axis=0)
        if nan_count.max() == 0:
            break
        worst_sub_idx = int(np.argmax(nan_count))
        keep_idx_list = np.where(keep)[0]
        keep[keep_idx_list[worst_sub_idx]] = False
    keep_idx = np.where(keep)[0]
    corr_clean = corr[np.ix_(keep_idx, keep_idx)]
    kept = [symbols[i] for i in keep_idx]
    return corr_clean, kept


def cluster_order(corr: np.ndarray, method: str = "average") -> np.ndarray:
    """Return permutation indices from hierarchical clustering on 1 - corr distance."""
    dist = 1.0 - corr
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    dist = np.clip(dist, 0.0, 2.0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method=method)
    return leaves_list(Z)


def _mp_pdf(var: float, q: float, pts: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Marchenko-Pastur theoretical PDF for variance `var` and ratio q = N/T."""
    lam_plus = var * (1 + np.sqrt(q)) ** 2
    lam_minus = var * (1 - np.sqrt(q)) ** 2
    lam = np.linspace(lam_minus, lam_plus, pts)
    pdf = np.sqrt((lam_plus - lam) * (lam - lam_minus)) / (2 * np.pi * var * q * lam)
    return lam, pdf


def _fit_mp_variance(eigvals: np.ndarray, q: float, bandwidth: float = 0.25) -> float:
    """Fit MP-implied variance by matching theoretical PDF to empirical KDE of eigenvalues."""
    kde = gaussian_kde(eigvals, bw_method=bandwidth)

    def sse(var_arr):
        var = float(var_arr[0])
        if var <= 0:
            return 1e9
        lam, pdf_th = _mp_pdf(var, q)
        pdf_emp = kde.evaluate(lam)
        return float(np.sum((pdf_emp - pdf_th) ** 2))

    res = minimize(sse, x0=np.array([0.5]), bounds=[(1e-4, 1.0)])
    return float(res.x[0])


def denoise_correlation(
    corr: np.ndarray, q: float, bandwidth: float = 0.25
) -> tuple[np.ndarray, dict]:
    """Lopez de Prado constant-residual-eigenvalue denoising.

    Parameters
    ----------
    corr : square correlation matrix (N, N)
    q : N / T ratio used during estimation of `corr`
    bandwidth : KDE bandwidth for fitting MP variance

    Returns
    -------
    corr_denoised : denoised correlation matrix (signal preserved, noise flattened)
    info : dict with keys var, lam_plus, n_signal, eigvals, eigvecs
    """
    eigvals, eigvecs = np.linalg.eigh(corr)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    var = _fit_mp_variance(eigvals, q, bandwidth=bandwidth)
    lam_plus = var * (1 + np.sqrt(q)) ** 2
    n_signal = int(np.sum(eigvals > lam_plus))
    if n_signal == 0:
        n_signal = 1  # always keep at least the market mode

    eigvals_new = eigvals.copy()
    if n_signal < len(eigvals):
        noise_mean = eigvals[n_signal:].mean()
        eigvals_new[n_signal:] = noise_mean

    C_tilde = eigvecs @ np.diag(eigvals_new) @ eigvecs.T
    d = np.sqrt(np.diag(C_tilde))
    corr_denoised = C_tilde / np.outer(d, d)
    np.fill_diagonal(corr_denoised, 1.0)

    info = {
        "var": var,
        "lam_plus": lam_plus,
        "n_signal": n_signal,
        "eigvals": eigvals,
        "eigvals_denoised": eigvals_new,
        "eigvecs": eigvecs,
    }
    return corr_denoised, info


def detone_correlation(
    corr_denoised: np.ndarray, n_market: int = 1
) -> tuple[np.ndarray, dict]:
    """Remove the top `n_market` eigenpairs (the market component) from a denoised matrix.

    WARNING: result is singular by construction — do NOT use for portfolio optimization.
    Intended for clustering, visualization, and revealing non-market structure.
    """
    eigvals, eigvecs = np.linalg.eigh(corr_denoised)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    W_M = eigvecs[:, :n_market]
    L_M = np.diag(eigvals[:n_market])
    C_tilde = corr_denoised - W_M @ L_M @ W_M.T
    d = np.sqrt(np.clip(np.diag(C_tilde), 1e-12, None))
    corr_detoned = C_tilde / np.outer(d, d)
    np.fill_diagonal(corr_detoned, 1.0)

    info = {
        "market_eigvals": eigvals[:n_market],
        "market_variance_fraction": float(eigvals[:n_market].sum() / eigvals.sum()),
    }
    return corr_detoned, info


def summary_stats(corr: np.ndarray, symbols: list[str], top_k: int = 10) -> dict:
    """Summary statistics for a correlation matrix."""
    iu = np.triu_indices_from(corr, k=1)
    off = corr[iu]
    pairs = [(symbols[iu[0][k]], symbols[iu[1][k]], off[k]) for k in range(len(off))]
    pairs_sorted = sorted(pairs, key=lambda x: x[2])
    return {
        "n_symbols": len(symbols),
        "n_pairs": len(off),
        "mean": float(off.mean()),
        "median": float(np.median(off)),
        "std": float(off.std()),
        "min": float(off.min()),
        "max": float(off.max()),
        "q25": float(np.quantile(off, 0.25)),
        "q75": float(np.quantile(off, 0.75)),
        "highest": pairs_sorted[-top_k:][::-1],
        "lowest": pairs_sorted[:top_k],
    }
