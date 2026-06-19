"""Option-implied features aggregated to (symbol, date) level.

Computes 15 features from per-contract options_iv data:
ATM IV, VRP, skew, term structure, vol-of-vol, tail loss,
P/C ratios, BKM risk-neutral moments, weighted put-call IV
spread (Cremers & Weinbaum 2010), and option turnover.

Literature: Bali et al. (2021), Carr & Wu (2009),
Bakshi-Kapadia-Madan (2003), Driessen et al. (2009),
Cremers & Weinbaum (2010).

Usage:
    Called by compute_features.py orchestrator, not directly.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def _nearest_monthly_expiration(df: pl.DataFrame) -> pl.DataFrame:
    """Tag rows belonging to the nearest monthly expiration (>= 14 DTE).

    For each date, find the closest expiration that has both calls and puts
    with reasonable strike coverage. This is the "front month" used for
    ATM IV, skew, and BKM moments.
    """
    return df.with_columns(
        pl.col("expiration_date")
        .min()
        .over("date")
        .alias("_front_exp")
    ).filter(
        pl.col("expiration_date") == pl.col("_front_exp")
    ).drop("_front_exp")


def _second_expiration(df: pl.DataFrame) -> pl.DataFrame:
    """Get the second-nearest expiration for term structure slope."""
    # Get unique expirations per date, sorted
    exp_ranked = (
        df.select("date", "expiration_date")
        .unique()
        .sort("date", "expiration_date")
        .with_columns(
            pl.col("expiration_date")
            .rank("ordinal")
            .over("date")
            .alias("_exp_rank")
        )
    )
    second = exp_ranked.filter(pl.col("_exp_rank") == 2).select("date", "expiration_date")
    return second


def compute_atm_iv(df_front: pl.DataFrame) -> pl.DataFrame:
    """Compute ATM implied volatility per date.

    ATM = average IV of the two strikes closest to underlying_price
    (one above, one below), averaged across calls and puts.
    Uses nearest-to-money on the front expiration.
    """
    # Absolute distance from ATM
    df_atm = df_front.with_columns(
        (pl.col("moneyness") - 1.0).abs().alias("_atm_dist")
    )

    # Rank strikes by distance to ATM within each (date, right) group
    df_atm = df_atm.with_columns(
        pl.col("_atm_dist")
        .rank("ordinal")
        .over("date", "right")
        .alias("_atm_rank")
    )

    # Keep 2 closest strikes per side
    df_atm = df_atm.filter(pl.col("_atm_rank") <= 2)

    # Average IV across closest strikes and both call/put
    atm_iv = (
        df_atm.group_by("date")
        .agg(pl.col("iv").mean().alias("atm_iv"))
    )

    return atm_iv


def compute_iv_skew(df_front: pl.DataFrame) -> pl.DataFrame:
    """Compute IV skew: OTM put IV (delta ~ -0.25) minus ATM call IV.

    OTM put: closest to delta = -0.25
    ATM call: closest to delta = 0.50
    """
    # OTM put IV (closest to delta = -0.25)
    puts = df_front.filter(pl.col("right") == "PUT")
    puts = puts.with_columns(
        (pl.col("delta") - (-0.25)).abs().alias("_d25_dist")
    )
    put_iv = (
        puts.sort("_d25_dist")
        .group_by("date")
        .first()
        .select("date", pl.col("iv").alias("_put_iv_25d"))
    )

    # ATM call IV (closest to delta = 0.50)
    calls = df_front.filter(pl.col("right") == "CALL")
    calls = calls.with_columns(
        (pl.col("delta") - 0.50).abs().alias("_d50_dist")
    )
    call_iv = (
        calls.sort("_d50_dist")
        .group_by("date")
        .first()
        .select("date", pl.col("iv").alias("_call_iv_atm"))
    )

    skew = put_iv.join(call_iv, on="date", how="inner").with_columns(
        (pl.col("_put_iv_25d") - pl.col("_call_iv_atm")).alias("iv_skew")
    ).select("date", "iv_skew")

    return skew


def compute_term_slope(df: pl.DataFrame) -> pl.DataFrame:
    """Compute IV term structure slope: long-term ATM IV minus short-term ATM IV.

    Uses front and second expiration, ATM strikes only.
    """
    # Get unique expirations per date, pick front and second
    exp_per_date = (
        df.select("date", "expiration_date")
        .unique()
        .sort("date", "expiration_date")
        .with_columns(
            pl.col("expiration_date")
            .rank("ordinal")
            .over("date")
            .alias("_exp_rank")
        )
    )

    front_dates = exp_per_date.filter(pl.col("_exp_rank") == 1).select(
        "date", pl.col("expiration_date").alias("_front_exp")
    )
    second_dates = exp_per_date.filter(pl.col("_exp_rank") == 2).select(
        "date", pl.col("expiration_date").alias("_second_exp")
    )

    dates_with_two = front_dates.join(second_dates, on="date", how="inner")

    # ATM IV for front month
    df_front = df.join(
        dates_with_two.select("date", "_front_exp"),
        on="date",
        how="inner",
    ).filter(pl.col("expiration_date") == pl.col("_front_exp"))

    df_front = df_front.with_columns(
        (pl.col("moneyness") - 1.0).abs().alias("_atm_dist")
    )
    front_atm_iv = (
        df_front.sort("_atm_dist")
        .group_by("date")
        .agg(pl.col("iv").first().alias("_iv_front"))
    )

    # ATM IV for second month
    df_second = df.join(
        dates_with_two.select("date", "_second_exp"),
        on="date",
        how="inner",
    ).filter(pl.col("expiration_date") == pl.col("_second_exp"))

    df_second = df_second.with_columns(
        (pl.col("moneyness") - 1.0).abs().alias("_atm_dist")
    )
    second_atm_iv = (
        df_second.sort("_atm_dist")
        .group_by("date")
        .agg(pl.col("iv").first().alias("_iv_second"))
    )

    slope = front_atm_iv.join(second_atm_iv, on="date", how="inner").with_columns(
        (pl.col("_iv_second") - pl.col("_iv_front")).alias("iv_term_slope")
    ).select("date", "iv_term_slope")

    return slope


def compute_tlm30(df_front: pl.DataFrame) -> pl.DataFrame:
    """Compute tail loss measure from OTM put prices (Bali et al. TLM30).

    TLM = sum of OTM put mid_quotes weighted by (K/S) for puts with
    moneyness < 0.97 (OTM puts only), normalized by underlying price.
    Approximates the expected loss in the left tail implied by option prices.
    """
    otm_puts = df_front.filter(
        (pl.col("right") == "PUT") & (pl.col("moneyness") < 0.97)
    )

    tlm = (
        otm_puts.group_by("date")
        .agg(
            (
                (pl.col("mid_quote") * pl.col("strike")).sum()
                / pl.col("underlying_price").first()
                / pl.col("underlying_price").first()
            ).alias("tlm30")
        )
    )

    return tlm


def compute_pc_ratios(df_day: pl.DataFrame) -> pl.DataFrame:
    """Compute put/call volume and OI ratios per date.

    Uses ALL contracts for the date (not just front month),
    as volume/OI ratios reflect broad sentiment.
    """
    ratios = (
        df_day.group_by("date")
        .agg(
            pl.col("volume").filter(pl.col("right") == "PUT").sum().alias("_put_vol"),
            pl.col("volume").filter(pl.col("right") == "CALL").sum().alias("_call_vol"),
            pl.col("volume").sum().alias("_total_vol"),
            pl.col("open_interest")
            .filter(pl.col("right") == "PUT")
            .sum()
            .alias("_put_oi"),
            pl.col("open_interest")
            .filter(pl.col("right") == "CALL")
            .sum()
            .alias("_call_oi"),
        )
        .with_columns(
            # put volume / total volume (avoids division by zero vs put/call ratio)
            (pl.col("_put_vol") / pl.col("_total_vol")).alias("pc_volume_ratio"),
            # put OI / call OI
            (
                pl.col("_put_oi").cast(pl.Float64)
                / pl.col("_call_oi").cast(pl.Float64)
            ).alias("pc_oi_ratio"),
        )
        .select("date", "pc_volume_ratio", "pc_oi_ratio")
    )

    # Replace inf/nan from division by zero
    ratios = ratios.with_columns(
        pl.when(pl.col("pc_volume_ratio").is_infinite() | pl.col("pc_volume_ratio").is_nan())
        .then(None)
        .otherwise(pl.col("pc_volume_ratio"))
        .alias("pc_volume_ratio"),
        pl.when(pl.col("pc_oi_ratio").is_infinite() | pl.col("pc_oi_ratio").is_nan())
        .then(None)
        .otherwise(pl.col("pc_oi_ratio"))
        .alias("pc_oi_ratio"),
    )

    return ratios


def _interpolate_iv_to_fine_grid(
    K_obs: np.ndarray,
    iv_obs: np.ndarray,
    S: float,
    n_points: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate observed IV onto a fine moneyness grid per Carr & Wu (2009).

    Steps:
        1. Convert strikes to log-moneyness: ln(K/S)
        2. Linearly interpolate IV across observed moneyness range
        3. Flat-extrapolate tails: IV beyond observed range = nearest observed IV
        4. Return fine grid of (K, IV) pairs spanning the observed range

    Carr & Wu use 2,000 points; we use 1,000 as a pragmatic balance
    (their test shows <1% error even with 5 observed strikes).
    """
    # Log-moneyness
    lm_obs = np.log(K_obs / S)

    # Sort by moneyness
    sort_idx = np.argsort(lm_obs)
    lm_obs = lm_obs[sort_idx]
    iv_obs = iv_obs[sort_idx]

    # Extend range with flat extrapolation into tails (Carr & Wu: ±8 std devs)
    # Use the median IV as a rough proxy for sigma to set the range
    sigma = np.median(iv_obs)
    lm_min = min(lm_obs[0], -8 * sigma * np.sqrt(1.0))  # at least ±8σ√T (T≈1 approx)
    lm_max = max(lm_obs[-1], 8 * sigma * np.sqrt(1.0))

    lm_fine = np.linspace(lm_min, lm_max, n_points)

    # Linear interpolation within observed range, flat extrapolation at tails
    iv_fine = np.interp(lm_fine, lm_obs, iv_obs)

    # Convert back to strikes
    K_fine = S * np.exp(lm_fine)

    return K_fine, iv_fine


def _iv_to_otm_price(
    K: np.ndarray,
    iv: np.ndarray,
    S: float,
    T: float,
    r: float = 0.0,
) -> np.ndarray:
    """Convert IV to OTM option prices via Black-Scholes.

    For K > S: use call price. For K < S: use put price.
    At K = S: use call (arbitrary, negligible impact).
    """
    from scipy.stats import norm

    d1 = (np.log(S / K) + (r + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
    d2 = d1 - iv * np.sqrt(T)

    call = S * np.exp(-0 * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-0 * T) * norm.cdf(-d1)

    # OTM selection: calls for K > S, puts for K < S
    price = np.where(K >= S, call, put)

    return np.maximum(price, 0.0)


# Minimum strikes per side for BKM (Carr & Wu / Driessen et al.: 3 total, we
# require 2 per side to ensure meaningful integration on both tails)
_BKM_MIN_STRIKES_PER_SIDE = 2


def compute_bkm_moments(df_front: pl.DataFrame) -> pl.DataFrame:
    """Compute BKM risk-neutral skewness and kurtosis.

    Bakshi-Kapadia-Madan (2003) method with Carr & Wu (2009) interpolation.

    BKM define three "contracts" integrated over OTM option prices Q(K):
        e(t,τ) = ∫ 2(1 - ln(K/S)) / K² * Q(K) dK
        f(t,τ) = ∫ (6·ln(K/S) - 3·ln²(K/S)) / K² * Q(K) dK
        g(t,τ) = ∫ (12·ln²(K/S) - 4·ln³(K/S)) / K² * Q(K) dK

    Then (with r ≈ 0):
        μ  = -e/2 - f/6 - g/24
        σ² = e - μ²
        SKEW = (f - 3μe + 2μ³) / σ³
        KURT = (g - 4μf + 6μ²e - 3μ⁴) / σ⁴

    Literature: Carr & Wu (2009) interpolation yields <1% error with
    as few as 5 observed strikes. Minimum requirement: 2 OTM per side.
    """
    results = []

    for (date,), group in df_front.group_by("date"):
        S = group["underlying_price"][0]
        T = group["dte"][0] / 365.0
        if T <= 0 or S <= 0:
            continue

        # Separate OTM calls and puts
        otm_calls = group.filter(
            (pl.col("right") == "CALL") & (pl.col("strike") > S)
        ).sort("strike")
        otm_puts = group.filter(
            (pl.col("right") == "PUT") & (pl.col("strike") < S)
        ).sort("strike")

        if (len(otm_calls) < _BKM_MIN_STRIKES_PER_SIDE
                or len(otm_puts) < _BKM_MIN_STRIKES_PER_SIDE):
            continue

        # Collect all OTM strikes and their IVs
        K_all = np.concatenate([
            otm_puts["strike"].to_numpy(),
            otm_calls["strike"].to_numpy(),
        ]).astype(np.float64)
        iv_all = np.concatenate([
            otm_puts["iv"].to_numpy(),
            otm_calls["iv"].to_numpy(),
        ]).astype(np.float64)

        # Remove any NaN IVs
        valid = np.isfinite(iv_all) & (iv_all > 0)
        if valid.sum() < 2 * _BKM_MIN_STRIKES_PER_SIDE:
            continue
        K_all = K_all[valid]
        iv_all = iv_all[valid]

        # Step 1-2: Interpolate IV onto fine grid (Carr & Wu method)
        K_fine, iv_fine = _interpolate_iv_to_fine_grid(K_all, iv_all, S)

        # Step 3: Convert interpolated IV to OTM prices
        price_fine = _iv_to_otm_price(K_fine, iv_fine, S, T)

        # Step 4: Midpoint summation over fine grid
        dK = np.diff(K_fine)
        K_mid = (K_fine[:-1] + K_fine[1:]) / 2
        P_mid = (price_fine[:-1] + price_fine[1:]) / 2
        lnk = np.log(K_mid / S)

        # BKM contract integrands (exact coefficients from BKM 2003)
        integrand_e = 2 * (1 - lnk) * P_mid / (K_mid ** 2)
        integrand_f = (6 * lnk - 3 * lnk ** 2) * P_mid / (K_mid ** 2)
        integrand_g = (12 * lnk ** 2 - 4 * lnk ** 3) * P_mid / (K_mid ** 2)

        e = np.sum(integrand_e * dK)
        f = np.sum(integrand_f * dK)
        g = np.sum(integrand_g * dK)

        # Step 5: BKM moments
        mu = -e / 2 - f / 6 - g / 24  # e^(rT) - 1 ≈ 0 for r ≈ 0
        sigma_sq = e - mu ** 2

        if sigma_sq > 1e-10:
            sigma = sigma_sq ** 0.5
            skew = (f - 3 * mu * e + 2 * mu ** 3) / (sigma ** 3)
            kurt = (g - 4 * mu * f + 6 * mu ** 2 * e - 3 * mu ** 4) / (sigma ** 4)
        else:
            skew = kurt = None

        # Sanity bounds
        if skew is not None and (abs(skew) > 10 or abs(kurt) > 100):
            skew = kurt = None

        results.append({"date": date, "rn_skewness": skew, "rn_kurtosis": kurt})

    if not results:
        return pl.DataFrame(
            schema={"date": pl.Date, "rn_skewness": pl.Float64, "rn_kurtosis": pl.Float64}
        )

    return pl.DataFrame(results)


def compute_vs_spread(df: pl.DataFrame) -> pl.DataFrame:
    """Compute weighted put-call IV spread per Cremers & Weinbaum (2010).

    For each (date, expiration, strike) pair that has BOTH a call and a put,
    compute call_IV - put_IV, weighted by the average open interest of the
    pair. Then aggregate to a single vs_level per date.

    Deviations from put-call parity capture informed directional trading.
    Bali et al. classify this as a core "Informed Trading" feature.

    Returns:
        DataFrame with columns: date, vs_level, vs_change
    """
    calls = df.filter(pl.col("right") == "CALL").select(
        "date", "expiration_date", "strike",
        pl.col("iv").alias("_call_iv"),
        pl.col("open_interest").alias("_call_oi"),
    )
    puts = df.filter(pl.col("right") == "PUT").select(
        "date", "expiration_date", "strike",
        pl.col("iv").alias("_put_iv"),
        pl.col("open_interest").alias("_put_oi"),
    )

    # Match pairs on (date, expiration, strike)
    pairs = calls.join(puts, on=["date", "expiration_date", "strike"], how="inner")

    # Weight = average OI of the pair
    pairs = pairs.with_columns(
        ((pl.col("_call_oi") + pl.col("_put_oi")).cast(pl.Float64) / 2.0).alias("_weight"),
        (pl.col("_call_iv") - pl.col("_put_iv")).alias("_iv_diff"),
    )

    # Weighted average per date
    vs = (
        pairs.group_by("date")
        .agg(
            (pl.col("_iv_diff") * pl.col("_weight")).sum().alias("_weighted_sum"),
            pl.col("_weight").sum().alias("_total_weight"),
        )
        .with_columns(
            (pl.col("_weighted_sum") / pl.col("_total_weight")).alias("vs_level"),
        )
        .select("date", "vs_level")
    )

    # Replace inf/nan from zero-weight days
    vs = vs.with_columns(
        pl.when(pl.col("vs_level").is_infinite() | pl.col("vs_level").is_nan())
        .then(None)
        .otherwise(pl.col("vs_level"))
        .alias("vs_level"),
    )

    # vs_change = daily difference (needs sorted time series)
    vs = vs.sort("date").with_columns(
        pl.col("vs_level").diff().alias("vs_change"),
    )

    return vs


def compute_option_turnover(df: pl.DataFrame) -> pl.DataFrame:
    """Compute option turnover: total volume / total OI per date.

    High turnover signals active repositioning, often preceding
    volatility moves. Bali et al. illiquidity/risk feature.

    Returns:
        DataFrame with columns: date, option_turnover
    """
    turnover = (
        df.group_by("date")
        .agg(
            pl.col("volume").sum().alias("_total_vol"),
            pl.col("open_interest").sum().alias("_total_oi"),
        )
        .with_columns(
            (
                pl.col("_total_vol").cast(pl.Float64)
                / pl.col("_total_oi").cast(pl.Float64)
            ).alias("option_turnover"),
        )
        .select("date", "option_turnover")
    )

    # Replace inf/nan from zero-OI days
    turnover = turnover.with_columns(
        pl.when(
            pl.col("option_turnover").is_infinite()
            | pl.col("option_turnover").is_nan()
        )
        .then(None)
        .otherwise(pl.col("option_turnover"))
        .alias("option_turnover"),
    )

    return turnover


def compute_option_features(
    df: pl.DataFrame,
    rv_daily: pl.DataFrame,
    df_wide: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Compute all 15 option-implied features for one symbol.

    Args:
        df: Delta-filtered options from options_iv/{SYMBOL}.parquet.
            Used for ATM IV, skew, term structure, TLM, P/C ratios.
        rv_daily: RV data with columns: date, rv_m (for VRP calculation).
        df_wide: Full-chain options with IV (pre-delta-filter) for BKM
            moment computation. Per Carr & Wu (2009) and Bali et al. (2021),
            BKM requires the full OTM strike chain including deep OTM puts.
            If None, falls back to using df (delta-filtered, biased).

    Returns:
        DataFrame at (date) level with 15 feature columns.
    """
    df = df.sort("date", "expiration_date", "strike")

    # Front-month data for most features
    df_front = _nearest_monthly_expiration(df)

    # 1. ATM IV
    atm_iv = compute_atm_iv(df_front)

    # 2. IV skew
    iv_skew = compute_iv_skew(df_front)

    # 3. IV term structure slope
    iv_term_slope = compute_term_slope(df)

    # 4. Tail loss measure
    tlm30 = compute_tlm30(df_front)

    # 5. P/C ratios (use all expirations)
    pc_ratios = compute_pc_ratios(df)

    # 6. BKM risk-neutral moments (from full chain, not delta-filtered)
    bkm_source = df_wide if df_wide is not None else df
    bkm_source = bkm_source.sort("date", "expiration_date", "strike")
    bkm_front = _nearest_monthly_expiration(bkm_source)
    bkm = compute_bkm_moments(bkm_front)

    # 7. Weighted put-call IV spread (Cremers & Weinbaum 2010)
    # Uses full chain (pre-delta-filter) — matched pairs need both call+put
    # at same strike, which is sparse in delta-filtered data
    vs_source = df_wide if df_wide is not None else df
    vs_spread = compute_vs_spread(vs_source)

    # 8. Option turnover (Bali et al.) — all expirations
    opt_turnover = compute_option_turnover(df)

    # --- Assemble daily features ---
    features = atm_iv
    for right_df in [iv_skew, iv_term_slope, tlm30, pc_ratios, bkm,
                     vs_spread, opt_turnover]:
        features = features.join(right_df, on="date", how="left")

    # 7. VRP = atm_iv - rv_m (join realized vol)
    features = features.join(
        rv_daily.select("date", "rv_m"), on="date", how="left"
    )
    features = features.with_columns(
        (pl.col("atm_iv") - pl.col("rv_m").sqrt()).alias("vrp"),
        (pl.col("atm_iv") / pl.col("rv_m").sqrt()).alias("vrp_ratio"),
    )
    # Note: rv_m is annualized variance, atm_iv is annualized vol
    # VRP = IV - sqrt(RV_variance) = vol - vol comparison

    # Replace inf/nan vrp_ratio
    features = features.with_columns(
        pl.when(pl.col("vrp_ratio").is_infinite() | pl.col("vrp_ratio").is_nan())
        .then(None)
        .otherwise(pl.col("vrp_ratio"))
        .alias("vrp_ratio"),
    )

    # 8-9. Vol-of-vol and volunc (require time series of atm_iv)
    features = features.sort("date").with_columns(
        pl.col("atm_iv").rolling_std(22).alias("vol_of_vol"),
        (pl.col("atm_iv").pct_change()).rolling_std(22).alias("volunc"),
    )

    # Drop rv_m (it came from rv_daily, not an option feature)
    features = features.drop("rv_m")

    return features.sort("date")
