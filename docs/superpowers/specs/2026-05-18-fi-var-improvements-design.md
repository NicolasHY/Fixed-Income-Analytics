# FI VaR Improvements — Design

**Date:** 2026-05-18
**Source:** Port selected sections from `portfolio_risk_equity.ipynb` (past equity-risk project) into the VaR module of `main.ipynb` (current EM FI Analytics Suite).

## Goal

Strengthen Module 2 of `main.ipynb` (the multi-method VaR engine) with three additions from the past equity-risk project that translate cleanly to the duration-proxy FI setup:

1. Stressed VaR + distribution overlay
2. Comparative parametric-t VaR table at multiple degrees of freedom
3. Factor (PC) vs idiosyncratic variance decomposition

Each addition gets short narrative commentary matching the tone of the equity notebook (what the numbers mean, which model is most reliable, caveats).

## Out of scope

- PC-based factor stress scenarios (considered but not selected).
- OLS assumption diagnostics, rolling betas, Fama-French factor model — these belong to the equity setup and do not apply to the duration-proxy FI portfolio.
- Refactoring existing VaR code into `src/`.

## Where the changes go

- `main.ipynb` — Module 2 (VaR Engine). Three new code cells plus inline markdown commentary cells. No changes to Modules 1, 3, or 4.
- `config/funds.yaml` — new `var.stress_windows` block.
- `tests/test_var.py` — three new tests.

## 1. Stressed VaR + distribution overlay

### Config

Add to `config/funds.yaml` under `var:`:

```yaml
stress_windows:
  COVID: ["2020-02-19", "2020-05-15"]
  Ukraine_Fed: ["2022-02-24", "2022-10-31"]
primary_stress_window: COVID  # used for the overlay plot
```

The dates above are the defaults; the YAML is the source of truth so the analyst can swap windows without code changes.

### Computation

New cell after the historical-VaR cell. For each window:

- Slice `portfolio_pnl` between the two dates.
- Compute VaR/CVaR at 95% and 99% via the historical method (same quantile + tail-mean approach already used in the file).
- Store in a `stressed_var_results` dict keyed by window name.

Display: a single DataFrame placing the stress windows next to `Historical 1Y` and `Historical 3Y` from the existing table, so the analyst sees the stress VaR alongside the full-sample VaR at a glance.

### Plot

One overlay histogram (matching the equity notebook's `Portfolio Return Distribution: Full Sample vs COVID Stress`):

- Blue histogram: full-sample `portfolio_pnl` (density).
- Red histogram: primary stress window slice (density).
- Three vertical lines (at the negative of the VaR, i.e. on the loss tail): full-sample historical VaR 95%, parametric-normal VaR 95%, stressed VaR 95%.
- One short annotation if the stressed VaR sits materially to the left of the full-sample VaR.

Save to `data/output/var_stress_overlay.png`.

### Commentary

Short paragraph after the plot: stressed VaR is the historical VaR you would have measured *during* the crisis; if it is materially larger than full-sample VaR, the parametric/historical models trained on the full sample are under-pricing tail risk relative to crisis-conditional levels.

## 2. Comparative parametric-t VaR table at multiple ν

### Computation

New cell after the existing parametric-VaR cell. Keep everything that is already there (normal and MLE-fitted-t). Add a comparison table for ν ∈ {4, 5, 8, 20, ∞} using the variance-corrected scaling from the equity project:

```python
for nu in [4, 5, 8, 20]:  # all > 2 so the variance correction is well-defined
    scale = sigma * np.sqrt((nu - 2) / nu)
    VaR_95 = -(mu + t_dist.ppf(0.05, df=nu) * scale)
    VaR_99 = -(mu + t_dist.ppf(0.01, df=nu) * scale)
    CVaR_95 = -portfolio_pnl[portfolio_pnl <= -VaR_95].mean()
    CVaR_99 = -portfolio_pnl[portfolio_pnl <= -VaR_99].mean()
```

The `np.sqrt((nu - 2) / nu)` correction ensures the scaled-t variance matches the sample variance, so the table compares like-for-like across ν. (Defined only for ν > 2, which is why the ν grid starts at 4.) The ν = ∞ row uses the existing normal values.

### Display

DataFrame indexed by ν with columns `VaR 95%`, `VaR 99%`, `CVaR 95%`, `CVaR 99%`. The existing MLE-fitted-t row is kept as the "primary" parametric-t for the summary and backtest tables.

### Commentary

Short paragraph: lower ν → fatter tails → larger 99% VaR; the table makes the model's sensitivity to ν explicit and shows how the MLE-fitted ν sits in the spectrum.

## 3. Factor (PC) vs idiosyncratic variance decomposition

### Setup

Uses the panel PCA already computed in Module 1 (`panel_scores_df`, indexed by date, columns PC1/PC2/PC3). The decomposition operates on the **5Y yield changes** of the four LC-fund countries (the same series used to build `portfolio_pnl`).

### Computation

For each country `c` in `lc_weights`:

```python
# panel_scores_df columns are: "PC1 (global level)", "PC2 (global slope)", "PC3 (global curvature)"
pc_cols = list(panel_scores_df.columns)

# Align: PC scores (panel) and the country's 5Y yield change on a common index
y_c = change_dfs[c]["5Y"]
common_idx = panel_scores_df.index.intersection(y_c.index)
X = sm.add_constant(panel_scores_df.loc[common_idx])
y_aligned = y_c.loc[common_idx]
model = sm.OLS(y_aligned, X).fit()
beta[c] = model.params[pc_cols].values
resid_var[c] = model.resid.var()
```

Build `B` (4 × 3, country × PC), `D = diag(resid_var)` (4 × 4), `Sigma_F = panel_scores_df.cov()` (3 × 3, near-diagonal by PCA orthogonality).

Decompose total portfolio yield-change variance:

- `var_systematic = w @ B @ Sigma_F @ B.T @ w`
- `var_idiosyncratic = w @ D @ w`
- `var_total = var_systematic + var_idiosyncratic`

### Display

- Print `% systematic` and `% idiosyncratic` of `var_total`.
- Horizontal bar chart (mirroring the equity notebook), saved to `data/output/var_risk_decomposition.png`.

### Caveat (called out in markdown)

The decomposition ignores the cross-country residual correlation: `Var(w' Δy)` strictly equals `w' (B Σ_F B' + Σ_ε) w`, and `Σ_ε` is not in general diagonal. Using `D` instead of `Σ_ε` matches the equity notebook's methodology and is acceptable for the rough split, but the two figures will not sum exactly to the empirical portfolio variance. Show both the decomposition total and the empirical `Var(portfolio_dy)` for transparency.

### Commentary

Short paragraph: how much of the LC fund's daily yield-change variance is explained by common EM rate moves (PC1/2/3) versus country-specific noise. High systematic share → the fund's risk is dominated by global EM rate factors and diversification across the four countries is limited.

## Tests

Add to `tests/test_var.py`. The existing test file uses synthetic fixtures from `tests/conftest.py` — extend that pattern.

### Test 1: stressed VaR is at least as large as full-sample VaR

```python
def test_stressed_var_is_at_least_as_large_as_full_sample_var(portfolio_pnl):
    # Construct a stress window from the worst 20% of days; its historical VaR
    # at 95% must be >= the full-sample 95% VaR by construction of the stress slice.
```

This tests the *helper logic*, not the COVID slice (which depends on real data). Use a synthetic stress slice built from the worst tail of the synthetic `portfolio_pnl` fixture.

### Test 2: multi-ν table is monotonic in ν at the 99% level

```python
def test_multi_nu_99var_monotonic_decreasing_in_nu(portfolio_pnl):
    # For fixed mu, sigma, increasing nu should produce smaller 99% VaR
    # (thinner tails). Tolerance applied for the sample-noise regime.
```

### Test 3: factor + idiosyncratic decomposition sums close to total

```python
def test_decomposition_sums_close_to_empirical_variance(synthetic_yield_changes):
    # Inside the test, build a synthetic 3-factor structure: simulate PC scores,
    # set per-country betas, add small idiosyncratic noise, and form yield-change
    # series whose residual cross-correlation is small by construction. Then run
    # the decomposition and assert var_systematic + var_idiosyncratic is within
    # 15% of empirical Var(w' Δy).
```

The decomposition test builds its own PC-score panel inside the test body — it does not depend on a new shared fixture. The tolerance reflects the residual-correlation simplification. The point is to detect bugs (e.g., wrong matrix orientation), not to enforce exact equality.

### Existing-test impact

No existing tests change. The Kupiec/Christoffersen helpers stay untouched.

## Reproducibility

- Monte Carlo cells already seed `np.random` via `var.monte_carlo.random_seed` in `funds.yaml`. The new additions are deterministic (no random sampling).
- All thresholds, windows, ν grid, and `lc_weights` come from the config / existing code — no new hardcoded constants in the cells.

## Output artifacts

New files written to `data/output/`:

- `var_stress_overlay.png`
- `var_risk_decomposition.png`

The existing `var_pnl_bands.png` stays as-is.
