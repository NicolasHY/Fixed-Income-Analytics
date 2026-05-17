# Risk Statistics — Data Gaps & Approximations

This document lists every metric in the **Risk Statistics** tab, notes how it is currently computed, and flags what additional data or decisions are needed to make it exact.

> **Last updated**: 2026-05-17
> **Status legend**: ✅ Resolved · ⚠️ Approximation (usable) · ❌ Blocked (missing data)

---

## Recently Resolved

### ✅ Sharpe Ratio — now uses €STR (EUR OIS)
**Source**: FRED API (`ECBESTRVOLWGTTRMDMNRT` from 2019-10-01, no EONIA backfill needed since portfolio data starts 2018 and €STR covers the overlapping period). SOFR (`SOFR` + `DFF` backfill) also fetched and displayed. Rates cached in `data/output/risk_free_rates.csv`.

**Current levels** (as of 2026-05-14): €STR = 1.930%, SOFR = 3.560%.
**Average €STR over portfolio history**: ~1.345%.

**Implementation**: Daily excess return = `r_t − [(1 + €STR_t/100)^(1/252) − 1]`, aligned to portfolio dates. Sharpe = annualised excess return / annualised excess-return volatility.

**Residual note**: The portfolio P&L is a local-currency *price* return approximation. A EUR investor's true excess return should also include FX return on LC positions — this is not yet captured (see FX gap below).

### ✅ Sortino Ratio — now uses €STR as MAR
Same excess-return series as above. Downside semi-deviation = `√(E[min(excess, 0)²]) × √252`.

---

---

## Return Metrics

### Cumulative Log Return
- **Implementation**: `Σ ln(1 + r_t)` over the full history.
- **Status**: Exact given the P&L proxy.
- **Gap**: The P&L proxy itself uses a duration approximation (`ΔP/P ≈ −D_eff × Δy/100`). Actual bond pricing would require coupon schedules and exact maturity dates.

### Annualised Return
- **Implementation**: `(∏(1 + r_t))^(252/T) − 1`, assuming 252 trading days per year.
- **Status**: Exact given the P&L proxy.
- **Gap**: Same as above. Calendar-year annualisation would need exact start/end dates.

### Carry — Weighted Average Yield
- **Implementation**: Portfolio-weighted average of the **latest** benchmark-maturity (5Y) yield for each country.
- **Approximations**:
  - Uses the published **redemption yield** (YTM) as a proxy for carry, which is exact only for a zero-coupon bond or a bond held to maturity.
  - Does **not** deduct funding cost (repo rate), so this is gross carry, not net carry.
  - For **local-currency** bonds (Brazil, Mexico, South Africa, Poland): carry should ideally be currency-hedged (spot minus forward points). FX forward data is not available.
  - For **hard-currency** bonds (Colombia, Hungary, Romania): carry is USD-denominated, so no FX hedging adjustment is needed — this is closer to true carry.
- **Missing data**:
  - FX spot and forward rates (or cross-currency swap rates) for BRL, MXN, ZAR, PLN versus USD/EUR.
  - Repo / financing rates to compute **net carry** = gross yield − funding cost.
  - Coupon accrual schedule (for **accrued-interest carry** vs yield-to-maturity carry).

### Roll-Down Return
- **Implementation**: `D_eff × (y_benchmark − y_next_shorter) / (mat_benchmark − mat_shorter)`, annualised, portfolio-weighted. Uses the latest yield snapshot.
  - Brazil 5Y → 3Y; Colombia 5Y → 4Y; Hungary 5Y → 1Y; Mexico 5Y → 3Y; Poland 5Y → 3Y; Romania 5Y → 3Y.
  - South Africa is **excluded** (no maturity shorter than 5Y in the dataset).
- **Approximations**:
  - Assumes the yield curve **does not shift** between now and the roll-down horizon (static-curve roll-down, not a forward roll-down).
  - Uses a single duration `D_eff = 5.22` for the price impact, which ignores the convexity correction.
  - The roll-down is computed using the closest available shorter maturity, not the true (t−1)-year point. E.g., Hungary uses the 1Y yield because no 2Y, 3Y, 4Y data exists, creating a coarser approximation.
- **Missing data**:
  - Exact residual maturity dates of the bonds held (which maturity are they rolling toward?).
  - Full yield curve at intermediate maturities (e.g., Hungary 2Y, 3Y, 4Y) to interpolate the roll-down point precisely.
  - FX dynamics for local-currency roll-down (if the currency depreciates, roll-down is partially offset).

---

## Risk & Ratio Metrics

### Annualised Volatility
- **Implementation**: `std(r_t) × √252`.
- **Status**: Exact for the historical P&L series used.
- **Gap**: Uses the full sample period; does not account for volatility regime changes. A rolling or GARCH-based estimate would be more forward-looking.

### Maximum Drawdown
- **Implementation**: `min((cum_ret / running_max_cum_ret) − 1)`.
- **Status**: Exact.
- **Gap**: Max drawdown duration and recovery period are not displayed.

### Sharpe Ratio
- **Implementation**: `annualised_return / annualised_vol`, with **risk-free rate = 0**.
- **Missing data**:
  - A risk-free rate series (e.g., US 3-month T-bill, EUR 3-month OIS, or a currency-specific risk-free rate per country).
  - Decision needed: which risk-free rate to use for a multi-currency EM portfolio (USD SOFR? EUR OIS? Each country's local rate?).

### Sortino Ratio
- **Implementation**: `annualised_return / annualised_downside_semi_deviation`, with MAR = 0. Downside semi-deviation = `√(E[min(r, 0)²]) × √252`.
- **Gap**: MAR = 0 is a simplification. A proper MAR could be the fund's performance target or risk-free rate (see Sharpe gap above).

### Calmar Ratio
- **Implementation**: `annualised_return / |max_drawdown|`.
- **Status**: Exact.
- **Gap**: Max drawdown is based on the full history; some practitioners use 36-month rolling Calmar.

---

## Bond Analytics

### Modified Duration
- **Implementation**: Uses `D_eff = 5.22` from `config/funds.yaml` — a **constant** for both portfolios.
- **Approximations**:
  - This is the **effective duration** of the benchmark LC fund (Company PLBEMSA), not re-derived from the actual bond universe.
  - Modified duration changes over time as yields move and bonds age; this implementation treats it as fixed.
  - Both portfolios use the same `D_eff` because the config does not specify per-portfolio durations. Portfolio 2 (South Africa-heavy) likely has a **different** modified duration since South Africa's curve has a different shape.
- **Missing data**:
  - Per-country (or better, per-bond ISIN) modified duration.
  - For a proper per-portfolio MD: `MD_portfolio = Σ w_i × MD_i` where `MD_i` is each country's benchmark bond duration at the current yield.
  - Coupon schedule and exact maturity date for each benchmark bond, so that `D_mod = D_mac / (1 + y)` can be computed precisely (rather than approximated by `D_eff`).

### DV01
- **Implementation**: `D_eff × 0.01%`, i.e., the percentage change in portfolio NAV for a 1bp parallel shift in all yields simultaneously.
- **Approximations**:
  - Computed in **relative terms** (% of NAV). To express in **dollar terms**, multiply by the portfolio notional value (not available).
  - Assumes a parallel yield shift; does not decompose into key-rate DV01s.
- **Missing data**:
  - Portfolio notional (AUM in USD/EUR) for dollar DV01.
  - Per-country DV01 (see KRD section).

### Convexity
- **Implementation**: Approximation `D_mac × (D_mac + 1) / (1 + y)²`, where `D_mac = D_eff × (1 + y)` and `y` is the weighted-average YTM as a decimal.
- **Approximations**:
  - This formula is exact only for a **zero-coupon bond** (Macaulay duration = maturity). For coupon bonds, convexity is lower because cash flows occur before maturity.
  - A tighter approximation for a par coupon bond: `C = [2/y² × (1 − 1/(1+y)^T) − 2T/y × 1/(1+y)^T + T(T+1)/(1+y)^T] × 1/(1+y)^2`; this requires knowing `T` (exact maturity) and the coupon rate.
- **Missing data**:
  - Coupon rates for each sovereign benchmark bond.
  - Exact remaining maturity (in years, not rounded to the benchmark bucket).

### YTM — Weighted Average Benchmark
- **Implementation**: Latest portfolio-weighted benchmark yield, identical to Carry above.
- **Gap**: This is the yield of the **benchmark maturity** proxy, not the YTM of the specific bonds held. Actual bond YTMs depend on the exact coupon, settlement date, and price.

### Yield Curve Slope
- **Implementation**: Latest `(y_longest_available − y_shortest_available)` per country, portfolio-weighted. Maturity range differs by country (e.g., Romania 3–10Y vs South Africa 5–30Y).
- **Gap**: No standardised maturity pair (e.g., 10Y–2Y) is used because not all countries have both. Comparisons across countries are not fully like-for-like. A common slope metric (e.g., 10Y–2Y) would require interpolation for Hungary (1Y, 5Y, 10Y) and Colombia (4Y, 5Y, 10Y).

---

## VaR / CVaR

### Parametric VaR / CVaR (Normal)
- **Implementation**: `VaR = −(μ + z_α × σ)`, `CVaR = −(μ − σ × φ(−z_α) / α)` where μ and σ are the full-sample daily P&L mean and std.
- **Approximations**:
  - Normal distribution assumption is violated by EM bonds, which exhibit **fat tails**, skewness, and volatility clustering.
  - Using full-sample μ and σ (stationary assumption) ignores regime changes in vol.
- **Missing data / decisions**:
  - A risk-free floor or target for the μ assumption.
  - Whether to use a **rolling window** (e.g., 252-day) vs full-sample for μ, σ.
  - Stress VaR (using a stressed period like 2020 COVID or 2022 Russia) is not computed.

### Historical VaR / CVaR
- **Implementation**: Empirical quantile of the P&L series; CVaR = mean of tail observations.
- **Status**: Exact for the historical P&L proxy.
- **Gap**: Limited history (1,356 obs ≈ 5.4 years). Regulatory stressed VaR typically requires a 250-day stressed window; Basel III requires 10-day horizon VaR.

### Monte Carlo VaR
- **Implementation**: 50,000 draws from `N(μ, σ²)`, seed = 42. `VaR = −percentile(sims, α × 100)`.
- **Approximations**:
  - Univariate normal — does not capture cross-country correlation structure.
  - A multivariate simulation using the **7×7 yield-change covariance matrix** would be more accurate: simulate country-level yield changes jointly, reweight by portfolio weights, then apply the duration approximation.
  - Does not model **fat tails** (Student-t copula, etc.).
- **Missing data / decisions**:
  - Covariance matrix of per-country yield changes (computable from existing data — planned enhancement).
  - Choice of distribution (Normal vs Student-t; if Student-t, degrees of freedom).

---

## Key-Rate Duration (KRD)

- **Implementation**: Per-country KRD = `w_i × D_eff`. Sum across countries = portfolio modified duration.
- **Approximations**:
  - All duration is allocated to the **5Y benchmark point** for every country. There is no within-country maturity decomposition.
  - A proper KRD grid (1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y) requires knowing the **cash-flow schedule** of each bond held and its sensitivity to each key rate.
- **Missing data**:
  - Full bond universe per country with ISIN, coupon, maturity, and weight in the portfolio.
  - To implement multi-maturity KRD without a full bond schedule: a regression of portfolio P&L against yield changes at each individual maturity can serve as an approximation. This is feasible with the existing multi-maturity CSVs.

---

## Correlation Matrix

- **Implementation**: Pearson correlation of daily first-differences of 5Y benchmark yields across all 7 countries, over all available common dates.
- **Status**: Exact for the yield-change series used.
- **Gap**:
  - This is the **yield-change** correlation, not the **P&L contribution** correlation. The latter would weight each country's yield change by `w_i × D_eff` — mathematically it produces the same correlation matrix (scalar weights cancel in Pearson correlation), so the matrix is portfolio-agnostic.
  - No display of **rolling** correlation (e.g., 60-day or 252-day) to capture time-varying co-movement.

---

## Summary of Minimum Additional Data Needed

| Status | Priority | Data Item | Metrics It Unlocks |
|--------|----------|-----------|-------------------|
| ✅ | ~~High~~ | ~~Risk-free rate series (SOFR / EUR OIS)~~ | ~~Proper Sharpe & Sortino~~ — **done via FRED** |
| ❌ | **High** | Per-country modified duration (not fixed D_eff) | Accurate MD, DV01, KRD, Convexity per portfolio |
| ❌ | **High** | FX spot & forward rates (BRL, MXN, ZAR, PLN) | Currency-hedged carry; FX-adjusted excess return for LC bond Sharpe |
| ❌ | **Medium** | Coupon rates for each sovereign benchmark bond | Exact convexity, accrued carry |
| ❌ | **Medium** | Exact residual maturity dates | Precise roll-down, modified duration drift |
| ❌ | **Medium** | Portfolio notional (AUM in base currency) | Dollar DV01, dollar VaR |
| ❌ | **Low** | Full bond universe per country (ISIN-level) | Multi-point KRD grid, exact P&L attribution |
| ❌ | **Low** | Intermediate maturities (e.g., Hungary 2–4Y) | Accurate roll-down for all countries |
| ❌ | **Low** | Funding / repo rates per country | Net carry (gross carry is already displayed) |
| ❌ | **Low** | Credit rating history | Rating-adjusted KRD, spread DV01 (HC bonds) |

---

## Metrics Not Yet Implemented (Require Above Data)

- **Tracking Error vs Benchmark** — needs benchmark index weights & returns.
- **Information Ratio** — needs benchmark returns.
- **Beta to EM benchmark** — needs benchmark total-return index.
- **Spread Duration** (HC bonds) — needs OAS decomposition (risk-free + spread).
- **Dollar DV01 / Dollar Convexity** — needs portfolio AUM.
- **Liquidity-adjusted VaR** — needs bid-ask spread data per instrument.
- **Stressed VaR** — needs a designated stressed period (e.g., 2020-03 or 2022-02).
- **10-day horizon VaR** (Basel III) — can be approximated as `VaR_1d × √10` once daily VaR is exact.
- **Currency contribution to returns** — needs FX spot returns per currency.
- **Interest rate sensitivity per maturity bucket** (true multi-point KRD) — needs bond cash-flow schedules or regression approach (feasible with existing multi-maturity CSV data).
