# FI VaR Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port three additions from `portfolio_risk_equity.ipynb` into Module 2 of `main.ipynb`: stressed VaR + distribution overlay, comparative parametric-t VaR table at multiple ν, and factor (PC) vs idiosyncratic variance decomposition.

**Architecture:** All work lands in `main.ipynb` Module 2, with one config block added to `config/funds.yaml` and three new tests in `tests/test_var.py`. Helper functions are added to `tests/test_var.py` (self-contained copies of the notebook logic, matching the project's existing test pattern documented in `CLAUDE.md`). Notebook cells are inserted programmatically using `nbformat` anchored on existing markdown headers so the edits are reproducible and reviewable.

**Tech Stack:** Python 3, pandas, numpy, scipy.stats (norm, t, chi2), statsmodels (OLS for PC regressions), matplotlib, sklearn (already used for PCA in Module 1), nbformat, jupyter nbconvert, pytest, PyYAML.

---

## File Structure

- **Modify:** `config/funds.yaml` — add `var.stress_windows` and `var.primary_stress_window`.
- **Modify:** `tests/test_var.py` — add three helper functions and three tests.
- **Modify:** `main.ipynb` — insert three groups of cells (markdown + code + plot) into Module 2 via nbformat scripts.
- **Spec reference:** `docs/superpowers/specs/2026-05-18-fi-var-improvements-design.md`.

Each task below modifies exactly one of these areas and ends with a commit.

---

## Task 1: Add stress-window config to `funds.yaml`

**Files:**
- Modify: `config/funds.yaml` (insert new keys under the existing `var:` block, currently lines 46-55)

- [ ] **Step 1: Open `config/funds.yaml` and locate the `var:` block**

The block currently ends with `copula_dof: 5  # Student-t copula degrees of freedom`. Insert the two new keys directly after it, before the `# -----` separator that starts the PCA section.

- [ ] **Step 2: Add the new keys**

Insert the following inside the `var:` block, after `monte_carlo:` and before the `# ----` separator at line ~57:

```yaml
  stress_windows:
    COVID: ["2020-02-19", "2020-05-15"]
    Ukraine_Fed: ["2022-02-24", "2022-10-31"]
  primary_stress_window: COVID
```

The final `var:` block should look like:

```yaml
var:
  confidence_levels: [0.95, 0.99]
  historical_windows:
    1Y: 252
    2Y: 504
    3Y: 756
  monte_carlo:
    n_simulations: 10000
    random_seed: 42
    copula_dof: 5
  stress_windows:
    COVID: ["2020-02-19", "2020-05-15"]
    Ukraine_Fed: ["2022-02-24", "2022-10-31"]
  primary_stress_window: COVID
```

- [ ] **Step 3: Verify the YAML parses**

Run:
```powershell
python -c "import yaml; cfg = yaml.safe_load(open('config/funds.yaml')); print(cfg['var']['stress_windows']); print(cfg['var']['primary_stress_window'])"
```
Expected output:
```
{'COVID': ['2020-02-19', '2020-05-15'], 'Ukraine_Fed': ['2022-02-24', '2022-10-31']}
COVID
```

- [ ] **Step 4: Commit**

```powershell
git add config/funds.yaml
git commit -m "config(var): add stress windows + primary_stress_window"
```

---

## Task 2: TDD — stressed VaR helper

**Files:**
- Modify: `tests/test_var.py` (add helper after `compute_historical_var`, add test at the end of the file)

- [ ] **Step 1: Write the failing test**

Add to the bottom of `tests/test_var.py`:

```python
def test_stressed_var_exceeds_full_sample_var(portfolio_pnl):
    """
    A stress window constructed from the worst N days must produce a 95%
    historical VaR no smaller than the full-sample 95% historical VaR.
    Tests the compute_stressed_var helper, not any specific real-world date.
    """
    full = compute_historical_var(portfolio_pnl, window=len(portfolio_pnl), confidence=0.95)

    worst_dates = portfolio_pnl.nsmallest(int(len(portfolio_pnl) * 0.2)).index
    start, end = worst_dates.min(), worst_dates.max()
    stressed = compute_stressed_var(portfolio_pnl, start, end, confidence=0.95)

    assert stressed["VaR"] >= full["VaR"] - 1e-9, (
        f"Stressed VaR ({stressed['VaR']:.6f}) should be >= full-sample VaR "
        f"({full['VaR']:.6f}) when the window is drawn from the worst tail"
    )
    assert stressed["n_obs"] > 0
    assert stressed["CVaR"] >= stressed["VaR"], "CVaR must be >= VaR"
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
pytest tests/test_var.py::test_stressed_var_exceeds_full_sample_var -v
```
Expected: FAIL with `NameError: name 'compute_stressed_var' is not defined`.

- [ ] **Step 3: Add the helper function**

Insert into `tests/test_var.py` directly after the `compute_historical_var` function (around line 35):

```python
def compute_stressed_var(pnl_series, start, end, confidence=0.95):
    """
    Historical VaR/CVaR over a stress window.

    pnl_series : pd.Series indexed by date.
    start, end : window bounds (anything accepted by pandas .loc, inclusive).
    confidence : confidence level (e.g. 0.95).
    """
    sample = pnl_series.loc[start:end]
    if len(sample) == 0:
        raise ValueError(f"Empty stress window: {start} to {end}")
    alpha = 1 - confidence
    q = np.quantile(sample, alpha)
    var = -q
    cvar = -sample[sample <= q].mean()
    return {"VaR": var, "CVaR": cvar, "n_obs": len(sample),
            "start": str(sample.index.min().date()),
            "end": str(sample.index.max().date())}
```

- [ ] **Step 4: Run test to verify it passes**

```powershell
pytest tests/test_var.py::test_stressed_var_exceeds_full_sample_var -v
```
Expected: PASS.

- [ ] **Step 5: Run the full test file to confirm no regressions**

```powershell
pytest tests/test_var.py -v
```
Expected: all tests pass, including the new one.

- [ ] **Step 6: Commit**

```powershell
git add tests/test_var.py
git commit -m "test(var): add stressed-VaR helper and tail-window test"
```

---

## Task 3: TDD — multi-ν parametric-t helper

**Files:**
- Modify: `tests/test_var.py` (add helper after `compute_parametric_var`, add test at the end)

- [ ] **Step 1: Write the failing test**

Add to the bottom of `tests/test_var.py`:

```python
def test_multi_nu_99var_monotonic_in_nu(portfolio_pnl):
    """
    Higher nu = thinner tails. Holding mu and sigma constant via the variance
    correction, the 99% parametric-t VaR must shrink as nu increases, and the
    nu -> infinity row must match the normal parametric VaR.
    """
    nus = [4, 5, 8, 20]
    table = compute_multi_nu_var_table(portfolio_pnl, nus=nus)

    var_99 = [table.loc[nu, "VaR 99%"] for nu in nus]
    assert all(var_99[i] >= var_99[i + 1] for i in range(len(var_99) - 1)), (
        f"Expected VaR 99% non-increasing in nu, got {var_99}"
    )

    normal = compute_parametric_var(portfolio_pnl, confidence=0.99)
    assert abs(table.loc["inf", "VaR 99%"] - normal["VaR"]) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
pytest tests/test_var.py::test_multi_nu_99var_monotonic_in_nu -v
```
Expected: FAIL with `NameError: name 'compute_multi_nu_var_table' is not defined`.

- [ ] **Step 3: Add the helper function**

Insert into `tests/test_var.py` directly after the `compute_parametric_var` function (around line 26):

```python
def compute_multi_nu_var_table(pnl_series, nus=(4, 5, 8, 20)):
    """
    Parametric VaR/CVaR at 95% and 99% for Student-t with the variance
    correction scale = sigma * sqrt((nu - 2) / nu), plus a 'inf' (normal) row.

    Returns a DataFrame indexed by nu (with 'inf' as the last row),
    columns = ['VaR 95%', 'VaR 99%', 'CVaR 95%', 'CVaR 99%'].
    """
    mu = pnl_series.mean()
    sigma = pnl_series.std()
    rows = []
    for nu in nus:
        if nu <= 2:
            raise ValueError(f"nu must be > 2 for variance correction, got {nu}")
        s = sigma * np.sqrt((nu - 2) / nu)
        var_95 = -(mu + t_dist.ppf(0.05, df=nu) * s)
        var_99 = -(mu + t_dist.ppf(0.01, df=nu) * s)
        cvar_95 = -pnl_series[pnl_series <= -var_95].mean()
        cvar_99 = -pnl_series[pnl_series <= -var_99].mean()
        rows.append([nu, var_95, var_99, cvar_95, cvar_99])

    # Normal row (nu -> infinity)
    var_95_n = -(mu + norm.ppf(0.05) * sigma)
    var_99_n = -(mu + norm.ppf(0.01) * sigma)
    cvar_95_n = -(mu - sigma * norm.pdf(norm.ppf(0.05)) / 0.05)
    cvar_99_n = -(mu - sigma * norm.pdf(norm.ppf(0.01)) / 0.01)
    rows.append(["inf", var_95_n, var_99_n, cvar_95_n, cvar_99_n])

    df = pd.DataFrame(rows, columns=["nu", "VaR 95%", "VaR 99%", "CVaR 95%", "CVaR 99%"])
    df = df.set_index("nu")
    return df
```

- [ ] **Step 4: Run test to verify it passes**

```powershell
pytest tests/test_var.py::test_multi_nu_99var_monotonic_in_nu -v
```
Expected: PASS.

- [ ] **Step 5: Run the full test file**

```powershell
pytest tests/test_var.py -v
```
Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add tests/test_var.py
git commit -m "test(var): add multi-nu parametric-t helper and monotonicity test"
```

---

## Task 4: TDD — factor (PC) vs idiosyncratic decomposition helper

**Files:**
- Modify: `tests/test_var.py` (add helper after `compute_monte_carlo_var`, add test at the end)

- [ ] **Step 1: Write the failing test**

Add to the bottom of `tests/test_var.py`:

```python
def test_decomposition_sums_close_to_empirical_variance():
    """
    Build a synthetic 3-factor structure with small idiosyncratic noise and
    check that var_systematic + var_idiosyncratic is within 15% of the
    empirical variance of the weighted portfolio yield change.
    """
    import statsmodels.api as sm

    np.random.seed(123)
    n = 1500
    dates = pd.bdate_range("2020-01-01", periods=n)

    # 3 latent factors with distinct variances
    f1 = np.random.normal(0, 0.05, n)
    f2 = np.random.normal(0, 0.03, n)
    f3 = np.random.normal(0, 0.02, n)
    factors = pd.DataFrame({"PC1": f1, "PC2": f2, "PC3": f3}, index=dates)

    # Country betas (4 x 3) and small idiosyncratic noise
    betas_true = np.array([
        [1.0, 0.5, 0.2],
        [0.8, 0.4, 0.1],
        [1.2, -0.3, 0.1],
        [0.7, 0.6, -0.1],
    ])
    eps = np.random.normal(0, 0.005, (n, 4))
    Y = factors.values @ betas_true.T + eps
    yield_changes = pd.DataFrame(Y, index=dates, columns=["c1", "c2", "c3", "c4"])

    weights = np.array([0.3, 0.25, 0.25, 0.2])

    decomp = compute_factor_idio_decomposition(
        yield_changes=yield_changes,
        factor_scores=factors,
        weights=weights,
    )

    empirical_var = float(np.var(yield_changes.values @ weights, ddof=1))
    total = decomp["var_systematic"] + decomp["var_idiosyncratic"]

    assert abs(total - empirical_var) / empirical_var < 0.15, (
        f"Decomposition sum {total:.6f} vs empirical {empirical_var:.6f} "
        f"differs by more than 15%"
    )
    assert decomp["pct_systematic"] + decomp["pct_idiosyncratic"] == pytest.approx(100.0, abs=1e-6)
    assert decomp["B"].shape == (4, 3)
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
pytest tests/test_var.py::test_decomposition_sums_close_to_empirical_variance -v
```
Expected: FAIL with `NameError: name 'compute_factor_idio_decomposition' is not defined`.

- [ ] **Step 3: Add the helper function**

Insert into `tests/test_var.py` directly after `compute_monte_carlo_var` (around line 48). It needs `statsmodels`, so add the import to the top of the file (just after the scipy import). The function:

```python
def compute_factor_idio_decomposition(yield_changes, factor_scores, weights):
    """
    Decompose Var(w' delta_y) into systematic (driven by factor_scores) and
    idiosyncratic (per-series residual) components via OLS per series.

    yield_changes : DataFrame, columns = series (e.g. countries), index = dates.
    factor_scores : DataFrame, columns = factors (e.g. PC1/PC2/PC3), index = dates.
    weights       : 1-D array, same length as yield_changes.columns.

    Returns dict with keys:
      B (n_series x n_factors), D (n_series x n_series diag of resid variances),
      Sigma_F (n_factors x n_factors), var_systematic, var_idiosyncratic,
      pct_systematic, pct_idiosyncratic, var_total.

    Cross-series residual correlation is ignored (matches the equity project's
    methodology and the spec). The caller should display empirical Var(w' dy)
    alongside the decomposition total for transparency.
    """
    import statsmodels.api as sm

    common = yield_changes.index.intersection(factor_scores.index)
    Y = yield_changes.loc[common]
    F = factor_scores.loc[common]
    w = np.asarray(weights, dtype=float)
    if w.shape[0] != Y.shape[1]:
        raise ValueError(f"weights length {w.shape[0]} != n_series {Y.shape[1]}")

    n_series = Y.shape[1]
    n_factors = F.shape[1]
    B = np.zeros((n_series, n_factors))
    resid_var = np.zeros(n_series)

    X = sm.add_constant(F.values)
    for i, col in enumerate(Y.columns):
        model = sm.OLS(Y[col].values, X).fit()
        B[i, :] = model.params[1:]
        resid_var[i] = model.resid.var(ddof=1)

    Sigma_F = F.cov().values
    D = np.diag(resid_var)

    var_systematic = float(w @ B @ Sigma_F @ B.T @ w)
    var_idiosyncratic = float(w @ D @ w)
    var_total = var_systematic + var_idiosyncratic
    pct_systematic = 100.0 * var_systematic / var_total
    pct_idiosyncratic = 100.0 * var_idiosyncratic / var_total

    return {
        "B": B, "D": D, "Sigma_F": Sigma_F,
        "var_systematic": var_systematic,
        "var_idiosyncratic": var_idiosyncratic,
        "var_total": var_total,
        "pct_systematic": pct_systematic,
        "pct_idiosyncratic": pct_idiosyncratic,
    }
```

Also add `import statsmodels.api as sm` at the top imports — but **note** that statsmodels is already in the project (used by the equity notebook); confirm via:
```powershell
python -c "import statsmodels; print(statsmodels.__version__)"
```
If it errors, add `statsmodels` to `requirements.txt` and `pip install statsmodels` before continuing.

- [ ] **Step 4: Run test to verify it passes**

```powershell
pytest tests/test_var.py::test_decomposition_sums_close_to_empirical_variance -v
```
Expected: PASS.

- [ ] **Step 5: Run the full test file**

```powershell
pytest tests/test_var.py -v
```
Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add tests/test_var.py
git commit -m "test(var): add factor/idiosyncratic decomposition helper and test"
```

---

## Task 5: Insert Stressed VaR cells into `main.ipynb`

**Files:**
- Modify: `main.ipynb` (insert 1 markdown + 2 code cells after the existing parametric-VaR cell, anchored on its `§2.2A — Parametric (Variance-Covariance) VaR` comment)

> **Why this anchor (not the historical-VaR cell):** In the current notebook the execution order is *historical → parametric → MC* (labels §2.2C, §2.2A, §2.2B), and the overlay plot needs `VaR_95_param_n`, which is defined inside the parametric cell. Anchoring on parametric guarantees every variable referenced by the plot is in scope. The narrative still reads cleanly: "full sample (historical / parametric / MC) vs stressed."

Notebook edits are done via a small `nbformat` script that locates the anchor cell by source-text match and inserts new cells directly after it. This is reproducible and reviewable.

- [ ] **Step 1: Create the insertion script**

Create `scripts/insert_stressed_var_cells.py`:

```python
"""
Insert Stressed VaR cells (markdown + 2 code cells) into main.ipynb,
directly after the cell containing '§2.2A — Parametric (Variance-Covariance) VaR'.
We anchor on parametric (not historical) because the overlay plot references
VaR_95_param_n, which is defined in the parametric cell — and in the current
notebook, historical runs before parametric.
Idempotent: refuses to insert if the marker already appears just after the anchor.
"""
import nbformat
from pathlib import Path

NB = Path("main.ipynb")
ANCHOR = "§2.2A — Parametric (Variance-Covariance) VaR"
MARKER = "§2.2D — Stressed VaR (crisis-window historical)"

nb = nbformat.read(NB, as_version=4)

# Locate anchor
anchor_idx = None
for i, cell in enumerate(nb.cells):
    if cell.cell_type == "code" and ANCHOR in cell.source:
        anchor_idx = i
        break
if anchor_idx is None:
    raise SystemExit(f"Anchor not found: {ANCHOR!r}")

# Idempotency check
for cell in nb.cells[anchor_idx + 1:anchor_idx + 4]:
    if MARKER in cell.source:
        print("Stressed VaR cells already inserted; nothing to do.")
        raise SystemExit(0)

md_cell = nbformat.v4.new_markdown_cell(
    "### 2.2D — Stressed VaR\n"
    "\n"
    "Historical VaR/CVaR computed over crisis windows defined in "
    "`config/funds.yaml` (`var.stress_windows`). The point is to see how the "
    "tail would have looked **during** a stress regime, not on average. "
    "When stressed VaR is materially larger than full-sample VaR, models "
    "trained on the full sample under-price tail risk relative to crisis-"
    "conditional levels."
)

code_cell_compute = nbformat.v4.new_code_cell(
    "# §2.2D — Stressed VaR (crisis-window historical)\n"
    "import yaml\n"
    "\n"
    "with open('config/funds.yaml') as f:\n"
    "    _cfg = yaml.safe_load(f)\n"
    "stress_windows = _cfg['var']['stress_windows']\n"
    "primary_stress = _cfg['var']['primary_stress_window']\n"
    "\n"
    "stressed_var_results = {}\n"
    "for name, (start, end) in stress_windows.items():\n"
    "    sample = portfolio_pnl.loc[start:end]\n"
    "    if len(sample) == 0:\n"
    "        print(f\"  WARNING: window {name} ({start} to {end}) has 0 obs; skipping\")\n"
    "        continue\n"
    "    q_95 = np.quantile(sample, 0.05)\n"
    "    q_99 = np.quantile(sample, 0.01)\n"
    "    stressed_var_results[name] = {\n"
    "        'VaR_95': -q_95,\n"
    "        'VaR_99': -q_99,\n"
    "        'CVaR_95': -sample[sample <= q_95].mean(),\n"
    "        'CVaR_99': -sample[sample <= q_99].mean(),\n"
    "        'n_obs': len(sample),\n"
    "        'start': str(sample.index.min().date()),\n"
    "        'end':   str(sample.index.max().date()),\n"
    "    }\n"
    "    print(f\"Stressed {name} ({sample.index.min().date()} to {sample.index.max().date()}, n={len(sample)}): \"\n"
    "          f\"VaR95={-q_95:.4%}  VaR99={-q_99:.4%}\")\n"
    "\n"
    "# Combine with full-sample historical for direct comparison\n"
    "_cols = {}\n"
    "for _hist_label in ['Historical 1Y', 'Historical 3Y']:\n"
    "    _r = hist_var_results[_hist_label]\n"
    "    _cols[_hist_label] = [_r['VaR_95'], _r['VaR_99'], _r['CVaR_95'], _r['CVaR_99']]\n"
    "for _name, _r in stressed_var_results.items():\n"
    "    _cols[f'Stressed {_name}'] = [_r['VaR_95'], _r['VaR_99'], _r['CVaR_95'], _r['CVaR_99']]\n"
    "stressed_summary = pd.DataFrame(\n"
    "    _cols, index=['VaR 95%', 'VaR 99%', 'CVaR 95%', 'CVaR 99%']\n"
    ")\n"
    "stressed_summary.map(lambda x: f'{x:.2%}')\n"
)

code_cell_plot = nbformat.v4.new_code_cell(
    "# §2.2D — Distribution overlay: full sample vs primary stress window\n"
    "primary = stressed_var_results[primary_stress]\n"
    "stress_slice = portfolio_pnl.loc[primary['start']:primary['end']]\n"
    "VaR_95_hist_full = hist_var_results['Historical 3Y']['VaR_95']\n"
    "VaR_95_stress = primary['VaR_95']\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(10, 6))\n"
    "ax.hist(portfolio_pnl, bins=80, density=True, alpha=0.5, color='steelblue',\n"
    "        label=f'Full sample (n={len(portfolio_pnl)})', edgecolor='white')\n"
    "ax.hist(stress_slice, bins=20, density=True, alpha=0.5, color='crimson',\n"
    "        label=f'{primary_stress} stress (n={len(stress_slice)})', edgecolor='white')\n"
    "ax.axvline(-VaR_95_hist_full, color='steelblue', linestyle='--', linewidth=1.5,\n"
    "           label=f'Historical VaR 95% (full): {VaR_95_hist_full:.2%}')\n"
    "ax.axvline(-VaR_95_param_n,   color='black',     linestyle=':',  linewidth=1.5,\n"
    "           label=f'Parametric normal VaR 95%: {VaR_95_param_n:.2%}')\n"
    "ax.axvline(-VaR_95_stress,    color='crimson',   linestyle='--', linewidth=1.5,\n"
    "           label=f'Stressed VaR 95% ({primary_stress}): {VaR_95_stress:.2%}')\n"
    "ax.set_xlabel('Daily portfolio P&L')\n"
    "ax.set_ylabel('Density')\n"
    "ax.set_title('LC Fund Proxy — P&L Distribution: Full Sample vs Stress', fontweight='bold')\n"
    "ax.legend(loc='upper left', fontsize=9)\n"
    "fig.tight_layout()\n"
    "plt.savefig('data/output/var_stress_overlay.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()\n"
)

nb.cells = nb.cells[:anchor_idx + 1] + [md_cell, code_cell_compute, code_cell_plot] + nb.cells[anchor_idx + 1:]
nbformat.write(nb, NB)
print(f"Inserted 3 cells after anchor at index {anchor_idx}.")
```

Create the `scripts/` directory if it doesn't exist:
```powershell
if (-not (Test-Path scripts)) { New-Item -ItemType Directory scripts }
```

- [ ] **Step 2: Run the insertion script**

```powershell
python scripts/insert_stressed_var_cells.py
```
Expected: `Inserted 3 cells after anchor at index N.`

- [ ] **Step 3: Verify the cells inserted by re-running the script**

```powershell
python scripts/insert_stressed_var_cells.py
```
Expected: `Stressed VaR cells already inserted; nothing to do.` (idempotency check).

- [ ] **Step 4: Execute the notebook to make sure it still runs end-to-end**

```powershell
jupyter nbconvert --to notebook --execute main.ipynb --output main.ipynb
```
Expected: completes without errors. The new `data/output/var_stress_overlay.png` is created.

Verify:
```powershell
if (Test-Path data/output/var_stress_overlay.png) { "OK: overlay PNG written" } else { "FAIL: PNG missing" }
```
Expected: `OK: overlay PNG written`.

- [ ] **Step 5: Commit**

```powershell
git add main.ipynb scripts/insert_stressed_var_cells.py data/output/var_stress_overlay.png
git commit -m "feat(var): add stressed VaR + distribution overlay (Module 2)"
```

---

## Task 6: Insert multi-ν parametric-t table cells into `main.ipynb`

**Files:**
- Modify: `main.ipynb` (insert 1 markdown + 1 code cell after the existing parametric-VaR cell, anchored on its `§2.2A — Parametric (Variance-Covariance) VaR` comment)

- [ ] **Step 1: Create the insertion script**

Create `scripts/insert_multi_nu_cells.py`:

```python
"""
Insert multi-nu parametric-t comparison cells into main.ipynb,
directly after the cell containing '§2.2A — Parametric (Variance-Covariance) VaR'.
"""
import nbformat
from pathlib import Path

NB = Path("main.ipynb")
ANCHOR = "§2.2A — Parametric (Variance-Covariance) VaR"
MARKER = "§2.2A.bis — Multi-nu parametric-t comparison"

nb = nbformat.read(NB, as_version=4)

anchor_idx = None
for i, cell in enumerate(nb.cells):
    if cell.cell_type == "code" and ANCHOR in cell.source:
        anchor_idx = i
        break
if anchor_idx is None:
    raise SystemExit(f"Anchor not found: {ANCHOR!r}")

for cell in nb.cells[anchor_idx + 1:anchor_idx + 3]:
    if MARKER in cell.source:
        print("Multi-nu cells already inserted; nothing to do.")
        raise SystemExit(0)

md_cell = nbformat.v4.new_markdown_cell(
    "### 2.2A.bis — Parametric-t sensitivity to degrees of freedom\n"
    "\n"
    "The MLE-fit above pins a single ν. To make the model's sensitivity to ν "
    "explicit, the table below reports parametric VaR/CVaR for a grid "
    "ν ∈ {4, 5, 8, 20, ∞}. The variance correction "
    "`scale = σ · √((ν − 2) / ν)` keeps the scaled-t standard deviation "
    "matched to the sample, so the rows compare like-for-like. The ν → ∞ row "
    "reproduces the normal parametric numbers. Lower ν → fatter tails → "
    "larger 99% VaR."
)

code_cell = nbformat.v4.new_code_cell(
    "# §2.2A.bis — Multi-nu parametric-t comparison\n"
    "nu_grid = [4, 5, 8, 20]\n"
    "rows = []\n"
    "for nu in nu_grid:\n"
    "    s = sigma * np.sqrt((nu - 2) / nu)\n"
    "    var_95 = -(mu + t_dist.ppf(0.05, df=nu) * s)\n"
    "    var_99 = -(mu + t_dist.ppf(0.01, df=nu) * s)\n"
    "    cvar_95 = -portfolio_pnl[portfolio_pnl <= -var_95].mean()\n"
    "    cvar_99 = -portfolio_pnl[portfolio_pnl <= -var_99].mean()\n"
    "    rows.append([f'nu={nu}', var_95, var_99, cvar_95, cvar_99])\n"
    "\n"
    "# nu -> infinity (normal)\n"
    "rows.append(['nu -> inf', VaR_95_param_n, VaR_99_param_n,\n"
    "             CVaR_95_param_n, CVaR_99_param_n])\n"
    "\n"
    "nu_table = pd.DataFrame(rows, columns=['df', 'VaR 95%', 'VaR 99%',\n"
    "                                       'CVaR 95%', 'CVaR 99%']).set_index('df')\n"
    "print(f'MLE-fitted nu was {nu_fit:.1f}; comparison grid:')\n"
    "nu_table.map(lambda x: f'{x:.2%}')\n"
)

nb.cells = nb.cells[:anchor_idx + 1] + [md_cell, code_cell] + nb.cells[anchor_idx + 1:]
nbformat.write(nb, NB)
print(f"Inserted 2 cells after anchor at index {anchor_idx}.")
```

- [ ] **Step 2: Run the insertion script**

```powershell
python scripts/insert_multi_nu_cells.py
```
Expected: `Inserted 2 cells after anchor at index N.`

- [ ] **Step 3: Verify idempotency**

```powershell
python scripts/insert_multi_nu_cells.py
```
Expected: `Multi-nu cells already inserted; nothing to do.`

- [ ] **Step 4: Execute the notebook end-to-end**

```powershell
jupyter nbconvert --to notebook --execute main.ipynb --output main.ipynb
```
Expected: completes without errors.

- [ ] **Step 5: Commit**

```powershell
git add main.ipynb scripts/insert_multi_nu_cells.py
git commit -m "feat(var): add multi-nu parametric-t comparison table (Module 2)"
```

---

## Task 7: Insert factor / idiosyncratic decomposition cells into `main.ipynb`

**Files:**
- Modify: `main.ipynb` (insert 1 markdown + 2 code cells immediately before the backtest section, anchored on the cell containing the `kupiec_pof` function definition)

The decomposition logically belongs *before* the backtests, since it is descriptive (about the portfolio's variance structure) rather than predictive (about the VaR model's calibration). Anchor on the cell that defines `kupiec_pof`.

- [ ] **Step 1: Create the insertion script**

Create `scripts/insert_decomposition_cells.py`:

```python
"""
Insert factor (PC) vs idiosyncratic variance decomposition cells into
main.ipynb, immediately BEFORE the backtest cell (the one that defines
kupiec_pof). Anchor: 'def kupiec_pof'.
"""
import nbformat
from pathlib import Path

NB = Path("main.ipynb")
ANCHOR = "def kupiec_pof"
MARKER = "§2.3 — Factor (PC) vs idiosyncratic variance decomposition"

nb = nbformat.read(NB, as_version=4)

anchor_idx = None
for i, cell in enumerate(nb.cells):
    if cell.cell_type == "code" and ANCHOR in cell.source:
        anchor_idx = i
        break
if anchor_idx is None:
    raise SystemExit(f"Anchor not found: {ANCHOR!r}")

for cell in nb.cells[max(0, anchor_idx - 3):anchor_idx]:
    if MARKER in cell.source:
        print("Decomposition cells already inserted; nothing to do.")
        raise SystemExit(0)

md_cell = nbformat.v4.new_markdown_cell(
    "### 2.3 — Factor (PC) vs idiosyncratic variance decomposition\n"
    "\n"
    "How much of the LC fund's daily yield-change variance is driven by the "
    "common EM rate factors (PC1 level / PC2 slope / PC3 curvature from "
    "Module 1) versus country-specific noise? For each of the four LC "
    "countries, regress its 5Y daily yield change on the panel PC scores, "
    "stack the country β's into matrix B, and decompose:\n"
    "\n"
    "$$\\mathrm{Var}(\\mathbf{w}^\\top \\Delta y) = \\mathbf{w}^\\top B \\Sigma_F B^\\top \\mathbf{w} + \\mathbf{w}^\\top D \\mathbf{w}$$\n"
    "\n"
    "where $\\Sigma_F = \\mathrm{Cov}(\\text{PCs})$ and $D = \\mathrm{diag}(\\mathrm{Var}(\\epsilon_c))$. "
    "Cross-country residual correlation is ignored (matches the equity-project "
    "methodology); empirical $\\mathrm{Var}(\\mathbf{w}^\\top \\Delta y)$ is shown alongside the "
    "decomposition for transparency.\n"
    "\n"
    "A high systematic share means the fund's daily risk is dominated by "
    "global EM rate factors and diversification across the four countries is "
    "limited."
)

code_cell_compute = nbformat.v4.new_code_cell(
    "# §2.3 — Factor (PC) vs idiosyncratic variance decomposition\n"
    "import statsmodels.api as sm\n"
    "\n"
    "# Align PC scores with the 4-country 5Y proxy index used to build portfolio_pnl\n"
    "_dec_common = panel_scores_df.index.intersection(proxy_dy.index)\n"
    "F = panel_scores_df.loc[_dec_common]\n"
    "Y = proxy_dy.loc[_dec_common]\n"
    "pc_cols = list(F.columns)\n"
    "\n"
    "n_series = Y.shape[1]\n"
    "n_factors = F.shape[1]\n"
    "B = np.zeros((n_series, n_factors))\n"
    "resid_var = np.zeros(n_series)\n"
    "\n"
    "X_dec = sm.add_constant(F.values)\n"
    "for i, col in enumerate(Y.columns):\n"
    "    _m = sm.OLS(Y[col].values, X_dec).fit()\n"
    "    B[i, :] = _m.params[1:]\n"
    "    resid_var[i] = _m.resid.var(ddof=1)\n"
    "\n"
    "Sigma_F = F.cov().values\n"
    "D = np.diag(resid_var)\n"
    "\n"
    "# w_vec already defined in §2.1\n"
    "var_systematic    = float(w_vec @ B @ Sigma_F @ B.T @ w_vec)\n"
    "var_idiosyncratic = float(w_vec @ D @ w_vec)\n"
    "var_total         = var_systematic + var_idiosyncratic\n"
    "pct_systematic    = 100.0 * var_systematic / var_total\n"
    "pct_idiosyncratic = 100.0 * var_idiosyncratic / var_total\n"
    "\n"
    "var_empirical = float(np.var(Y.values @ w_vec, ddof=1))\n"
    "\n"
    "print('Decomposition of Var(weighted 5Y yield change):')\n"
    "print(f'  Systematic    (PC1/PC2/PC3): {pct_systematic:6.2f}%')\n"
    "print(f'  Idiosyncratic (residuals):  {pct_idiosyncratic:6.2f}%')\n"
    "print()\n"
    "print(f'  Decomposition total:        {var_total:.6e}')\n"
    "print(f'  Empirical Var(w. delta_y):  {var_empirical:.6e}')\n"
    "print(f'  Difference attributable to cross-country residual correlation: '\n"
    "      f'{(var_empirical - var_total) / var_empirical * 100:+.2f}%')\n"
    "\n"
    "B_df = pd.DataFrame(B, index=Y.columns, columns=pc_cols)\n"
    "B_df.style.format('{:.3f}')\n"
)

code_cell_plot = nbformat.v4.new_code_cell(
    "# §2.3 — Bar chart of the decomposition\n"
    "fig, ax = plt.subplots(figsize=(7, 4))\n"
    "bars = ax.bar(['Systematic (PCs)', 'Idiosyncratic'],\n"
    "              [pct_systematic, pct_idiosyncratic],\n"
    "              color=['#1f77b4', '#7f7f7f'])\n"
    "ax.set_ylabel('% of total decomposed variance')\n"
    "ax.set_title('LC Fund Proxy — Daily Yield-Change Variance Decomposition',\n"
    "             fontweight='bold')\n"
    "ax.bar_label(bars, fmt='%.2f%%')\n"
    "ax.set_ylim(0, max(pct_systematic, pct_idiosyncratic) * 1.15)\n"
    "fig.tight_layout()\n"
    "plt.savefig('data/output/var_risk_decomposition.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()\n"
)

# Insert BEFORE the anchor (the kupiec_pof cell)
nb.cells = nb.cells[:anchor_idx] + [md_cell, code_cell_compute, code_cell_plot] + nb.cells[anchor_idx:]
nbformat.write(nb, NB)
print(f"Inserted 3 cells before anchor at index {anchor_idx}.")
```

- [ ] **Step 2: Run the insertion script**

```powershell
python scripts/insert_decomposition_cells.py
```
Expected: `Inserted 3 cells before anchor at index N.`

- [ ] **Step 3: Verify idempotency**

```powershell
python scripts/insert_decomposition_cells.py
```
Expected: `Decomposition cells already inserted; nothing to do.`

- [ ] **Step 4: Execute the notebook end-to-end**

```powershell
jupyter nbconvert --to notebook --execute main.ipynb --output main.ipynb
```
Expected: completes without errors. `data/output/var_risk_decomposition.png` is created.

Verify:
```powershell
if (Test-Path data/output/var_risk_decomposition.png) { "OK: decomposition PNG written" } else { "FAIL: PNG missing" }
```
Expected: `OK: decomposition PNG written`.

- [ ] **Step 5: Commit**

```powershell
git add main.ipynb scripts/insert_decomposition_cells.py data/output/var_risk_decomposition.png
git commit -m "feat(var): add factor (PC) vs idiosyncratic variance decomposition (Module 2)"
```

---

## Task 8: Final verification — full test suite + end-to-end notebook run

**Files:** none modified. This is a verification-only task.

- [ ] **Step 1: Run the full test suite**

```powershell
pytest tests/ -v
```
Expected: all tests pass, including the three new ones added in Tasks 2, 3, 4.

- [ ] **Step 2: Re-execute the notebook from scratch**

```powershell
jupyter nbconvert --to notebook --execute main.ipynb --output main.ipynb
```
Expected: completes without errors end-to-end.

- [ ] **Step 3: Verify all three new output artifacts exist**

```powershell
$artifacts = @(
  "data/output/var_stress_overlay.png",
  "data/output/var_risk_decomposition.png",
  "data/output/var_pnl_bands.png"
)
foreach ($a in $artifacts) {
  if (Test-Path $a) { "OK: $a" } else { "FAIL: $a missing" }
}
```
Expected: three `OK:` lines.

- [ ] **Step 4: Visually inspect the notebook (optional, manual)**

Open `main.ipynb` in Jupyter and skim Module 2. Confirm the three new sections appear in the right order:
1. Historical VaR → **Stressed VaR (new)** → Parametric VaR → **Multi-ν table (new)** → Monte Carlo → Summary → **Factor/Idio decomposition (new)** → Kupiec/Christoffersen backtest → P&L band plot.

No commit needed for this verification task.

---

## Self-review notes

- **Spec coverage:** Every section of the spec (Stressed VaR + overlay, multi-ν table, decomposition, three tests, config block, narrative commentary) has a task. ✓
- **Order of cell placement:** The spec said the decomposition goes "before the backtest." Task 7 anchors on `def kupiec_pof` and inserts *before* that cell, satisfying the spec. ✓
- **No placeholders.** Every code step contains full code or an exact command. ✓
- **No external-dependency-not-installed risk:** `statsmodels` is checked in Task 4 Step 3 and surfaced to the user if missing.
- **Idempotency:** Each nbformat script refuses to re-insert if its marker is already present. Safe to re-run if a later task is reverted.
- **Working tree:** This plan modifies `app.py` only incidentally if at all — the working tree has unrelated dirty state (`app.py`, `2ndphoto.jpg`, `3rdphoto.jpg`) noted at session start; do not stage those files when running `git add` commands above.
