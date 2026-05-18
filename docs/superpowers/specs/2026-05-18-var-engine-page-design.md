# VaR Engine Page — Design

**Date:** 2026-05-18
**Source:** Port the three Module 2 additions from `main.ipynb` (stressed VaR + overlay, multi-ν parametric-t table, factor/idiosyncratic decomposition — see `docs/superpowers/specs/2026-05-18-fi-var-improvements-design.md`) into the Streamlit app at `app.py`, on the existing `VaR Engine` page.

## Goal

Surface the three new Module 2 outputs in the Streamlit dashboard so a PM can read stressed VaR, parametric-t sensitivity, and the PC/idiosyncratic risk split without opening the notebook. Keeps the app's offline-viewer pattern (reads pre-generated artifacts from `data/output/`, never recomputes).

## Out of scope

- Live recomputation in the app — all numbers come from notebook outputs.
- New VaR methods or statistical tests beyond what Module 2 already produces.
- User-editable stress windows (config-driven via `funds.yaml`).
- Changes to any page other than `VaR Engine`. The `Portfolios` page already has its own `VaR` tab; it is untouched.

## Compute strategy

Pre-generated. The notebook writes seven new artifacts to `data/output/`; the app reads them. Matches the existing pattern (`CLAUDE.md`: *"Offline-capable demo: reads pre-generated outputs from data/output/"*).

## Page structure

The existing 15-line `VaR Engine` block in `app.py` (currently a single section card displaying `var_pnl_bands.png`) is replaced by a 4-tab layout:

```
VaR Engine
├── Tab 1: P&L Bands               (existing content; unchanged)
├── Tab 2: Stressed VaR            (new)
├── Tab 3: Parametric-t Sensitivity (new)
└── Tab 4: Risk Decomposition      (new)
```

All four tabs follow the same fall-back pattern: if any required artifact is missing, render `st.warning("...artifact-name... not found. Run Module 2 (VaR Engine) in the notebook first.")` and skip the rest of the tab.

## Notebook side — artifacts to emit

A new code cell at the end of Module 2 (after the §2.3 decomposition plot, before the backtest section) writes the seven files below. The cell reads variables already in scope; it does not recompute anything. Inserted via a fourth idempotent `scripts/insert_*.py` script.

| File | Source variable(s) | Contents |
|---|---|---|
| `var_portfolio_pnl.csv` | `portfolio_pnl` | Date-indexed single-column daily P&L series (~1000 rows). |
| `var_stressed_summary.csv` | `stressed_summary` | DataFrame with rows `VaR 95% / VaR 99% / CVaR 95% / CVaR 99%`, columns `Historical 1Y`, `Historical 3Y`, and one `Stressed <name>` per window. Stored as raw floats (not pre-formatted percent strings) so the app can format on render. |
| `var_stress_windows.json` | `stressed_var_results`, `hist_var_results`, `VaR_95_param_n`, `primary_stress` | `{ "primary_stress": "COVID", "reference": {"hist_full_VaR_95": ..., "parametric_normal_VaR_95": ...}, "windows": { "COVID": {"start": "...", "end": "...", "n_obs": ..., "VaR_95": ..., "VaR_99": ..., "CVaR_95": ..., "CVaR_99": ...}, "Ukraine_Fed": {...} } }` |
| `var_multi_nu_table.csv` | `nu_table` | The ν-grid DataFrame as raw floats. |
| `var_multi_nu_fit.json` | `nu_fit` | Single-key metadata file `{"nu_fit": ...}` so the app can render the MLE-fitted ν above the grid. |
| `var_decomposition.json` | scalars from §2.3 | `{ "pct_systematic": ..., "pct_idiosyncratic": ..., "var_total": ..., "var_empirical": ..., "residual_corr_gap_pct": ..., "pc_cols": ["PC1 (global level)", ...] }`. |
| `var_decomposition_betas.csv` | `B_df` | 4-country × 3-PC β matrix with country labels as index and PC labels as columns. |

The cell is wrapped in a `try / except FileNotFoundError` so the notebook still completes if `data/output/` does not yet exist — it creates the directory first via `Path("data/output").mkdir(parents=True, exist_ok=True)`.

## App side — per-tab content

### Tab 1: P&L Bands (existing)

No change. The current image and captions stay verbatim. The only modification is wrapping them in `with tab1:` instead of the bare `elif page == "VaR Engine":` block.

### Tab 2: Stressed VaR (new)

Reads `var_portfolio_pnl.csv`, `var_stress_windows.json`, `var_stressed_summary.csv`.

UI:
1. `st.selectbox("Stress window to overlay", options=list(windows.keys()), index=list(windows.keys()).index(primary_stress))`.
2. Plotly figure with two `go.Histogram` traces (`histnorm='probability density'`, `opacity=0.5`, `barmode='overlay'`):
   - Full-sample P&L (blue, `#4682B4` steelblue).
   - Selected stress slice (crimson `#DC143C`).
3. Three `fig.add_vline` calls for the loss tail (`x = -VaR_95`):
   - Historical full-sample VaR 95% (steelblue dashed).
   - Parametric normal VaR 95% (black dotted).
   - Selected stressed VaR 95% (crimson dashed).
4. `st.dataframe` of `var_stressed_summary.csv`, percentage-formatted (4 decimals).
5. Caption (1–2 sentences) computed from the data: *"`{selected}` stress window 99% VaR is `{ratio:.1f}×` the full-sample historical 99% VaR — models calibrated on the full sample under-price crisis-conditional tail risk by this multiple."*

### Tab 3: Parametric-t Sensitivity (new)

Reads `var_multi_nu_table.csv`, `var_multi_nu_fit.json`.

UI:
1. `st.markdown(f"**MLE-fitted ν = {nu_fit:.1f}**; comparison grid:")`.
2. `st.dataframe` of the ν-grid table, percentage-formatted (4 decimals).
3. Plotly line chart: x-axis = ν (with `∞` plotted at a fixed right-side x-position labeled "∞" via `tickvals`/`ticktext`), y-axis = 99% VaR (%). Marker at `nu_fit` if it falls inside the plotted range. Title: *"99% VaR vs degrees of freedom"*.
4. Caption explaining the variance correction `σ · √((ν − 2) / ν)` and the monotonicity (lower ν → fatter tails → larger 99% VaR).

### Tab 4: Risk Decomposition (new)

Reads `var_decomposition.json`, `var_decomposition_betas.csv`.

UI:
1. Two-column layout (`col1, col2 = st.columns([2, 1])`):
   - **Left (col1):** Plotly horizontal bar chart `go.Bar` with two bars (`Systematic (PCs)` blue `#1f77b4`, `Idiosyncratic` grey `#7f7f7f`), x-axis in percent, bar labels `f"{pct:.2f}%"`. Title: *"Daily Yield-Change Variance Decomposition"*.
   - **Right (col2):** Three `st.metric` cards: `Decomposition total` (`var_total` in scientific notation), `Empirical Var(w'Δy)` (`var_empirical`), `Residual-corr gap` (`residual_corr_gap_pct` as `±X.XX%`).
2. Below the two columns: `st.dataframe` of the β matrix, 3-decimal formatting.
3. `st.latex` rendering of the decomposition identity (copied from the notebook §2.3 markdown):
   ```
   \mathrm{Var}(\mathbf{w}^\top \Delta y) = \mathbf{w}^\top B \Sigma_F B^\top \mathbf{w} + \mathbf{w}^\top D \mathbf{w}
   ```
4. Caption (1–2 sentences) on the systematic-vs-idiosyncratic split and what a high systematic share implies for diversification across the four LC countries.

## Loader helpers

Three new module-level functions in `app.py`, each decorated with `@st.cache_data`:

```python
@st.cache_data
def _load_stress_data() -> dict | None:
    """Returns {'pnl': Series, 'windows': dict, 'summary': DataFrame} or None if any file missing."""

@st.cache_data
def _load_multi_nu() -> dict | None:
    """Returns {'table': DataFrame, 'nu_fit': float} or None."""

@st.cache_data
def _load_decomposition() -> dict | None:
    """Returns {'scalars': dict, 'betas': DataFrame} or None."""
```

Each helper checks every file it needs; if any is missing it returns `None` and the calling tab renders the warning branch. Placed near the existing `_load_portfolio_data` helper around `app.py:1067`.

## Implementation footprint

- `app.py`: replace the 15-line `VaR Engine` block (lines 991–1005) with a ~150-line block (4 tabs, 3 loader helpers, 3 Plotly figures, captions).
- `main.ipynb`: insert one new code cell at the end of Module 2 via a fourth idempotent script.
- `scripts/insert_artifact_dump_cell.py`: new, mirrors the three existing insertion scripts.
- 7 new artifacts in `data/output/` generated on the next notebook run.

## Tests

No new tests. The math is already covered by the three tests added in `tests/test_var.py` from the prior spec. The new artifact-dump cell is a pure serialization step (reads in-scope variables, writes files); a unit test would just exercise `pandas.to_csv` and `json.dump`. The app rendering layer (Plotly + Streamlit widgets) is not unit-tested elsewhere in this project, and adding ad-hoc Streamlit tests for one page would be inconsistent with the project's existing test scope.

Manual verification on next notebook run:
- Seven new files appear in `data/output/`.
- `streamlit run app.py` loads `VaR Engine`, all four tabs render without warnings.
- Stressed-VaR dropdown switches between COVID and Ukraine_Fed and updates the overlay + vertical line.

## Reproducibility

- The artifact-dump cell is deterministic (no randomness).
- File paths and stress windows come from `config/funds.yaml`, not from app-side state.
- The insertion script is idempotent (refuses to re-insert if its marker is already present), matching the pattern of the three existing scripts.
