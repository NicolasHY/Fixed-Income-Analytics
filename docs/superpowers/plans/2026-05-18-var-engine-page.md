# VaR Engine Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the three Module 2 VaR additions (stressed VaR, multi-ν parametric-t, PC/idiosyncratic decomposition) from `main.ipynb` into the Streamlit dashboard's `VaR Engine` page (`app.py`), reading pre-generated artifacts and rendering with Plotly.

**Architecture:** Notebook adds one new code cell at the end of Module 2 that dumps seven sidecar artifacts to `data/output/` (raw floats, JSON metadata, no plots). The existing 15-line `VaR Engine` page block is replaced with a 4-tab Streamlit layout: Tab 1 keeps the existing P&L Bands content unchanged; Tabs 2–4 are new and read the sidecars via three cached loader helpers. All new charts are Plotly (interactive).

**Tech Stack:** Python 3, pandas, numpy, json (stdlib), streamlit, plotly.graph_objects, nbformat (for the insertion script). All already in `requirements.txt` or transitively present.

**Spec reference:** `docs/superpowers/specs/2026-05-18-var-engine-page-design.md`.

---

## File Structure

- **Create:** `scripts/insert_artifact_dump_cell.py` — idempotent inserter that adds the artifact-dump cell to `main.ipynb`.
- **Modify:** `main.ipynb` — one new code cell at the end of Module 2 (inserted before the `def kupiec_pof` cell, after the §2.3 decomposition plot).
- **Modify:** `app.py` — three new loader helpers near `_load_portfolio_data` (~line 1067); the `VaR Engine` page block (lines 991–1005) replaced with a 4-tab layout.
- **Generated (not source):** 7 new files in `data/output/`:
  - `var_portfolio_pnl.csv`
  - `var_stressed_summary.csv`
  - `var_stress_windows.json`
  - `var_multi_nu_table.csv`
  - `var_multi_nu_fit.json`
  - `var_decomposition.json`
  - `var_decomposition_betas.csv`

Each task below produces one self-contained commit.

---

## Task 1: Add artifact-dump cell to `main.ipynb`

**Files:**
- Create: `scripts/insert_artifact_dump_cell.py`
- Modify (via script): `main.ipynb`

The new cell relies on these in-scope variables, all defined upstream in Module 2 (verified against the current notebook): `portfolio_pnl`, `stressed_summary`, `stressed_var_results`, `primary_stress`, `hist_var_results`, `VaR_95_param_n`, `nu_table`, `nu_fit`, `pct_systematic`, `pct_idiosyncratic`, `var_total`, `var_empirical`, `pc_cols`, `B_df`.

- [ ] **Step 1: Create the insertion script**

Create `scripts/insert_artifact_dump_cell.py`:

```python
"""
Insert the VaR Engine artifact-dump cell into main.ipynb, immediately BEFORE
the backtest cell (the one that defines kupiec_pof). Anchor: 'def kupiec_pof'.
Idempotent: refuses to insert if the marker already appears just before the anchor.
"""
import nbformat
from pathlib import Path

NB = Path("main.ipynb")
ANCHOR = "def kupiec_pof"
MARKER = "§2.4 — VaR Engine artifact dump"

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
        print("Artifact-dump cell already inserted; nothing to do.")
        raise SystemExit(0)

code_cell = nbformat.v4.new_code_cell(
    "# §2.4 — VaR Engine artifact dump (sidecar files for the Streamlit page)\n"
    "import json\n"
    "from pathlib import Path\n"
    "\n"
    "_out = Path('data/output')\n"
    "_out.mkdir(parents=True, exist_ok=True)\n"
    "\n"
    "# 1. Portfolio P&L (full sample, date-indexed)\n"
    "portfolio_pnl.to_csv(_out / 'var_portfolio_pnl.csv', header=['pnl'])\n"
    "\n"
    "# 2. Stressed VaR summary (raw floats)\n"
    "stressed_summary.to_csv(_out / 'var_stressed_summary.csv')\n"
    "\n"
    "# 3. Stress windows + reference VaR values\n"
    "_stress_payload = {\n"
    "    'primary_stress': primary_stress,\n"
    "    'reference': {\n"
    "        'hist_full_VaR_95': float(hist_var_results['Historical 3Y']['VaR_95']),\n"
    "        'parametric_normal_VaR_95': float(VaR_95_param_n),\n"
    "    },\n"
    "    'windows': stressed_var_results,\n"
    "}\n"
    "with open(_out / 'var_stress_windows.json', 'w') as f:\n"
    "    json.dump(_stress_payload, f, indent=2, default=str)\n"
    "\n"
    "# 4. Multi-nu grid table (raw floats)\n"
    "nu_table.to_csv(_out / 'var_multi_nu_table.csv')\n"
    "\n"
    "# 5. MLE-fitted nu metadata\n"
    "with open(_out / 'var_multi_nu_fit.json', 'w') as f:\n"
    "    json.dump({'nu_fit': float(nu_fit)}, f)\n"
    "\n"
    "# 6. Decomposition scalars\n"
    "_decomp_payload = {\n"
    "    'pct_systematic':       float(pct_systematic),\n"
    "    'pct_idiosyncratic':    float(pct_idiosyncratic),\n"
    "    'var_total':            float(var_total),\n"
    "    'var_empirical':        float(var_empirical),\n"
    "    'residual_corr_gap_pct': float((var_empirical - var_total) / var_empirical * 100),\n"
    "    'pc_cols':              list(pc_cols),\n"
    "}\n"
    "with open(_out / 'var_decomposition.json', 'w') as f:\n"
    "    json.dump(_decomp_payload, f, indent=2)\n"
    "\n"
    "# 7. Decomposition betas (B matrix as DataFrame, country x PC)\n"
    "B_df.to_csv(_out / 'var_decomposition_betas.csv')\n"
    "\n"
    "print(f'Wrote 7 VaR Engine artifacts to {_out}/')\n"
)

# Insert BEFORE the anchor (the kupiec_pof cell)
nb.cells = nb.cells[:anchor_idx] + [code_cell] + nb.cells[anchor_idx:]
nbformat.write(nb, NB)
print(f"Inserted 1 cell before anchor at index {anchor_idx}.")
```

- [ ] **Step 2: Run the insertion script**

```powershell
.venv\Scripts\python scripts/insert_artifact_dump_cell.py
```
Expected: `Inserted 1 cell before anchor at index N.`

- [ ] **Step 3: Verify idempotency**

```powershell
.venv\Scripts\python scripts/insert_artifact_dump_cell.py
```
Expected: `Artifact-dump cell already inserted; nothing to do.`

- [ ] **Step 4: Execute the notebook end-to-end**

```powershell
.venv\Scripts\jupyter nbconvert --to notebook --execute main.ipynb --output main.ipynb
```
Expected: completes without errors; the notebook prints `Wrote 7 VaR Engine artifacts to data/output/`.

- [ ] **Step 5: Verify all 7 artifacts exist**

```powershell
$expected = @(
  'data/output/var_portfolio_pnl.csv',
  'data/output/var_stressed_summary.csv',
  'data/output/var_stress_windows.json',
  'data/output/var_multi_nu_table.csv',
  'data/output/var_multi_nu_fit.json',
  'data/output/var_decomposition.json',
  'data/output/var_decomposition_betas.csv'
)
foreach ($p in $expected) {
  if (Test-Path $p) { "OK: $p" } else { "FAIL: $p missing" }
}
```
Expected: seven `OK:` lines.

- [ ] **Step 6: Sanity-check one JSON file**

```powershell
.venv\Scripts\python -c "import json; d = json.load(open('data/output/var_stress_windows.json')); print('primary:', d['primary_stress']); print('windows:', list(d['windows'].keys())); print('ref:', d['reference'])"
```
Expected: prints `primary: COVID`, `windows: ['COVID', 'Ukraine_Fed']`, and numeric reference VaRs.

- [ ] **Step 7: Commit**

```powershell
git add main.ipynb scripts/insert_artifact_dump_cell.py data/output/var_portfolio_pnl.csv data/output/var_stressed_summary.csv data/output/var_stress_windows.json data/output/var_multi_nu_table.csv data/output/var_multi_nu_fit.json data/output/var_decomposition.json data/output/var_decomposition_betas.csv
git commit -m "feat(var): dump Module 2 artifacts for the Streamlit VaR Engine page"
```

---

## Task 2: Add three loader helpers to `app.py`

**Files:**
- Modify: `app.py` — insert three new functions just before the existing `_load_portfolio_data` function (search for `def _load_portfolio_data`; insert above it).

The three helpers return `None` if any required file is missing, so each tab can render a uniform warning branch.

- [ ] **Step 1: Locate the insertion point**

Search `app.py` for `def _load_portfolio_data`. The new helpers go immediately above that line. Confirm the file already imports `json`, `pd`, `np`, `Path`, `st`, and `go` — these are imports at the top of `app.py` (lines 8–17). Confirm `OUT` is defined as a `Path` to `data/output/` (used elsewhere in the file, e.g. `OUT / "var_pnl_bands.png"`).

- [ ] **Step 2: Add the three loaders**

Insert this block immediately above `def _load_portfolio_data`:

```python
@st.cache_data
def _load_stress_data():
    """Load Stressed VaR sidecars. Returns dict or None if any file is missing."""
    pnl_path = OUT / "var_portfolio_pnl.csv"
    win_path = OUT / "var_stress_windows.json"
    sum_path = OUT / "var_stressed_summary.csv"
    if not (pnl_path.exists() and win_path.exists() and sum_path.exists()):
        return None
    pnl = pd.read_csv(pnl_path, index_col=0, parse_dates=True)["pnl"]
    with open(win_path) as f:
        windows = json.load(f)
    summary = pd.read_csv(sum_path, index_col=0)
    return {"pnl": pnl, "windows": windows, "summary": summary}


@st.cache_data
def _load_multi_nu():
    """Load multi-nu parametric-t sidecars. Returns dict or None if any file is missing."""
    table_path = OUT / "var_multi_nu_table.csv"
    fit_path = OUT / "var_multi_nu_fit.json"
    if not (table_path.exists() and fit_path.exists()):
        return None
    table = pd.read_csv(table_path, index_col=0)
    with open(fit_path) as f:
        nu_fit = json.load(f)["nu_fit"]
    return {"table": table, "nu_fit": nu_fit}


@st.cache_data
def _load_decomposition():
    """Load PC/idiosyncratic decomposition sidecars. Returns dict or None."""
    json_path = OUT / "var_decomposition.json"
    betas_path = OUT / "var_decomposition_betas.csv"
    if not (json_path.exists() and betas_path.exists()):
        return None
    with open(json_path) as f:
        scalars = json.load(f)
    betas = pd.read_csv(betas_path, index_col=0)
    return {"scalars": scalars, "betas": betas}


```

(Trailing blank line keeps the helpers visually separated from `_load_portfolio_data`.)

- [ ] **Step 3: Sanity-check the helpers via import**

```powershell
.venv\Scripts\python -c "import ast; ast.parse(open('app.py', encoding='utf-8').read()); print('OK: app.py parses')"
```
Expected: `OK: app.py parses`.

Then a manual call:
```powershell
.venv\Scripts\python -c "import sys; sys.path.insert(0, '.'); from app import _load_stress_data, _load_multi_nu, _load_decomposition; print('stress:', _load_stress_data() is not None); print('multi-nu:', _load_multi_nu() is not None); print('decomp:', _load_decomposition() is not None)"
```
Expected: three `True` lines (the seven artifacts from Task 1 are in place).

If the import fails because `st.cache_data` runs at import time on the helpers — it shouldn't (the decorator just wraps the function; it does not call it). If it errors, drop the decorator temporarily, retry, then put it back.

- [ ] **Step 4: Commit**

```powershell
git add app.py
git commit -m "feat(app): add VaR Engine sidecar loader helpers"
```

---

## Task 3: Convert the VaR Engine page to a 4-tab layout (Tab 1 only)

**Files:**
- Modify: `app.py` — replace the current `elif page == "VaR Engine":` block (lines 991–1005) with a 4-tab structure. Only Tab 1 (existing content, unchanged) is filled in this task. Tabs 2–4 are populated in Tasks 4–6.

- [ ] **Step 1: Replace the VaR Engine block**

Find the block:
```python
# ── VaR Engine ────────────────────────────────────────────────────────────────
elif page == "VaR Engine":
    img_var = OUT / "var_pnl_bands.png"
    if not img_var.exists():
        st.warning("var_pnl_bands.png not found. Run Module 2 (VaR Engine) in the notebook first.")
    else:
        st.markdown("<div class='section-card'><h3>Portfolio P&L with VaR / CVaR Bands</h3>", unsafe_allow_html=True)
        st.caption("Realised daily portfolio P&L overlaid with VaR (expected loss threshold) and CVaR (average tail loss) bands at the configured confidence level — a breach below the band counts as a backtest exception.")
        st.image(str(img_var), use_container_width=True)
        st.caption(
            "LC Fund P&L proxy (duration approximation: ΔP/P ≈ −D_eff × weighted_avg_Δy/100) "
            "with parametric, historical and Monte Carlo VaR/CVaR bands. "
            "Backtested via Kupiec POF and Christoffersen independence tests."
        )
        st.markdown("</div>", unsafe_allow_html=True)
```

Replace with:
```python
# ── VaR Engine ────────────────────────────────────────────────────────────────
elif page == "VaR Engine":
    tab1, tab2, tab3, tab4 = st.tabs([
        "P&L Bands", "Stressed VaR", "Parametric-t Sensitivity", "Risk Decomposition"
    ])

    with tab1:
        img_var = OUT / "var_pnl_bands.png"
        if not img_var.exists():
            st.warning("var_pnl_bands.png not found. Run Module 2 (VaR Engine) in the notebook first.")
        else:
            st.markdown("<div class='section-card'><h3>Portfolio P&L with VaR / CVaR Bands</h3>", unsafe_allow_html=True)
            st.caption("Realised daily portfolio P&L overlaid with VaR (expected loss threshold) and CVaR (average tail loss) bands at the configured confidence level — a breach below the band counts as a backtest exception.")
            st.image(str(img_var), use_container_width=True)
            st.caption(
                "LC Fund P&L proxy (duration approximation: ΔP/P ≈ −D_eff × weighted_avg_Δy/100) "
                "with parametric, historical and Monte Carlo VaR/CVaR bands. "
                "Backtested via Kupiec POF and Christoffersen independence tests."
            )
            st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.info("Stressed VaR — coming in next commit.")

    with tab3:
        st.info("Parametric-t sensitivity — coming in next commit.")

    with tab4:
        st.info("Risk decomposition — coming in next commit.")
```

The `st.info(...)` placeholders are intentional and temporary — each is replaced in Tasks 4–6.

- [ ] **Step 2: Smoke-test parses cleanly**

```powershell
.venv\Scripts\python -c "import ast; ast.parse(open('app.py', encoding='utf-8').read()); print('OK: app.py parses')"
```
Expected: `OK: app.py parses`.

- [ ] **Step 3: Launch Streamlit and confirm 4 tabs render**

```powershell
.venv\Scripts\streamlit run app.py
```

In the browser, navigate to the `VaR Engine` page in the sidebar. Confirm:
- Four tabs are visible at the top: "P&L Bands", "Stressed VaR", "Parametric-t Sensitivity", "Risk Decomposition".
- Tab 1 displays the existing `var_pnl_bands.png` and the two captions.
- Tabs 2–4 each show a single `st.info` placeholder line.

Stop streamlit (Ctrl+C).

- [ ] **Step 4: Commit**

```powershell
git add app.py
git commit -m "refactor(app): convert VaR Engine page to 4-tab layout (Tab 1 only)"
```

---

## Task 4: Implement Tab 2 — Stressed VaR

**Files:**
- Modify: `app.py` — replace the `with tab2:` block from Task 3 with the full Stressed VaR rendering.

- [ ] **Step 1: Replace the tab2 placeholder**

Find:
```python
    with tab2:
        st.info("Stressed VaR — coming in next commit.")
```

Replace with:
```python
    with tab2:
        data = _load_stress_data()
        if data is None:
            st.warning("Stressed VaR artifacts not found. Run Module 2 (VaR Engine) in the notebook first.")
        else:
            pnl = data["pnl"]
            windows = data["windows"]
            summary = data["summary"]
            primary = windows["primary_stress"]
            window_names = list(windows["windows"].keys())
            selected = st.selectbox(
                "Stress window to overlay",
                options=window_names,
                index=window_names.index(primary),
            )
            w = windows["windows"][selected]
            stress_slice = pnl.loc[w["start"]:w["end"]]
            VaR_full = windows["reference"]["hist_full_VaR_95"]
            VaR_param_n = windows["reference"]["parametric_normal_VaR_95"]
            VaR_stress = w["VaR_95"]

            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=pnl, histnorm="probability density", opacity=0.5,
                marker_color="#4682B4", name=f"Full sample (n={len(pnl)})",
                nbinsx=80,
            ))
            fig.add_trace(go.Histogram(
                x=stress_slice, histnorm="probability density", opacity=0.5,
                marker_color="#DC143C", name=f"{selected} (n={len(stress_slice)})",
                nbinsx=20,
            ))
            fig.add_vline(x=-VaR_full, line_dash="dash", line_color="#4682B4",
                          annotation_text=f"Hist VaR 95% (full): {VaR_full:.2%}",
                          annotation_position="top right")
            fig.add_vline(x=-VaR_param_n, line_dash="dot", line_color="black",
                          annotation_text=f"Param normal VaR 95%: {VaR_param_n:.2%}",
                          annotation_position="top right")
            fig.add_vline(x=-VaR_stress, line_dash="dash", line_color="#DC143C",
                          annotation_text=f"Stressed VaR 95% ({selected}): {VaR_stress:.2%}",
                          annotation_position="top right")
            fig.update_layout(
                barmode="overlay",
                title="LC Fund P&L Distribution: Full Sample vs Stress",
                xaxis_title="Daily portfolio P&L",
                yaxis_title="Density",
                legend=dict(x=0.01, y=0.99),
                height=480,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(summary.map(lambda x: f"{x:.4%}"), use_container_width=True)

            ratio_99 = summary.loc["VaR 99%", f"Stressed {selected}"] / summary.loc["VaR 99%", "Historical 3Y"]
            st.caption(
                f"{selected} stress 99% VaR is **{ratio_99:.1f}×** the full-sample historical 99% VaR — "
                "models calibrated on the full sample under-price crisis-conditional tail risk."
            )
```

- [ ] **Step 2: Parse-check**

```powershell
.venv\Scripts\python -c "import ast; ast.parse(open('app.py', encoding='utf-8').read()); print('OK: app.py parses')"
```
Expected: `OK: app.py parses`.

- [ ] **Step 3: Streamlit smoke test**

```powershell
.venv\Scripts\streamlit run app.py
```

Navigate to `VaR Engine → Stressed VaR`. Confirm:
- Selectbox shows `COVID` (default) and `Ukraine_Fed`.
- The Plotly histogram renders with blue (full sample) + crimson (stress) overlays.
- Three vertical lines appear on the loss side of the histogram with labelled VaR percentages.
- A 4×4 table below the chart with rows `VaR 95%, VaR 99%, CVaR 95%, CVaR 99%` and four columns.
- The caption ends with `… X.X× the full-sample historical 99% VaR …`.
- Switching the selectbox to `Ukraine_Fed` updates the crimson histogram and the crimson vertical line.

Stop streamlit (Ctrl+C).

- [ ] **Step 4: Commit**

```powershell
git add app.py
git commit -m "feat(app): VaR Engine Tab 2 — Stressed VaR overlay + table"
```

---

## Task 5: Implement Tab 3 — Parametric-t Sensitivity

**Files:**
- Modify: `app.py` — replace the `with tab3:` block from Task 3.

- [ ] **Step 1: Replace the tab3 placeholder**

Find:
```python
    with tab3:
        st.info("Parametric-t sensitivity — coming in next commit.")
```

Replace with:
```python
    with tab3:
        data = _load_multi_nu()
        if data is None:
            st.warning("Multi-ν parametric-t artifacts not found. Run Module 2 (VaR Engine) in the notebook first.")
        else:
            table = data["table"]
            nu_fit = data["nu_fit"]
            st.markdown(f"**MLE-fitted ν = {nu_fit:.1f}** &nbsp; · &nbsp; comparison grid:")
            st.dataframe(table.map(lambda x: f"{x:.4%}"), use_container_width=True)

            # 99% VaR vs nu line chart. Plot 'inf' at x=30 for visual position;
            # tick-label it as '∞'. Marker grid: [4, 5, 8, 20] for the t rows,
            # plus the normal row at x=30.
            x_numeric = [4, 5, 8, 20, 30]
            x_labels = ["4", "5", "8", "20", "∞"]
            y_99 = table["VaR 99%"].tolist()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_numeric, y=y_99,
                mode="lines+markers",
                marker=dict(size=10, color="#1f77b4"),
                line=dict(color="#1f77b4", width=2),
                hovertemplate="ν = %{text}<br>99% VaR = %{y:.4%}<extra></extra>",
                text=x_labels,
            ))
            fig.update_layout(
                title="99% VaR vs degrees of freedom",
                xaxis=dict(title="ν (Student-t degrees of freedom)",
                           tickvals=x_numeric, ticktext=x_labels),
                yaxis=dict(title="99% VaR", tickformat=".2%"),
                height=420,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "Variance correction `σ · √((ν−2)/ν)` keeps the scaled-t standard deviation "
                "matched to the sample, so rows compare like-for-like. Lower ν → fatter tails "
                "→ larger 99% VaR. The ν → ∞ row reproduces the normal parametric VaR."
            )
```

- [ ] **Step 2: Parse-check**

```powershell
.venv\Scripts\python -c "import ast; ast.parse(open('app.py', encoding='utf-8').read()); print('OK: app.py parses')"
```
Expected: `OK: app.py parses`.

- [ ] **Step 3: Streamlit smoke test**

```powershell
.venv\Scripts\streamlit run app.py
```

Navigate to `VaR Engine → Parametric-t Sensitivity`. Confirm:
- The bold MLE-fitted ν line reads e.g. `**MLE-fitted ν = 5.X**`.
- A 5-row table with `df` index (`nu=4, nu=5, nu=8, nu=20, nu -> inf`) and four percent-formatted columns.
- The line chart is monotonically decreasing in ν (lower ν → larger 99% VaR), with five markers, the rightmost x-tick labeled `∞`.
- Hover on a point shows `ν = X` and `99% VaR = Y.YY%`.

Stop streamlit (Ctrl+C).

- [ ] **Step 4: Commit**

```powershell
git add app.py
git commit -m "feat(app): VaR Engine Tab 3 — multi-ν parametric-t table + chart"
```

---

## Task 6: Implement Tab 4 — Risk Decomposition

**Files:**
- Modify: `app.py` — replace the `with tab4:` block from Task 3.

- [ ] **Step 1: Replace the tab4 placeholder**

Find:
```python
    with tab4:
        st.info("Risk decomposition — coming in next commit.")
```

Replace with:
```python
    with tab4:
        data = _load_decomposition()
        if data is None:
            st.warning("Decomposition artifacts not found. Run Module 2 (VaR Engine) in the notebook first.")
        else:
            scalars = data["scalars"]
            betas = data["betas"]

            col1, col2 = st.columns([2, 1])
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[scalars["pct_systematic"], scalars["pct_idiosyncratic"]],
                    y=["Systematic (PCs)", "Idiosyncratic"],
                    orientation="h",
                    marker_color=["#1f77b4", "#7f7f7f"],
                    text=[f"{scalars['pct_systematic']:.2f}%",
                          f"{scalars['pct_idiosyncratic']:.2f}%"],
                    textposition="auto",
                ))
                fig.update_layout(
                    title="Daily Yield-Change Variance Decomposition",
                    xaxis=dict(title="% of total decomposed variance", range=[0, 100]),
                    height=320,
                    margin=dict(l=10, r=10, t=50, b=40),
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.metric("Decomposition total",   f"{scalars['var_total']:.3e}")
                st.metric("Empirical Var(w'Δy)",   f"{scalars['var_empirical']:.3e}")
                st.metric("Residual-corr gap",     f"{scalars['residual_corr_gap_pct']:+.2f}%")

            st.markdown("**β matrix (country × PC):**")
            st.dataframe(betas.map(lambda x: f"{x:.3f}"), use_container_width=True)

            st.latex(
                r"\mathrm{Var}(\mathbf{w}^\top \Delta y) "
                r"= \mathbf{w}^\top B \Sigma_F B^\top \mathbf{w} "
                r"+ \mathbf{w}^\top D \mathbf{w}"
            )

            st.caption(
                f"Systematic share: **{scalars['pct_systematic']:.2f}%** — the fraction of the "
                "LC fund's daily yield-change variance driven by the global EM rate factors "
                "(PC1 level, PC2 slope, PC3 curvature). A high systematic share means "
                "diversification across the four LC countries is limited."
            )
```

- [ ] **Step 2: Parse-check**

```powershell
.venv\Scripts\python -c "import ast; ast.parse(open('app.py', encoding='utf-8').read()); print('OK: app.py parses')"
```
Expected: `OK: app.py parses`.

- [ ] **Step 3: Streamlit smoke test**

```powershell
.venv\Scripts\streamlit run app.py
```

Navigate to `VaR Engine → Risk Decomposition`. Confirm:
- Two-column layout: bar chart on the left, three metric cards on the right.
- Bar chart shows two horizontal bars with `%` labels summing to 100.
- Three metric cards: `Decomposition total` (scientific notation), `Empirical Var(w'Δy)` (scientific notation), `Residual-corr gap` (signed percent).
- 4×3 β-matrix table with country index and PC1/PC2/PC3 columns.
- LaTeX identity renders correctly.
- Caption ends with `… diversification across the four LC countries is limited.`

Stop streamlit (Ctrl+C).

- [ ] **Step 4: Commit**

```powershell
git add app.py
git commit -m "feat(app): VaR Engine Tab 4 — PC/idiosyncratic risk decomposition"
```

---

## Task 7: Final verification

**Files:** none modified. Verification only.

- [ ] **Step 1: Run the full test suite (regression sanity check)**

```powershell
.venv\Scripts\python -m pytest tests/ -v
```
Expected: all tests pass (37/37 as of the prior plan). The Streamlit changes should not affect any test.

- [ ] **Step 2: Re-execute the notebook from scratch**

```powershell
.venv\Scripts\jupyter nbconvert --to notebook --execute main.ipynb --output main.ipynb
```
Expected: completes end-to-end; prints `Wrote 7 VaR Engine artifacts to data/output/`.

- [ ] **Step 3: Verify all 7 sidecar artifacts present**

```powershell
$expected = @(
  'data/output/var_portfolio_pnl.csv',
  'data/output/var_stressed_summary.csv',
  'data/output/var_stress_windows.json',
  'data/output/var_multi_nu_table.csv',
  'data/output/var_multi_nu_fit.json',
  'data/output/var_decomposition.json',
  'data/output/var_decomposition_betas.csv'
)
foreach ($p in $expected) { if (Test-Path $p) { "OK: $p" } else { "FAIL: $p missing" } }
```
Expected: seven `OK:` lines.

- [ ] **Step 4: Final Streamlit walkthrough**

```powershell
.venv\Scripts\streamlit run app.py
```

Walk through all four `VaR Engine` tabs and confirm each renders without errors and matches the spec sections in `docs/superpowers/specs/2026-05-18-var-engine-page-design.md`. Stop streamlit (Ctrl+C).

No commit needed for this verification task.

---

## Self-review notes

- **Spec coverage:** Each spec section maps to a task — artifacts (Task 1), loaders (Task 2), Tab 1 (Task 3), Tab 2 (Task 4), Tab 3 (Task 5), Tab 4 (Task 6). ✓
- **No placeholders.** The `st.info(...)` strings in Task 3 are intentional temporaries that get fully replaced in Tasks 4–6, not unfilled TBDs.
- **Type consistency.** Loader return-dict keys are used consistently:
  - `_load_stress_data` → `{pnl, windows, summary}` (Task 2) consumed in Task 4.
  - `_load_multi_nu` → `{table, nu_fit}` (Task 2) consumed in Task 5.
  - `_load_decomposition` → `{scalars, betas}` (Task 2) consumed in Task 6.
- **Anchor reuse safety.** Task 1's insertion script anchors on `def kupiec_pof` — the same anchor used by the previous `insert_decomposition_cells.py`. The new cell is inserted BEFORE that anchor; the decomposition cells were also inserted BEFORE that anchor in the previous plan. The notebook execution order ends up: §2.3 plot (last decomposition cell) → §2.4 artifact dump (new) → backtest. That is the intended order.
- **No new dependencies.** `nbformat`, `plotly`, `pandas`, `streamlit`, `numpy`, `json` are already used; nothing to add to `requirements.txt`.
- **Working tree.** `app.py` had pre-existing modifications when the prior plan started; the current branch is `feat/fi-var-improvements`. Tasks here continue on this branch.
