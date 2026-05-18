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
