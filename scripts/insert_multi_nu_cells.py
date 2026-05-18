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
