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
