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
