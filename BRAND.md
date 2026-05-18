# EM Fixed Income Intelligence — Graphic Charter

This document is the single source of truth for the visual identity of the
EM Fixed Income Intelligence Platform (Streamlit dashboard, notebook plots,
exported PDFs). All design tokens defined here are mirrored in:

- `app.py` — `:root` CSS custom properties (UI surface)
- `.streamlit/config.toml` — native widget theming
- `src/ui_theme.py` — Plotly chart template (registered globally as `company`)

Whenever a token below changes, update the three mirrors together.

---

## 1. Voice and tone

Institutional, restrained, evidence-led. We are presenting fixed-income
analytics to a portfolio manager — not pitching a consumer app. Visual
decisions should reinforce that posture:

- **Surfaces** stay calm and neutral; the data is the figure.
- **Color** carries semantic weight (status, performance sign) — never decoration.
- **Type** is dense without being cramped; numerics are tabular-aligned.
- **Motion** is functional (state changes) — never animated for ornament.

---

## 2. Color system

### 2.1 Brand palette

| Token              | Hex       | Role                                                |
|--------------------|-----------|-----------------------------------------------------|
| `--c-navy-900`     | `#0d1b2a` | Deepest navy — sidebar base, headers, body type    |
| `--c-navy-800`     | `#152740` | Sidebar gradient stop, dark surfaces                |
| `--c-navy-700`     | `#1b3a5c` | Brand slate — accents, badges, primary borders      |
| `--c-navy-600`     | `#2e6ea8` | Hover/active border on dark surfaces                |
| `--c-azure-300`    | `#7ec8e3` | Brand azure — primary accent, sidebar emphasis      |
| `--c-azure-200`    | `#8ab4d4` | Subdued azure — captions on dark surfaces           |
| `--c-azure-100`    | `#c9d6e3` | Sidebar body type                                   |

### 2.2 Neutral surface

| Token             | Hex       | Role                                              |
|-------------------|-----------|---------------------------------------------------|
| `--c-bg`          | `#f6f8fb` | Main app background (warm-cool neutral)           |
| `--c-surface`     | `#ffffff` | Card surface                                      |
| `--c-surface-alt` | `#f8fafc` | Nested / inset surface, table header              |
| `--c-border`      | `#e2e8f0` | Default hairline                                  |
| `--c-border-soft` | `#eef2f7` | Subtler hairline                                  |
| `--c-text`        | `#0f172a` | Primary text on light                             |
| `--c-text-muted`  | `#64748b` | Secondary text, captions                          |
| `--c-text-subtle` | `#94a3b8` | Tertiary / placeholder                            |

### 2.3 Semantic palette

Reserved exclusively for state and sign. Do **not** use as decoration.

| Token                | Hex       | Use                                       |
|----------------------|-----------|-------------------------------------------|
| `--c-success`        | `#16a34a` | Positive return, green status             |
| `--c-success-soft`   | `#dcfce7` | Success pill background                   |
| `--c-warn`           | `#b45309` | Caution / amber status                    |
| `--c-warn-soft`      | `#fef3c7` | Caution pill / disclaimer background      |
| `--c-warn-border`    | `#fcd34d` | Caution disclaimer border                 |
| `--c-danger`         | `#dc2626` | Negative return, red status               |
| `--c-danger-soft`    | `#fee2e2` | Danger pill background                    |
| `--c-info`           | `#1e40af` | Informational banner text                 |
| `--c-info-soft`      | `#eff6ff` | Informational banner background           |
| `--c-info-border`    | `#bfdbfe` | Informational banner border               |

### 2.4 Chart palette (categorical)

Used for the two-portfolio comparison and any extension to more series.
Defined in `src/ui_theme.py` as `CHART_COLORS`.

| Index | Hex       | Intended role                                |
|-------|-----------|----------------------------------------------|
| 0     | `#1b3a5c` | Portfolio 1 — brand slate                    |
| 1     | `#e67e22` | Portfolio 2 — warm contrast                  |
| 2     | `#16a34a` | 3rd series (benchmark, target)               |
| 3     | `#7ec8e3` | 4th series                                   |
| 4     | `#9333ea` | 5th series                                   |
| 5     | `#64748b` | 6th — fallback                               |

Heatmaps use the diverging `RdBu_r` scale for ρ-style ranges and the
sequential `Blues` scale for unsigned magnitudes.

---

## 3. Typography

### 3.1 Font stack

Headings and body share one stack — no Google Fonts dependency:

```
"Inter", "SF Pro Text", -apple-system, BlinkMacSystemFont, "Segoe UI",
Roboto, "Helvetica Neue", Arial, sans-serif
```

Numerics use `font-variant-numeric: tabular-nums` so columns of figures align.

### 3.2 Type scale

| Token              | Size      | Weight | Line-height | Letter-spacing | Use                       |
|--------------------|-----------|--------|-------------|----------------|---------------------------|
| `--fs-display`     | `1.55rem` | 700    | 1.2         | -0.01em        | Page header                |
| `--fs-h2`          | `1.10rem` | 600    | 1.3         | normal         | Section card title         |
| `--fs-h3`          | `0.95rem` | 600    | 1.35        | normal         | Subsection                 |
| `--fs-body`        | `0.92rem` | 400    | 1.55        | normal         | Body text                  |
| `--fs-stat-value`  | `1.65rem` | 700    | 1.1         | -0.005em       | KPI number on stat card    |
| `--fs-caption`     | `0.80rem` | 400    | 1.5         | normal         | Captions, footnotes        |
| `--fs-label`       | `0.72rem` | 600    | 1.4         | 0.08em (UPPER) | Section labels, stat label |

### 3.3 Headings rule

Never style a heading with a non-token size. If you need a new size,
add a token here first.

---

## 4. Spacing and layout

Spacing follows a **4 px base unit**.

| Token       | Value  | Use                              |
|-------------|--------|----------------------------------|
| `--sp-1`    | `4px`  | Hairline gap                     |
| `--sp-2`    | `8px`  | Tight stack                      |
| `--sp-3`    | `12px` | Stack within a card              |
| `--sp-4`    | `16px` | Card inset, section gap          |
| `--sp-5`    | `20px` | Generous stack                   |
| `--sp-6`    | `24px` | Between sections                 |
| `--sp-7`    | `32px` | Major section break              |
| `--sp-8`    | `48px` | Top-of-page                      |

Radii:

| Token       | Value  | Use                              |
|-------------|--------|----------------------------------|
| `--r-sm`    | `6px`  | Pills, inputs                    |
| `--r-md`    | `10px` | Cards, banners                   |
| `--r-lg`    | `14px` | Page header, hero cards          |
| `--r-pill`  | `999px`| Status pills                     |

---

## 5. Elevation

Three levels — use the smallest one that establishes hierarchy.

| Token       | Shadow                                                        | Use                          |
|-------------|---------------------------------------------------------------|------------------------------|
| `--e-1`     | `0 1px 2px rgba(15,23,42,0.04), 0 1px 3px rgba(15,23,42,0.06)`| Default card                 |
| `--e-2`     | `0 2px 4px rgba(15,23,42,0.05), 0 4px 12px rgba(15,23,42,0.08)`| Hovered card, hero card     |
| `--e-3`     | `0 4px 8px rgba(15,23,42,0.06), 0 12px 28px rgba(15,23,42,0.10)`| Modal, popover               |

---

## 6. Components

### 6.1 Stat card

A KPI display.

- Surface `--c-surface`, radius `--r-md`, elevation `--e-1`.
- Top hairline accent (`3px` solid) carries semantic color
  (`--c-navy-700` default, or success/danger).
- Padding `--sp-4 --sp-5`.
- Label: `--fs-label` style, `--c-text-muted`.
- Value: `--fs-stat-value`, `--c-text`, **tabular numerics**.
- Hover: lift to `--e-2` and translate `-1px` (1 px upward).

### 6.2 Section card

A grouped surface containing a chart or table.

- Surface `--c-surface`, radius `--r-md`, elevation `--e-1`.
- Padding `--sp-6 --sp-7`.
- Header: `--fs-h2`, `--c-text`, bottom hairline `--c-border-soft`
  (padding-bottom `--sp-3`).
- Margin-bottom `--sp-5`.

### 6.3 Hero / page header

- Linear gradient `90deg, --c-navy-900 → --c-navy-700`.
- Padding `--sp-5 --sp-6`, radius `--r-lg`, margin-bottom `--sp-6`.
- Title: `--fs-display`, white.
- Subtitle: `--fs-caption`, `--c-azure-200`.
- Right-side badge: pill, `--c-navy-600` border, `--c-azure-300` text.

### 6.4 Status pills

Inline labels for severity / classification. Background = `*-soft`, text =
the corresponding strong token. Padding `2px 10px`, radius `--r-pill`,
font-size `--fs-caption`, weight 600.

### 6.5 Disclaimer banner (caution)

Replaces a harsh yellow flag.

- Background `--c-warn-soft`, border `1px solid --c-warn-border`.
- Left border `3px solid --c-warn` (rule-stripe).
- Text color `--c-warn` (darker for contrast).
- Radius `--r-md`, padding `--sp-3 --sp-4`.

### 6.6 Info banner

- Background `--c-info-soft`, border `1px solid --c-info-border`.
- Left border `3px solid --c-info`.
- Text `--c-info`, otherwise as caution banner.

### 6.7 Sidebar navigation

- Surface: vertical gradient `--c-navy-900 → --c-navy-800`.
- Items: radio inputs visually replaced with link rows.
- 3 px left border carries selected/hover state in `--c-azure-300`.
- Subtle 90° gradient wash on selected item
  (`rgba(126,200,227,0.16) → 0`).

### 6.8 Plotly chart

All charts inherit the `company` template (registered in `src/ui_theme.py`):

- `plot_bgcolor` / `paper_bgcolor` = `--c-surface`.
- Font family = the brand stack, size 12, color `--c-text`.
- Grid: `--c-border`, zero-line off.
- Categorical palette = `CHART_COLORS`.
- Margins `t=56 r=20 b=44 l=64` unless otherwise stated.
- Legend: horizontal, top, no background.

Individual charts may override `height`, `margin`, or `barmode` —
never colors, fonts, or gridlines.

---

## 7. Accessibility

- All text on a white surface uses `--c-text` (#0f172a) — AAA contrast.
- Caption text uses `--c-text-muted` — AA contrast at 0.80rem.
- Status colors are paired with an icon or text label, never used alone
  to convey state (color-blind safety).
- Focus rings: keep Streamlit defaults (do not suppress with `outline: 0`).

---

## 8. What to avoid

- Inline hex codes in component code — always reference a token.
- Pure-black (`#000`) or pure-white text on color — use `--c-text` / `--c-surface`.
- Drop shadows with `rgba(0,0,0, >0.15)` — always tint with navy
  (`rgba(15,23,42, …)`) for cohesion.
- Decorative use of semantic colors (e.g. green border just because).
- More than 6 distinct categorical series in one chart — fold the rest
  into "Other" or use a separate chart.
- Adding new emoji as UI icons without checking they render on Windows
  (the dashboard ships on Windows; some skin-tone modifiers don't render).

---

## 9. Change log

| Date         | Change                                                       |
|--------------|--------------------------------------------------------------|
| 2026-05-18   | Initial charter extracted from `app.py` v7d00511.            |
