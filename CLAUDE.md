# Cheminformatics in marimo

## Project Overview

Interactive marimo tutorial for cheminformatics practitioners. Single-file app (`main.py`) teaching marimo features using chemistry as a vehicle.

## Tech Stack

- **marimo** — Reactive Python notebook framework
- **RDKit** — Cheminformatics toolkit (Mol objects, descriptors, 2D drawing)
- **pandas** — DataFrame with Mol column
- **Altair** — Declarative visualization (`mo.ui.altair_chart`)
- **Matplotlib** — Plotting (`mo.ui.matplotlib` for interactive selection)
- **scikit-learn** — t-SNE, HDBSCAN (Section G)
- **numpy** — Distance matrix computation (Section G)

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | marimo app (single file, all sections A-G) |
| `pyproject.toml` | Project dependencies |
| `plans/` | Implementation plans |

## marimo Cell Conventions

### `hide_code=True` on ALL cells

Every `@app.cell` MUST use `@app.cell(hide_code=True)`.

**CRITICAL: ruff removes `hide_code=True`** — The ruff post-hook strips the argument, reverting decorators to bare `@app.cell`. After any edit, run:

```bash
sed -i '' 's/^@app\.cell$/@app.cell(hide_code=True)/' main.py
```

### Variable naming

- `_` prefix = cell-private (not returned, not visible to other cells)
- Non-prefixed = returned and available to downstream cells
- Avoid name collisions across cells (e.g., `cs_` prefix for Section G chemical space variables)

### UIElement rule

**Never access `.value` in the same cell that creates a UIElement.** Create in one cell, use `.value` in a downstream cell.

### Cell outputs

- Use `mo.md()` for all text/explanations (no docstrings)
- Use `mo.vstack()` / `mo.hstack()` to compose multiple outputs
- Use `mo.stop()` for guarded computation (show placeholder until interaction)

### Import conflicts

When multiple cells need the same stdlib/package import (e.g., `os`, `RDConfig`), use `_` alias in later cells to avoid `MultipleDefinitionError`:

```python
import os as _os
from rdkit import RDConfig as _RDConfig
```

## Tutorial Structure

| Section | Topics | Key marimo features |
|---------|--------|---------------------|
| A: Setup & Basics | Imports, title, reactivity | `@app.cell`, `mo.md()`, `mo.callout()` |
| B: Data | RDKit demo SDF, DataFrame, table | `mo.ui.table`, `format_mapping`, `mol_to_svg` |
| C: UI Elements | Widget catalog, dropdown+slider combo | `mo.ui.slider`, `mo.ui.dropdown`, `mo.ui.checkbox`, `mo.ui.number` |
| D: Visualization | Altair scatter (interactive), Matplotlib histogram | `mo.ui.altair_chart`, `chart.value` |
| E: Layout | Tabs, accordion | `mo.ui.tabs`, `mo.accordion`, `mo.hstack`, `mo.vstack` |
| F: Advanced | Batch form, guarded computation | `mo.md().batch().form()`, `mo.stop()` |
| G: Applied | Chemical space: t-SNE + HDBSCAN | `mo.ui.matplotlib`, `.value.get_mask()` |

## Content Language

- All text content in **Japanese**
- Matplotlib axis labels in **English** (font compatibility)
- Code comments in English

## Verification

```bash
# Verify all cells execute without errors
uv run marimo export html main.py > /dev/null

# Interactive editing
uv run marimo edit main.py
```

## Dependencies

Managed in two places (keep in sync):
1. PEP 723 inline metadata at top of `main.py`
2. `pyproject.toml` for project-level management

After adding dependencies: `uv lock`
