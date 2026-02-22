# Phase 1: Create marimo cheminformatics tutorial skeleton

## Context

Create an interactive marimo tutorial for cheminformatics practitioners migrating to marimo. The focus is on **marimo features** (reactivity, UI, visualization, layout), using chemistry as the vehicle. Single file with PEP 723 inline metadata.

## Target file

- `/Users/nagaet/cheminfo-in-marimo/main.py` — Rewrite as a marimo app

## Reference

- `/Users/nagaet/chemspace-marimo/app.py` — Patterns: `@app.cell`, return tuples, `mo.ui.*`, `rdMolDraw2D` SVG

## Tutorial Structure (25 cells)

### Section A: Setup & Basics (Cells 1-4)

| Cell | Purpose | marimo feature | Returns |
|------|---------|----------------|---------|
| (file-level) | PEP 723 metadata + `marimo.App()` | Inline dependencies | `app` |
| 1 | Shared imports | `@app.cell`, return tuple | `Chem, Descriptors, ...` |
| 2 | Title / intro | `mo.md()` | — |
| 3 | Reactivity explainer | `mo.md()`, `mo.callout()` | — |

### Section B: Data — Inline SMILES & Polars (Cells 4-6)

| Cell | Purpose | marimo feature | Returns |
|------|---------|----------------|---------|
| 4 | Section header | `mo.md()` | — |
| 5 | Sample molecules → Polars DataFrame | Auto-display of DataFrame | `df, mols` |
| 6 | Filtered count (reactivity demo) | DAG re-execution | `drug_like` |

### Section C: UI Elements (Cells 7-12)

| Cell | Purpose | marimo feature | Returns |
|------|---------|----------------|---------|
| 7 | Section header | `mo.md()` | — |
| 8 | MW slider | `mo.ui.slider()` | `mw_slider` |
| 9 | Reactive filtering | `.value`, auto re-run | `filtered_df` |
| 10 | Property dropdown | `mo.ui.dropdown()` | `property_dropdown` |
| 11 | Checkbox + number input | `mo.ui.checkbox()`, `mo.ui.number()`, `mo.hstack()` | `lipinski_check, custom_threshold` |
| 12 | Interactive table with selection | `mo.ui.table(selection="multi")` | `mol_table` |

### Section D: Visualization (Cells 13-17)

| Cell | Purpose | marimo feature | Returns |
|------|---------|----------------|---------|
| 13 | Section header | `mo.md()` | — |
| 14 | Altair scatter plot | Altair auto-display, reactive | `chart` |
| 15 | Matplotlib histogram | `plt` figure display | — |
| 16 | RDKit molecule SVG | `mo.Html()`, `rdMolDraw2D` | `mol_to_svg` |
| 17 | Table selection → structure display | `mo.stop()`, `mo.callout()`, `mol_table.value` | — |

### Section E: Layout (Cells 18-20)

| Cell | Purpose | marimo feature | Returns |
|------|---------|----------------|---------|
| 18 | Section header | `mo.md()` | — |
| 19 | Tabs layout | `mo.ui.tabs()` | — |
| 20 | Accordion with nested layout | `mo.accordion()`, nested `mo.hstack` | — |

### Section F: Advanced Interactivity (Cells 21-23)

| Cell | Purpose | marimo feature | Returns |
|------|---------|----------------|---------|
| 21 | Section header | `mo.md()` | — |
| 22 | Batch + Form | `mo.md().batch()`, `.form()` | `param_form` |
| 23 | Guarded computation | `mo.stop()`, combined filtering + display | — |

### File-level footer

```python
if __name__ == "__main__":
    app.run()
```

## Dependency DAG (simplified)

```
Imports (Cell 1)
 ├─> df, mols (Cell 5) ──> drug_like (Cell 6)
 │       │
 │       ├─> filtered_df (Cell 9) <── mw_slider (Cell 8)
 │       │       │
 │       │       ├─> mol_table (Cell 12)
 │       │       │       └─> structure display (Cell 17) <── mol_to_svg (Cell 16)
 │       │       │
 │       │       ├─> altair chart (Cell 14) <── property_dropdown (Cell 10)
 │       │       ├─> matplotlib hist (Cell 15) <── property_dropdown (Cell 10)
 │       │       └─> tabs layout (Cell 19) <── property_dropdown (Cell 10)
 │       │
 │       ├─> accordion (Cell 20) <── mol_to_svg (Cell 16)
 │       └─> guarded results (Cell 23) <── param_form (Cell 22), mol_to_svg (Cell 16)
 │
 └─> markdown headers (Cells 2,3,4,7,13,18,21) ── depend only on `mo`
```

## Sample data

10 well-known drug molecules as inline SMILES (Aspirin, Caffeine, Ibuprofen, Paracetamol, Naproxen, Diazepam, Metformin, Omeprazole, Atorvastatin, Loratadine). No external files needed.

## Implementation steps

- [ ] Rewrite `main.py` as a marimo app with all 23 cells + file-level code
- [ ] Verify with `uv run marimo edit main.py` that it loads and renders
- [ ] Update `README.md` with tutorial description and usage instructions

## Verification

```bash
# Run the tutorial
uv run marimo edit main.py

# Check: all cells execute without errors
# Check: slider/dropdown/table interactions work
# Check: molecule SVGs render correctly
```

---
- [ ] **DONE** - Phase complete
