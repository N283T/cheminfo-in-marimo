# Cheminformatics in marimo

ケモインフォマティクス実務者向けの [marimo](https://marimo.io) インタラクティブチュートリアルです。
化学を題材に marimo の主要機能（リアクティビティ、UI 要素、可視化、レイアウト）を体験できます。

## クイックスタート

```bash
# uv で実行（PEP 723 メタデータから依存関係を自動解決）
uv run marimo edit main.py

# または依存関係を先にインストール
uv sync
uv run marimo edit main.py
```

## チュートリアル構成

| セクション | 内容 | marimo 機能 |
|-----------|------|------------|
| **A: セットアップ** | インポート、タイトル、リアクティビティ解説 | `@app.cell`, `mo.md()`, `mo.callout()` |
| **B: データ** | RDKit デモ SDF (~200 NCI 分子), pandas DataFrame | `mo.ui.table`, `format_mapping`, `mol_to_svg` |
| **C: UI 要素** | ウィジェットカタログ、ドロップダウン+スライダー連携 | `mo.ui.slider`, `mo.ui.dropdown`, `mo.ui.checkbox`, `mo.ui.number` |
| **D: 可視化** | Altair 散布図（インタラクティブ選択）, Matplotlib ヒストグラム | `mo.ui.altair_chart`, `chart.value` |
| **E: レイアウト** | タブ、アコーディオン | `mo.ui.tabs`, `mo.accordion`, `mo.hstack`, `mo.vstack` |
| **F: 高度な機能** | バッチフォーム、ガード付き計算 | `mo.md().batch().form()`, `mo.stop()` |
| **G: 応用** | ケミカルスペース可視化 (t-SNE + HDBSCAN) | `mo.ui.matplotlib`, `.value.get_mask()` |

## サンプルデータ

- **Section B:** RDKit 同梱の NCI 分子セット (`first_200.props.sdf`, ~200 分子)
- **Section G:** NCI SMILES ファイル (`first_5K.smi`, 500 分子を使用)

外部ファイル不要 — すべて RDKit に同梱されています。

## 依存関係

- [marimo](https://marimo.io) — リアクティブ Python ノートブック
- [RDKit](https://www.rdkit.org) — ケモインフォマティクスツールキット
- [pandas](https://pandas.pydata.org) — DataFrame
- [Altair](https://altair-viz.github.io) — 宣言的可視化
- [Matplotlib](https://matplotlib.org) — プロットライブラリ
- [scikit-learn](https://scikit-learn.org) — t-SNE, HDBSCAN
- [NumPy](https://numpy.org) — 数値計算
