#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.20.1",
#     "altair>=6.0.0",
#     "matplotlib>=3.10.8",
#     "numpy",
#     "pandas>=3.0.1",
#     "pyarrow",
#     "rdkit>=2025.9.5",
#     "scikit-learn",
# ]
# ///

import marimo

__generated_with = "0.20.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import altair as alt
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from rdkit import Chem, DataStructs
    from rdkit.Chem import Descriptors, Lipinski, rdFingerprintGenerator
    from rdkit.Chem.Draw import rdMolDraw2D
    from sklearn.cluster import HDBSCAN
    from sklearn.manifold import TSNE

    return (
        Chem,
        DataStructs,
        Descriptors,
        HDBSCAN,
        Lipinski,
        TSNE,
        alt,
        mo,
        np,
        pd,
        plt,
        rdFingerprintGenerator,
        rdMolDraw2D,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # ケモインフォマティクス in marimo

    ケモインフォマティクスの実務者が **marimo** へ移行するための
    インタラクティブチュートリアルです。化学を題材に marimo の
    主要機能を体験しましょう:

    - **リアクティビティ** — セルの依存関係が変わると自動で再実行
    - **UI 要素** — スライダー、ドロップダウン、チェックボックス、テーブル
    - **可視化** — Altair, Matplotlib, RDKit SVG レンダリング
    - **レイアウト** — タブ、アコーディオン、水平・垂直スタック

    > **コードを見るには:** セル右上の `···` → **Show code**、
    > またはセルを選択して **Cmd + H** で表示/非表示を切り替えられます。
    > 気になるセルのコードを確認しながら読み進めてみましょう!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            """
            **marimo のリアクティビティの仕組み**

            marimo はセル間の依存関係を有向非巡回グラフ (DAG) として構築します。
            あるセルの出力が変わると、それに依存する下流のセルがすべて自動で
            再実行されます — 手動での「全セル実行」は不要です。

            このチュートリアルでは、UI ウィジェットを操作して
            依存セルが即座に更新される様子を観察してみてください。
            """
        ),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## セクション B: RDKit デモデータ & テーブル表示

    RDKit に同梱されている NCI 分子セット (`first_200.props.sdf`, 約200分子)
    を読み込み、Mol オブジェクトと分子記述子を pandas DataFrame にまとめます。
    `mo.ui.table()` の `format_mapping` で Mol カラムから構造式を描画します。
    """)
    return


@app.cell(hide_code=True)
def _(Chem, Descriptors, Lipinski, mo, pd):
    import os

    from rdkit import RDConfig

    _sdf_path = os.path.join(RDConfig.RDDataDir, "NCI", "first_200.props.sdf")
    _suppl = Chem.SDMolSupplier(_sdf_path)
    _mols = [_m for _m in _suppl if _m is not None]

    _records = []
    for _mol in _mols:
        _records.append(
            {
                "Mol": _mol,
                "SMILES": Chem.MolToSmiles(_mol),
                "MW": Descriptors.MolWt(_mol),
                "LogP": Descriptors.MolLogP(_mol),
                "HBA": Lipinski.NumHAcceptors(_mol),
                "HBD": Lipinski.NumHDonors(_mol),
                "RotBonds": Lipinski.NumRotatableBonds(_mol),
                "TPSA": Descriptors.TPSA(_mol),
            }
        )

    df = pd.DataFrame(_records)
    mo.vstack(
        [
            mo.md(
                f"RDKit デモ SDF (`first_200.props.sdf`) から "
                f"**{len(df)}** 分子を読み込みました。"
            ),
            df,
        ]
    )
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### `mo.ui.table` と `format_mapping`

    DataFrame をそのまま `mo.ui.table()` に渡すとテーブル表示されますが、
    `Mol` カラムは RDKit のオブジェクトなのでそのままでは意味のある表示になりません。

    そこで `rdMolDraw2D` を使って Mol → SVG に変換する関数 `mol_to_svg` を作り、
    `format_mapping={"Mol": mol_to_svg}` として渡します。
    テーブルの各セルに対して `mol_to_svg(mol)` が呼ばれ、
    2D 構造式がインラインで描画されます。
    数値カラムにも `"{:.2f}".format` を渡して表示桁数を制御できます。

    *下の 2 つのセルのコードを Cmd + H で確認してみましょう*
    """)
    return


@app.cell(hide_code=True)
def _(mo, rdMolDraw2D):
    def mol_to_svg(mol, width: int = 250, height: int = 200) -> mo.Html:
        if mol is None:
            return mo.Html("<em>無効な分子</em>")
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return mo.Html(drawer.GetDrawingText())

    mo.md("`mol_to_svg` ヘルパー関数を定義しました。")
    return (mol_to_svg,)


@app.cell(hide_code=True)
def _(df, mo, mol_to_svg):
    _fmt_2f = "{:.2f}".format
    mol_table = mo.ui.table(
        df,
        selection="multi",
        page_size=5,
        format_mapping={
            "Mol": mol_to_svg,
            "MW": _fmt_2f,
            "LogP": _fmt_2f,
            "TPSA": _fmt_2f,
        },
        label="分子を選択（行をクリック）",
    )
    mol_table
    return (mol_table,)


@app.cell(hide_code=True)
def _(mo, mol_table, mol_to_svg):
    _selected = mol_table.value
    mo.stop(
        _selected.empty,
        mo.md("テーブルの行をクリックすると、ここに選択結果が表示されます。"),
    )
    _svgs = [mol_to_svg(_mol) for _mol in _selected["Mol"]]
    mo.vstack(
        [
            mo.md(f"**選択された {len(_selected)} 分子:**"),
            mo.hstack(_svgs, justify="start", gap=0.5),
        ]
    )
    return


@app.cell(hide_code=True)
def _(df, mo):
    drug_like = df[
        (df["MW"] <= 500) & (df["LogP"] <= 5) & (df["HBA"] <= 10) & (df["HBD"] <= 5)
    ]
    mo.md(
        f"""
        ### Lipinski の Rule of Five フィルタ

        **{len(df)}** 分子中 **{len(drug_like)}** 分子が
        Lipinski の 4 基準をすべて満たしています。
        `df` が変わるとこのセルも自動で再実行されます。
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## セクション C: UI 要素

    marimo には豊富なインタラクティブウィジェットがあり、
    `.value` 属性で常に最新の値を取得できます。
    別のセルでウィジェットを参照すると、リアクティブな依存関係が生まれます。

    まずは主要なウィジェットを紹介し、その後テーブルと組み合わせて使う例を示します。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(
                "### ウィジェットカタログ\n\n"
                "各ウィジェットの `.value` を別セルで参照すると、"
                "値が変わるたびに下流セルが自動で再実行されます。\n\n"
                "*Cmd + H でコードを表示してみましょう*"
            ),
            mo.hstack(
                [
                    mo.ui.slider(start=0, stop=100, step=1, value=50, label="slider"),
                    mo.ui.dropdown(
                        options=["Option A", "Option B", "Option C"],
                        value="Option A",
                        label="dropdown",
                    ),
                ],
                gap=1,
            ),
            mo.hstack(
                [
                    mo.ui.checkbox(label="checkbox", value=False),
                    mo.ui.number(
                        start=0, stop=1000, step=10, value=500, label="number"
                    ),
                ],
                gap=1,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### ウィジェットの組み合わせ

    ドロップダウンでフィルタ対象のプロパティを選び、
    スライダーでしきい値を調整するとテーブルがリアクティブに更新されます。

    **レイアウトヘルパー:**

    - `mo.hstack([...])` — 要素を**横並び**に配置
    - `mo.vstack([...])` — 要素を**縦並び**に配置

    ウィジェットとテーブルを組み合わせて 1 つのセル出力にまとめるのに便利です。

    **ポイント:** UIElement の `.value` は作成したセルでは参照できません。
    別セルで作成 → 別セルで `.value` を使う、が基本パターンです。

    *下のセルのコードを Cmd + H で確認してみましょう*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    property_dropdown = mo.ui.dropdown(
        options=["MW", "LogP", "HBA", "HBD", "RotBonds", "TPSA"],
        value="MW",
        label="フィルタするプロパティ",
    )
    return (property_dropdown,)


@app.cell(hide_code=True)
def _(df, mo, property_dropdown):
    import math

    _prop = property_dropdown.value
    _col = df[_prop]
    _min = math.floor(_col.min())
    _max = math.ceil(_col.max())
    _step = max(1, (_max - _min) // 50)
    filter_slider = mo.ui.slider(
        start=_min,
        stop=_max,
        step=_step,
        value=_max,
        label=f"{_prop} しきい値（以下を表示）",
    )
    return (filter_slider,)


@app.cell(hide_code=True)
def _(df, filter_slider, mo, mol_to_svg, property_dropdown):
    _fmt_2f = "{:.2f}".format
    _prop = property_dropdown.value
    filtered_df = df[df[_prop] <= filter_slider.value]
    mo.vstack(
        [
            mo.hstack([property_dropdown, filter_slider], gap=1),
            mo.md(
                f"**{_prop}** <= **{filter_slider.value}** の分子: "
                f"**{len(filtered_df)}** / {len(df)} 件"
            ),
            mo.ui.table(
                filtered_df,
                page_size=5,
                format_mapping={
                    "Mol": mol_to_svg,
                    "MW": _fmt_2f,
                    "LogP": _fmt_2f,
                    "TPSA": _fmt_2f,
                },
            ),
        ]
    )
    return (filtered_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## セクション D: 可視化

    marimo は Altair チャート、Matplotlib の図、HTML/SVG を自動表示します。
    すべての可視化は上流の変更に反応して更新されます。
    """)
    return


@app.cell(hide_code=True)
def _(alt, filtered_df, mo, property_dropdown):
    _prop = property_dropdown.value
    _alt_chart = (
        alt.Chart(filtered_df.drop(columns=["Mol"]))
        .mark_circle(size=60, opacity=0.7)
        .encode(
            x=alt.X(_prop, title=_prop),
            y=alt.Y("LogP", title="LogP"),
            tooltip=["SMILES", _prop, "LogP"],
        )
        .properties(
            title=f"{_prop} vs LogP",
            width=500,
            height=350,
        )
    )
    chart = mo.ui.altair_chart(_alt_chart)
    mo.vstack(
        [
            mo.md(
                "**`mo.ui.altair_chart`** — 散布図をインタラクティブに操作できます。\n\n"
                "- **クリック** で個別の点を選択\n"
                "- **Shift + クリック** で複数の点を追加選択\n"
                "- **ドラッグ** で矩形範囲を一括選択\n\n"
                "選択すると下のテーブルと構造式が連動して更新されます。"
            ),
            chart,
        ]
    )
    return (chart,)


@app.cell(hide_code=True)
def _(chart, filtered_df, mo, mol_to_svg):
    _selected = chart.value
    mo.stop(
        _selected.empty,
        mo.callout(
            mo.md("散布図上で点をドラッグ選択すると、ここに結果が表示されます。"),
            kind="warn",
        ),
    )
    _fmt_2f = "{:.2f}".format
    _selected_with_mol = filtered_df.loc[
        filtered_df["SMILES"].isin(_selected["SMILES"])
    ]
    mo.vstack(
        [
            mo.md(f"### 選択された {len(_selected_with_mol)} 分子"),
            mo.ui.table(
                _selected_with_mol,
                page_size=5,
                format_mapping={
                    "Mol": mol_to_svg,
                    "MW": _fmt_2f,
                    "LogP": _fmt_2f,
                    "TPSA": _fmt_2f,
                },
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(filtered_df, mo, plt, property_dropdown):
    _prop = property_dropdown.value
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(
        filtered_df[_prop].dropna(),
        bins=10,
        edgecolor="white",
        alpha=0.8,
    )
    ax.set_xlabel(_prop)
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of {_prop}")
    fig.tight_layout()
    mo.vstack(
        [mo.md("**Matplotlib ヒストグラム** — `plt` の図もそのまま表示されます。"), fig]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## セクション E: レイアウト

    `mo.ui.tabs()` と `mo.accordion()` を使うと、
    複雑な出力を整理して表示できます。
    """)
    return


@app.cell(hide_code=True)
def _(chart, filtered_df, mo, mol_to_svg, plt, property_dropdown):
    _prop = property_dropdown.value
    _fmt_2f = "{:.2f}".format

    # ヒストグラムタブ用の図を作成
    _fig, _ax = plt.subplots(figsize=(5, 3.5))
    _ax.hist(filtered_df[_prop].dropna(), bins=10, edgecolor="white", alpha=0.8)
    _ax.set_xlabel(_prop)
    _ax.set_ylabel("Count")
    _ax.set_title(f"Distribution of {_prop}")
    _fig.tight_layout()

    mo.vstack(
        [
            mo.md("**`mo.ui.tabs`** — 複数の出力をタブで切り替え表示"),
            mo.ui.tabs(
                {
                    "散布図": chart,
                    "ヒストグラム": _fig,
                    "データテーブル": mo.ui.table(
                        filtered_df,
                        page_size=5,
                        format_mapping={
                            "Mol": mol_to_svg,
                            "MW": _fmt_2f,
                            "LogP": _fmt_2f,
                            "TPSA": _fmt_2f,
                        },
                    ),
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(Chem, df, mo, mol_to_svg):
    _sections = {}
    for _mol in df["Mol"].iloc[:5]:
        _smi = Chem.MolToSmiles(_mol)
        _sections[_smi[:30]] = mo.hstack(
            [
                mol_to_svg(_mol),
                mo.md(f"**SMILES:** `{_smi}`"),
            ],
            align="center",
            gap=1,
        )

    mo.vstack(
        [
            mo.md("**`mo.accordion`** — 折りたたみで情報を整理"),
            mo.accordion(_sections),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## セクション F: 高度なインタラクティビティ

    `mo.md().batch().form()` を使うと、ユーザーが **Submit** を
    クリックしたときだけ下流の計算を実行するフォームを作成できます。

    *Cmd + H でフォームの作り方を確認してみましょう*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    param_form = (
        mo.md(
            """
            **フィルタパラメータ**

            - 最大 MW: {max_mw}
            - 最大 LogP: {max_logp}
            - 最小 HBA: {min_hba}
            """
        )
        .batch(
            max_mw=mo.ui.slider(start=100, stop=800, step=10, value=500),
            max_logp=mo.ui.slider(start=-2, stop=8, step=0.5, value=5.0),
            min_hba=mo.ui.number(start=0, stop=15, step=1, value=0),
        )
        .form(submit_button_label="フィルタ適用")
    )
    param_form
    return (param_form,)


@app.cell(hide_code=True)
def _(df, mo, mol_to_svg, param_form):
    _fmt_2f = "{:.2f}".format
    mo.stop(
        param_form.value is None,
        mo.callout(
            mo.md("上のパラメータを調整して **フィルタ適用** をクリックしてください。"),
            kind="info",
        ),
    )

    _vals = param_form.value
    _result = df[
        (df["MW"] <= _vals["max_mw"])
        & (df["LogP"] <= _vals["max_logp"])
        & (df["HBA"] >= _vals["min_hba"])
    ]

    mo.vstack(
        [
            mo.md(
                f"### フィルタ結果: {len(_result)} 分子\n\n"
                f"MW <= {_vals['max_mw']}, LogP <= {_vals['max_logp']}, "
                f"HBA >= {_vals['min_hba']}"
            ),
            mo.ui.table(
                _result,
                page_size=5,
                format_mapping={
                    "Mol": mol_to_svg,
                    "MW": _fmt_2f,
                    "LogP": _fmt_2f,
                    "TPSA": _fmt_2f,
                },
            )
            if not _result.empty
            else mo.md("_該当する分子はありません。_"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## セクション G: 応用 — ケミカルスペース可視化

    Morgan フィンガープリント (ECFP4) の Tanimoto 距離行列から
    t-SNE で 2D 射影し、HDBSCAN でクラスタリングした
    ケミカルスペースを可視化します。

    **新しい marimo 機能:**

    - `mo.ui.matplotlib(ax)` — Matplotlib の散布図でボックス/投げ縄選択が可能
    - `.value.get_mask(x, y)` — 選択範囲のブーリアンマスクを取得

    Section D の `mo.ui.altair_chart`（`.value` で DataFrame を返す）と
    対比しながら試してみてください。
    """)
    return


@app.cell(hide_code=True)
def _(
    Chem,
    DataStructs,
    Descriptors,
    Lipinski,
    mo,
    np,
    rdFingerprintGenerator,
):
    import os as _os

    from rdkit import RDConfig as _RDConfig

    _N_MOLS = 500
    _smi_path = _os.path.join(_RDConfig.RDDataDir, "NCI", "first_5K.smi")
    with open(_smi_path) as _f:
        _lines = _f.readlines()

    _mols = []
    for _line in _lines:
        if len(_mols) >= _N_MOLS:
            break
        _parts = _line.strip().split()
        if _parts:
            _mol = Chem.MolFromSmiles(_parts[0])
            if _mol is not None:
                _mols.append(_mol)

    _morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    _fps = [_morgan_gen.GetFingerprint(_m) for _m in _mols]

    _n = len(_mols)
    cs_dist = np.zeros((_n, _n))
    for _i in range(1, _n):
        _sims = DataStructs.BulkTanimotoSimilarity(_fps[_i], _fps[:_i])
        for _j, _s in enumerate(_sims):
            cs_dist[_i, _j] = 1 - _s
            cs_dist[_j, _i] = 1 - _s

    cs_props = []
    for _mol in _mols:
        cs_props.append(
            {
                "SMILES": Chem.MolToSmiles(_mol),
                "MW": round(Descriptors.MolWt(_mol), 2),
                "LogP": round(Descriptors.MolLogP(_mol), 2),
                "HBA": Lipinski.NumHAcceptors(_mol),
                "HBD": Lipinski.NumHDonors(_mol),
                "RotBonds": Lipinski.NumRotatableBonds(_mol),
            }
        )

    mo.md(
        f"NCI `first_5K.smi` から **{_n}** 分子を読み込み、"
        "Morgan FP (ECFP4) と Tanimoto 距離行列を計算しました。"
    )
    return cs_dist, cs_props


@app.cell(hide_code=True)
def _(TSNE, cs_dist, mo):
    _tsne = TSNE(
        n_components=2,
        perplexity=30,
        metric="precomputed",
        random_state=42,
        init="random",
    )
    cs_embedding = _tsne.fit_transform(cs_dist)
    mo.md("t-SNE 2D 射影を計算しました。")
    return (cs_embedding,)


@app.cell(hide_code=True)
def _(mo):
    cs_cluster_slider = mo.ui.slider(
        start=2,
        stop=20,
        step=1,
        value=10,
        label="HDBSCAN min_cluster_size",
    )
    return (cs_cluster_slider,)


@app.cell(hide_code=True)
def _(HDBSCAN, cs_cluster_slider, cs_embedding, mo, np, plt):
    _hdb = HDBSCAN(
        min_cluster_size=cs_cluster_slider.value,
        min_samples=3,
        copy=True,
    )
    cs_labels = _hdb.fit_predict(cs_embedding)
    _n_clusters = len(set(cs_labels.tolist()) - {-1})
    _n_noise = int(np.sum(cs_labels == -1))

    _fig, _ax = plt.subplots(figsize=(10, 6))
    _noise_mask = cs_labels == -1
    _ax.scatter(
        cs_embedding[_noise_mask, 0],
        cs_embedding[_noise_mask, 1],
        c="lightgray",
        s=15,
        alpha=0.4,
        edgecolors="none",
        label="noise",
    )
    _cluster_mask = ~_noise_mask
    _ax.scatter(
        cs_embedding[_cluster_mask, 0],
        cs_embedding[_cluster_mask, 1],
        c=cs_labels[_cluster_mask],
        cmap="tab20",
        s=20,
        alpha=0.8,
        edgecolors="none",
    )
    _ax.set_xlabel("t-SNE 1")
    _ax.set_ylabel("t-SNE 2")
    _ax.set_title(
        f"Chemical Space — {_n_clusters} clusters, "
        f"{_n_noise} noise (n={len(cs_labels)})"
    )
    _fig.tight_layout()

    cs_chart = mo.ui.matplotlib(_ax, debounce=True)
    return cs_chart, cs_labels


@app.cell(hide_code=True)
def _(
    Chem,
    cs_chart,
    cs_cluster_slider,
    cs_embedding,
    cs_labels,
    cs_props,
    mo,
    mol_to_svg,
    np,
):
    _mask = cs_chart.value.get_mask(cs_embedding[:, 0], cs_embedding[:, 1])
    _selected_idx = np.where(_mask)[0]

    mo.stop(
        len(_selected_idx) == 0,
        mo.vstack(
            [
                cs_cluster_slider,
                cs_chart,
                mo.callout(
                    mo.md(
                        "散布図上でボックスまたは投げ縄 (Shift+ドラッグ) で"
                        "分子を選択してください。"
                    ),
                    kind="warn",
                ),
            ]
        ),
    )

    _table_data = [
        {"Cluster": int(cs_labels[_i]), **cs_props[_i]} for _i in _selected_idx
    ]
    _fmt_2f = "{:.2f}".format

    def _smiles_to_svg(smiles):
        _mol = Chem.MolFromSmiles(smiles)
        return mol_to_svg(_mol) if _mol else mo.Html("<em>無効</em>")

    mo.vstack(
        [
            cs_cluster_slider,
            cs_chart,
            mo.md(f"### 選択: {len(_selected_idx)} 分子"),
            mo.ui.table(
                _table_data,
                page_size=5,
                format_mapping={
                    "SMILES": _smiles_to_svg,
                    "MW": _fmt_2f,
                    "LogP": _fmt_2f,
                },
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
