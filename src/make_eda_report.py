#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_eda_report.py

F1 Pit Stop Prediction 用の詳細EDAスクリプトです。
This script creates a detailed EDA report for the F1 Pit Stop Prediction project.

主な機能:
- train.csv / test.csv / sample_submission.csv / original dataset を自動または指定パスで読み込み
- id を除く全カラムについて、型に応じたグラフを自動作成
- PitNextLap との関係を可視化
- train/test/original の分布差を比較
- Markdown セクション `eda_section.md` を自動生成
- 既存の REPORT.md に `<!-- EDA_START -->` と `<!-- EDA_END -->` の間へ自動挿入可能

使い方の例:
    python src/make_eda_report.py --input-dir data --output-dir reports

    python src/make_eda_report.py \
        --train data/train.csv \
        --test data/test.csv \
        --sample-submission data/sample_submission.csv \
        --original-dir data/original \
        --output-dir reports \
        --report-md REPORT.md

出力例:
    reports/
      eda_section.md
      eda_summary_tables/
      figures/eda/

必要ライブラリ:
    pandas numpy matplotlib

Note:
- GitHub上で画像付きMarkdownとして読めるように、画像リンクは相対パスで出力します。
- target encoding, rolling, tag features などの特徴量そのものはこのファイルでは作らず、EDAとレポート生成に集中します。
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import textwrap
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# サーバーやKaggle環境でも画像保存できるようにする
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ==============================
# Basic utilities
# ==============================

@dataclass
class FigureRecord:
    title: str
    rel_path: str
    description: str = ""


@dataclass
class ColumnReport:
    column: str
    kind: str
    dtype: str
    missing: int
    missing_pct: float
    unique: int
    summary_jp: str
    figures: List[FigureRecord] = field(default_factory=list)


def log(message: str) -> None:
    print(f"[EDA] {message}", flush=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_filename(name: str, max_len: int = 120) -> str:
    """Make a safe filename from a column or chart name."""
    name = str(name)
    name = re.sub(r"[^0-9A-Za-z._\-一-龥ぁ-んァ-ン]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        name = "unnamed"
    return name[:max_len]


def rel_to_output(path: Path, output_dir: Path) -> str:
    return path.relative_to(output_dir).as_posix()


def safe_read_csv(path: Optional[Path], label: str) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    if not path.exists():
        log(f"{label}: file not found: {path}")
        return None
    try:
        df = pd.read_csv(path)
        log(f"Loaded {label}: {path} shape={df.shape}")
        return df
    except Exception as e:
        log(f"Failed to read {label}: {path} ({e})")
        return None


def find_first_file(input_dir: Optional[Path], candidates: Sequence[str]) -> Optional[Path]:
    if input_dir is None or not input_dir.exists():
        return None
    lower_map = {p.name.lower(): p for p in input_dir.glob("*") if p.is_file()}
    for c in candidates:
        p = lower_map.get(c.lower())
        if p is not None:
            return p
    # fallback: recursive search
    for c in candidates:
        matches = list(input_dir.rglob(c))
        if matches:
            return matches[0]
    return None


def maybe_sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    if n <= 0 or len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed)


def is_binary_series(s: pd.Series) -> bool:
    vals = pd.Series(s.dropna().unique())
    if len(vals) != 2:
        return False
    return set(vals.astype(str).tolist()).issubset({"0", "1", "False", "True", "false", "true"}) or len(vals) == 2


def is_probably_ordinal_or_time(col: str) -> bool:
    key = col.lower()
    keywords = [
        "lap", "time", "year", "date", "round", "raceprogress", "progress",
        "stint", "tyrelife", "position", "rank", "order", "stop", "pitstop",
        "degradation", "delta", "gap", "age", "distance", "number", "index"
    ]
    return any(k in key for k in keywords)


def fmt_float(x: object, digits: int = 4) -> str:
    try:
        if pd.isna(x):
            return "NA"
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def pct(x: object, digits: int = 2) -> str:
    try:
        if pd.isna(x):
            return "NA"
        return f"{100 * float(x):.{digits}f}%"
    except Exception:
        return str(x)


# ==============================
# Loading data
# ==============================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a detailed EDA markdown report and figures for F1 Pit Stop Prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=str, default="data", help="Directory containing train.csv/test.csv/etc.")
    parser.add_argument("--train", type=str, default=None, help="Path to train.csv")
    parser.add_argument("--test", type=str, default=None, help="Path to test.csv")
    parser.add_argument("--sample-submission", type=str, default=None, help="Path to sample_submission.csv")
    parser.add_argument("--original-csv", type=str, nargs="*", default=None, help="One or more CSV files from the original dataset")
    parser.add_argument("--original-dir", type=str, default=None, help="Directory containing original dataset CSV files")
    parser.add_argument("--output-dir", type=str, default="reports", help="Output directory for md, csv summaries, and figures")
    parser.add_argument("--report-md", type=str, default=None, help="Optional REPORT.md path to inject EDA section into")
    parser.add_argument("--target", type=str, default="PitNextLap", help="Target column name")
    parser.add_argument("--id-col", type=str, default="id", help="ID column to exclude from per-column EDA")
    parser.add_argument("--top-n", type=int, default=25, help="Top categories to show for categorical charts")
    parser.add_argument("--max-categories", type=int, default=40, help="Max unique categories before treating object-like columns as high cardinality")
    parser.add_argument("--max-plot-rows", type=int, default=300000, help="Sample rows for plotting if dataset is very large; stats still use full data")
    parser.add_argument("--max-corr-cols", type=int, default=35, help="Max number of numeric columns in correlation heatmap")
    parser.add_argument("--dpi", type=int, default=140, help="Figure DPI")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--no-inject", action="store_true", help="Do not inject into REPORT.md even if --report-md is given")
    parser.add_argument("--eda-start-marker", type=str, default="<!-- EDA_START -->", help="Start marker for REPORT.md injection")
    parser.add_argument("--eda-end-marker", type=str, default="<!-- EDA_END -->", help="End marker for REPORT.md injection")
    return parser.parse_args()


def load_original_csvs(args: argparse.Namespace) -> Dict[str, pd.DataFrame]:
    original: Dict[str, pd.DataFrame] = {}
    paths: List[Path] = []

    if args.original_csv:
        paths.extend(Path(p) for p in args.original_csv)

    if args.original_dir:
        odir = Path(args.original_dir)
        if odir.exists():
            paths.extend(sorted(odir.rglob("*.csv")))
        else:
            log(f"original-dir not found: {odir}")

    # 重複パスを除去
    seen = set()
    unique_paths = []
    for p in paths:
        rp = p.resolve() if p.exists() else p
        if str(rp) not in seen:
            seen.add(str(rp))
            unique_paths.append(p)

    for p in unique_paths:
        df = safe_read_csv(p, f"original:{p.name}")
        if df is not None:
            original[p.stem] = df

    return original


def choose_main_original(original_dfs: Dict[str, pd.DataFrame], train: pd.DataFrame, target: str) -> Optional[Tuple[str, pd.DataFrame]]:
    """Choose the original CSV with the largest overlap with train columns."""
    if not original_dfs:
        return None
    train_cols = set(train.columns)
    best_name = None
    best_df = None
    best_score = -1
    for name, df in original_dfs.items():
        overlap = len(set(df.columns) & train_cols)
        score = overlap + (5 if target in df.columns else 0)
        if score > best_score:
            best_score = score
            best_name = name
            best_df = df
    if best_name is None or best_df is None:
        return None
    return best_name, best_df


# ==============================
# Summaries
# ==============================

def build_dataset_overview(datasets: Dict[str, Optional[pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for name, df in datasets.items():
        if df is None:
            rows.append({"dataset": name, "rows": 0, "columns": 0, "missing_cells": 0, "missing_pct": np.nan, "duplicate_rows": 0})
        else:
            total = df.shape[0] * df.shape[1]
            missing = int(df.isna().sum().sum())
            rows.append({
                "dataset": name,
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "missing_cells": missing,
                "missing_pct": missing / total if total else np.nan,
                "duplicate_rows": int(df.duplicated().sum()),
            })
    return pd.DataFrame(rows)


def build_column_summary(df: pd.DataFrame, target: str, id_col: str) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        s = df[c]
        row = {
            "column": c,
            "role": "target" if c == target else ("id" if c.lower() == id_col.lower() else "feature"),
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "missing_pct": s.isna().mean(),
            "unique": int(s.nunique(dropna=True)),
            "sample_values": ", ".join(map(str, s.dropna().unique()[:5])),
        }
        if pd.api.types.is_numeric_dtype(s):
            desc = s.describe()
            row.update({
                "min": desc.get("min", np.nan),
                "q25": desc.get("25%", np.nan),
                "median": desc.get("50%", np.nan),
                "mean": desc.get("mean", np.nan),
                "q75": desc.get("75%", np.nan),
                "max": desc.get("max", np.nan),
                "std": desc.get("std", np.nan),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def build_corr_with_target(df: pd.DataFrame, target: str, id_col: str) -> pd.DataFrame:
    if target not in df.columns:
        return pd.DataFrame()
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c.lower() != id_col.lower()]
    if target not in numeric_cols:
        return pd.DataFrame()
    rows = []
    for c in numeric_cols:
        if c == target:
            continue
        sub = df[[c, target]].dropna()
        if len(sub) < 3 or sub[c].nunique() <= 1:
            corr = np.nan
        else:
            corr = sub[c].corr(sub[target])
        rows.append({"column": c, "corr_with_target": corr, "abs_corr_with_target": abs(corr) if pd.notna(corr) else np.nan})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("abs_corr_with_target", ascending=False)
    return out


def build_category_target_rates(df: pd.DataFrame, target: str, id_col: str, top_n: int) -> pd.DataFrame:
    if target not in df.columns:
        return pd.DataFrame()
    rows = []
    feature_cols = [c for c in df.columns if c != target and c.lower() != id_col.lower()]
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) > 30:
            continue
        tmp = df[[c, target]].dropna()
        if tmp.empty:
            continue
        g = tmp.groupby(c)[target].agg(["count", "mean"]).reset_index()
        g = g.sort_values("count", ascending=False).head(top_n)
        for _, r in g.iterrows():
            rows.append({
                "column": c,
                "category": r[c],
                "count": int(r["count"]),
                "pit_rate": float(r["mean"]),
            })
    return pd.DataFrame(rows)


# ==============================
# Plotting helpers
# ==============================

def savefig(path: Path, dpi: int) -> None:
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_target_distribution(df: pd.DataFrame, target: str, fig_dir: Path, output_dir: Path, dpi: int) -> Optional[FigureRecord]:
    if target not in df.columns:
        return None
    counts = df[target].value_counts(dropna=False).sort_index()
    rates = counts / counts.sum()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar([str(x) for x in counts.index], counts.values)
    axes[0].set_title(f"Target distribution: {target}")
    axes[0].set_xlabel(target)
    axes[0].set_ylabel("Count")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)

    axes[1].pie(rates.values, labels=[str(x) for x in rates.index], autopct="%1.1f%%", startangle=90)
    axes[1].set_title("Target ratio")

    path = fig_dir / "target_distribution.png"
    savefig(path, dpi)
    return FigureRecord(
        title="目的変数の分布 / Target distribution",
        rel_path=rel_to_output(path, output_dir),
        description=f"`{target}` のクラス比率を確認するグラフ。AUC評価ではクラス比率そのものより、正例をどれだけ上位に並べられるかが重要です。",
    )


def plot_missing_values(df: pd.DataFrame, fig_dir: Path, output_dir: Path, dpi: int) -> Optional[FigureRecord]:
    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    if miss.empty:
        return None
    miss = miss.head(30)
    fig, ax = plt.subplots(figsize=(10, max(4, len(miss) * 0.35)))
    ax.barh(miss.index.astype(str), miss.values)
    ax.invert_yaxis()
    ax.set_title("Missing value rate by column")
    ax.set_xlabel("Missing rate")
    path = fig_dir / "missing_values.png"
    savefig(path, dpi)
    return FigureRecord(
        title="欠損率 / Missing values",
        rel_path=rel_to_output(path, output_dir),
        description="欠損があるカラムを欠損率順に表示します。",
    )


def plot_numeric_column(
    df: pd.DataFrame,
    col: str,
    target: str,
    fig_dir: Path,
    output_dir: Path,
    dpi: int,
    seed: int,
    max_plot_rows: int,
) -> Optional[FigureRecord]:
    s = df[col]
    if s.dropna().empty:
        return None

    plot_df = maybe_sample(df[[col] + ([target] if target in df.columns else [])].dropna(subset=[col]), max_plot_rows, seed)
    if plot_df.empty:
        return None

    has_target = target in plot_df.columns and plot_df[target].notna().any()
    unique_n = plot_df[col].nunique(dropna=True)

    fig, axes = plt.subplots(1, 3 if has_target else 2, figsize=(17 if has_target else 11, 4.8))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    # Histogram
    axes[0].hist(plot_df[col].dropna(), bins="auto")
    axes[0].set_title(f"{col}: histogram")
    axes[0].set_xlabel(col)
    axes[0].set_ylabel("Count")

    # Boxplot
    if has_target:
        target_values = sorted(plot_df[target].dropna().unique())
        box_data = [plot_df.loc[plot_df[target] == v, col].dropna().values for v in target_values]
        box_data = [x for x in box_data if len(x) > 0]
        labels = [str(v) for v in target_values if len(plot_df.loc[plot_df[target] == v, col].dropna()) > 0]
        if box_data:
            axes[1].boxplot(box_data, labels=labels, showfliers=False)
            axes[1].set_title(f"{col} by {target}")
            axes[1].set_xlabel(target)
            axes[1].set_ylabel(col)
        else:
            axes[1].axis("off")
    else:
        axes[1].boxplot(plot_df[col].dropna().values, showfliers=False)
        axes[1].set_title(f"{col}: boxplot")
        axes[1].set_ylabel(col)

    # Binned target rate / ordinal line
    if has_target:
        tmp = plot_df[[col, target]].dropna()
        tmp[target] = pd.to_numeric(tmp[target], errors="coerce")
        tmp = tmp.dropna()
        if not tmp.empty and tmp[target].nunique() > 1:
            if unique_n <= 30 or is_probably_ordinal_or_time(col):
                g = tmp.groupby(col)[target].agg(["mean", "count"]).reset_index().sort_values(col)
                if len(g) > 60:
                    # too many values: bin to keep readability
                    tmp["_bin"] = pd.qcut(tmp[col], q=min(20, tmp[col].nunique()), duplicates="drop")
                    g = tmp.groupby("_bin")[target].agg(["mean", "count"]).reset_index()
                    x = np.arange(len(g))
                    labels = [str(v) for v in g["_bin"]]
                else:
                    x = np.arange(len(g))
                    labels = [str(v) for v in g[col]]
            else:
                try:
                    tmp["_bin"] = pd.qcut(tmp[col], q=20, duplicates="drop")
                except Exception:
                    tmp["_bin"] = pd.cut(tmp[col], bins=20, duplicates="drop")
                g = tmp.groupby("_bin")[target].agg(["mean", "count"]).reset_index()
                x = np.arange(len(g))
                labels = [str(v) for v in g["_bin"]]

            ax = axes[2]
            ax.plot(x, g["mean"].values, marker="o")
            ax.set_title(f"{col}: binned {target} rate")
            ax.set_xlabel(col)
            ax.set_ylabel(f"Mean {target}")
            if len(labels) <= 20:
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            else:
                ax.set_xticks([])

            ax2 = ax.twinx()
            ax2.bar(x, g["count"].values, alpha=0.2)
            ax2.set_ylabel("Count")
        else:
            axes[2].axis("off")

    path = fig_dir / "columns" / f"numeric_{sanitize_filename(col)}.png"
    savefig(path, dpi)
    return FigureRecord(
        title=f"{col} の分布と目的変数との関係",
        rel_path=rel_to_output(path, output_dir),
        description=f"数値カラム `{col}` について、分布、`{target}` 別の箱ひげ図、bin別の平均 `{target}` を確認します。",
    )


def plot_categorical_column(
    df: pd.DataFrame,
    col: str,
    target: str,
    fig_dir: Path,
    output_dir: Path,
    dpi: int,
    top_n: int,
    seed: int,
    max_plot_rows: int,
) -> Optional[FigureRecord]:
    if df[col].dropna().empty:
        return None

    cols = [col] + ([target] if target in df.columns else [])
    plot_df = maybe_sample(df[cols].dropna(subset=[col]), max_plot_rows, seed)
    if plot_df.empty:
        return None

    vc = plot_df[col].astype(str).value_counts(dropna=False).head(top_n)
    has_target = target in plot_df.columns and plot_df[target].notna().any()
    n_panels = 3 if has_target and len(vc) <= 12 else (2 if has_target else 1)
    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, max(4.8, min(10, len(vc) * 0.28))))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    # Count bar
    axes[0].barh(vc.index[::-1], vc.values[::-1])
    axes[0].set_title(f"{col}: top {len(vc)} counts")
    axes[0].set_xlabel("Count")

    if has_target:
        tmp = plot_df[[col, target]].copy()
        tmp[col] = tmp[col].astype(str)
        tmp[target] = pd.to_numeric(tmp[target], errors="coerce")
        tmp = tmp.dropna()
        if not tmp.empty:
            top_categories = vc.index.tolist()
            tmp = tmp[tmp[col].isin(top_categories)]
            g = tmp.groupby(col)[target].agg(["count", "mean"]).loc[top_categories].reset_index()
            g = g.sort_values("mean")

            axes[1].barh(g[col].astype(str), g["mean"].values)
            axes[1].set_title(f"{col}: {target} rate")
            axes[1].set_xlabel(f"Mean {target}")
            for i, (_, r) in enumerate(g.iterrows()):
                axes[1].text(r["mean"], i, f" n={int(r['count'])}", va="center", fontsize=8)

        if n_panels >= 3:
            axes[2].pie(vc.values, labels=vc.index, autopct="%1.1f%%", startangle=90)
            axes[2].set_title(f"{col}: share")

    path = fig_dir / "columns" / f"categorical_{sanitize_filename(col)}.png"
    savefig(path, dpi)
    return FigureRecord(
        title=f"{col} の頻度と目的変数との関係",
        rel_path=rel_to_output(path, output_dir),
        description=f"カテゴリカラム `{col}` について、カテゴリ頻度、カテゴリ別 `{target}` 率、必要に応じて円グラフを確認します。",
    )


def plot_correlation_heatmap(df: pd.DataFrame, target: str, id_col: str, fig_dir: Path, output_dir: Path, dpi: int, max_cols: int) -> List[FigureRecord]:
    records: List[FigureRecord] = []
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c.lower() != id_col.lower()]
    if len(numeric_cols) < 2:
        return records

    corr = df[numeric_cols].corr()

    if target in numeric_cols:
        order = corr[target].abs().sort_values(ascending=False).index.tolist()
        selected = order[:max_cols]
    else:
        selected = numeric_cols[:max_cols]
    corr_sel = corr.loc[selected, selected]

    fig, ax = plt.subplots(figsize=(max(8, len(selected) * 0.35), max(7, len(selected) * 0.35)))
    im = ax.imshow(corr_sel.values, aspect="auto", vmin=-1, vmax=1)
    ax.set_title("Numeric feature correlation heatmap")
    ax.set_xticks(np.arange(len(selected)))
    ax.set_yticks(np.arange(len(selected)))
    ax.set_xticklabels(selected, rotation=90, fontsize=7)
    ax.set_yticklabels(selected, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    path = fig_dir / "correlation_heatmap.png"
    savefig(path, dpi)
    records.append(FigureRecord(
        title="数値変数同士の相関 / Numeric correlation heatmap",
        rel_path=rel_to_output(path, output_dir),
        description="数値変数同士の相関を確認します。相関が高すぎる特徴量は、似た情報を持っている可能性があります。",
    ))

    if target in numeric_cols:
        ct = corr[target].drop(labels=[target]).dropna().sort_values(key=lambda s: s.abs(), ascending=False).head(max_cols)
        fig, ax = plt.subplots(figsize=(10, max(4, len(ct) * 0.32)))
        ax.barh(ct.index[::-1], ct.values[::-1])
        ax.set_title(f"Correlation with {target}")
        ax.set_xlabel("Pearson correlation")
        path = fig_dir / "correlation_with_target.png"
        savefig(path, dpi)
        records.append(FigureRecord(
            title=f"目的変数 `{target}` との相関",
            rel_path=rel_to_output(path, output_dir),
            description=f"各数値変数と `{target}` の線形相関を確認します。ただし、非線形な関係は相関係数だけでは見えない点に注意します。",
        ))

    return records


def plot_dataset_comparison(
    datasets: Dict[str, pd.DataFrame],
    col: str,
    target: str,
    fig_dir: Path,
    output_dir: Path,
    dpi: int,
    top_n: int,
    seed: int,
    max_plot_rows: int,
) -> Optional[FigureRecord]:
    available = {name: df for name, df in datasets.items() if df is not None and col in df.columns}
    if len(available) < 2:
        return None

    # Determine type from train if possible, otherwise first available
    base_df = available.get("train", next(iter(available.values())))
    is_num = pd.api.types.is_numeric_dtype(base_df[col])

    if is_num:
        fig, ax = plt.subplots(figsize=(10, 5))
        plotted = 0
        for name, df in available.items():
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if s.empty:
                continue
            s = maybe_sample(pd.DataFrame({col: s}), max_plot_rows, seed)[col]
            ax.hist(s, bins=40, density=True, alpha=0.35, label=name)
            plotted += 1
        if plotted < 2:
            plt.close()
            return None
        ax.set_title(f"Dataset comparison: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.legend()
    else:
        # Use top categories in train/base
        top_values = base_df[col].astype(str).value_counts().head(top_n).index.tolist()
        if not top_values:
            return None
        prop_rows = []
        for name, df in available.items():
            s = df[col].astype(str)
            denom = len(s)
            vc = s.value_counts(normalize=True)
            for v in top_values:
                prop_rows.append({"dataset": name, "category": v, "proportion": float(vc.get(v, 0.0))})
        prop_df = pd.DataFrame(prop_rows)
        x = np.arange(len(top_values))
        width = 0.8 / max(1, len(available))
        fig, ax = plt.subplots(figsize=(max(10, len(top_values) * 0.45), 5.5))
        for i, name in enumerate(available.keys()):
            vals = prop_df[prop_df["dataset"] == name].set_index("category").reindex(top_values)["proportion"].fillna(0).values
            ax.bar(x + i * width, vals, width=width, label=name)
        ax.set_title(f"Dataset comparison: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Proportion")
        ax.set_xticks(x + width * (len(available) - 1) / 2)
        ax.set_xticklabels(top_values, rotation=45, ha="right", fontsize=8)
        ax.legend()

    path = fig_dir / "dataset_comparison" / f"compare_{sanitize_filename(col)}.png"
    savefig(path, dpi)
    return FigureRecord(
        title=f"train/test/original 分布比較: {col}",
        rel_path=rel_to_output(path, output_dir),
        description=f"`{col}` について、train/test/original の分布差を確認します。分布差が大きい場合、CVとPublic LBのズレや汎化性能低下につながる可能性があります。",
    )


# ==============================
# Column interpretation text
# ==============================

def numeric_column_summary_text(df: pd.DataFrame, col: str, target: str) -> str:
    s = pd.to_numeric(df[col], errors="coerce")
    valid = s.dropna()
    if valid.empty:
        return "有効な数値がほとんどないため、解釈は限定的です。"

    base = (
        f"`{col}` は数値変数です。中央値は {fmt_float(valid.median())}、平均は {fmt_float(valid.mean())}、"
        f"範囲は {fmt_float(valid.min())} 〜 {fmt_float(valid.max())} です。"
    )
    if target in df.columns:
        tmp = pd.DataFrame({col: s, target: pd.to_numeric(df[target], errors="coerce")}).dropna()
        if not tmp.empty and tmp[target].nunique() >= 2:
            means = tmp.groupby(target)[col].mean()
            corr = tmp[col].corr(tmp[target]) if tmp[col].nunique() > 1 else np.nan
            detail = f" `{target}` との相関は {fmt_float(corr)} です。"
            if len(means) <= 5:
                mean_text = ", ".join([f"{target}={k}: {fmt_float(v)}" for k, v in means.items()])
                detail += f" クラス別平均は {mean_text} です。"
            return base + detail
    return base


def categorical_column_summary_text(df: pd.DataFrame, col: str, target: str, top_n: int = 5) -> str:
    s = df[col].astype(str)
    vc = s.value_counts(dropna=False)
    if vc.empty:
        return "有効なカテゴリがほとんどないため、解釈は限定的です。"
    top = ", ".join([f"{idx} ({cnt:,})" for idx, cnt in vc.head(top_n).items()])
    base = f"`{col}` はカテゴリ変数です。ユニーク数は {df[col].nunique(dropna=True):,} で、件数が多いカテゴリは {top} です。"
    if target in df.columns:
        tmp = df[[col, target]].dropna().copy()
        tmp[col] = tmp[col].astype(str)
        tmp[target] = pd.to_numeric(tmp[target], errors="coerce")
        tmp = tmp.dropna()
        if not tmp.empty:
            g = tmp.groupby(col)[target].agg(["count", "mean"])
            g = g[g["count"] >= max(10, len(tmp) * 0.001)]
            if not g.empty:
                hi = g.sort_values("mean", ascending=False).head(3)
                lo = g.sort_values("mean", ascending=True).head(3)
                hi_text = ", ".join([f"{idx}: {pct(r['mean'])}" for idx, r in hi.iterrows()])
                lo_text = ", ".join([f"{idx}: {pct(r['mean'])}" for idx, r in lo.iterrows()])
                base += f" 十分な件数があるカテゴリの中では、`{target}`率が高い例は {hi_text}、低い例は {lo_text} です。"
    return base


def infer_kind(df: pd.DataFrame, col: str, max_categories: int) -> str:
    s = df[col]
    nunique = s.nunique(dropna=True)
    if pd.api.types.is_numeric_dtype(s):
        if nunique <= max_categories and (is_probably_ordinal_or_time(col) or nunique <= 15):
            return "numeric_discrete_or_ordinal"
        return "numeric_continuous"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if nunique > max_categories:
        return "categorical_high_cardinality"
    return "categorical"


# ==============================
# Markdown rendering
# ==============================

def md_table(df: pd.DataFrame, max_rows: int = 30, float_digits: int = 4) -> str:
    if df is None or df.empty:
        return "該当データなし。\n"
    tmp = df.head(max_rows).copy()
    for c in tmp.columns:
        if pd.api.types.is_float_dtype(tmp[c]):
            tmp[c] = tmp[c].map(lambda x: fmt_float(x, float_digits))
    return tmp.to_markdown(index=False) + "\n"


def image_md(record: FigureRecord, image_prefix: str = "") -> str:
    rel = record.rel_path
    if image_prefix:
        rel = f"{image_prefix.rstrip('/')}/{rel}"
    alt = record.title.replace("`", "")
    desc = f"\n\n{record.description}" if record.description else ""
    return f"**{record.title}**{desc}\n\n![{alt}]({rel})\n"


def render_markdown(
    dataset_overview: pd.DataFrame,
    column_summary: pd.DataFrame,
    corr_with_target: pd.DataFrame,
    category_rates: pd.DataFrame,
    target_fig: Optional[FigureRecord],
    missing_fig: Optional[FigureRecord],
    corr_figs: List[FigureRecord],
    col_reports: List[ColumnReport],
    compare_figs: List[FigureRecord],
    target: str,
    id_col: str,
    image_prefix: str = "",
) -> str:
    lines: List[str] = []
    lines.append("# EDA Report / 探索的データ分析\n")
    lines.append(
        "このセクションは `make_eda_report.py` によって自動生成されています。"
        "目的は、`id` を除く全カラムについて、分布・目的変数との関係・データセット間の分布差を確認することです。\n"
    )

    lines.append("## 1. Data Overview / データ概要\n")
    lines.append(md_table(dataset_overview, max_rows=20))
    lines.append(
        "`train` はモデル学習用、`test` は提出予測用、`sample_submission` は提出形式確認用です。"
        "`original` が読み込まれている場合は、Playgroundのsynthetic dataとは別系統の元データとして、分布比較や追加学習データ候補の確認に使います。\n"
    )

    lines.append("## 2. Target Overview / 目的変数の確認\n")
    if target_fig is not None:
        lines.append(image_md(target_fig, image_prefix))
    else:
        lines.append(f"`{target}` がtrainに見つからなかったため、目的変数の可視化はスキップしました。\n")
    if missing_fig is not None:
        lines.append(image_md(missing_fig, image_prefix))

    lines.append("## 3. Column Summary / カラム一覧\n")
    show_cols = ["column", "role", "dtype", "missing", "missing_pct", "unique", "sample_values"]
    available_cols = [c for c in show_cols if c in column_summary.columns]
    lines.append(md_table(column_summary[available_cols], max_rows=200))

    lines.append("## 4. Correlation / 相関関係\n")
    if not corr_with_target.empty:
        lines.append("### 4.1 Correlation with Target / 目的変数との相関\n")
        lines.append(md_table(corr_with_target, max_rows=30))
    for r in corr_figs:
        lines.append(image_md(r, image_prefix))

    if not category_rates.empty:
        lines.append("### 4.2 Category-wise Target Rate / カテゴリ別目的変数率\n")
        lines.append("カテゴリ変数や離散値について、カテゴリ別の件数と平均目的変数率を確認します。\n")
        lines.append(md_table(category_rates.sort_values(["column", "pit_rate"], ascending=[True, False]), max_rows=80))

    lines.append("## 5. Per-column EDA / 各カラムの詳細EDA\n")
    lines.append(
        f"以下では `{id_col}` を除く全カラムについて、型に合わせてヒストグラム、箱ひげ図、棒グラフ、円グラフ、線グラフなどを使って確認します。\n"
    )
    for cr in col_reports:
        lines.append(f"### {cr.column}\n")
        lines.append(f"- Type: `{cr.kind}`\n")
        lines.append(f"- dtype: `{cr.dtype}`\n")
        lines.append(f"- Missing: {cr.missing:,} ({pct(cr.missing_pct)})\n")
        lines.append(f"- Unique values: {cr.unique:,}\n")
        lines.append(f"\n{cr.summary_jp}\n")
        for fig in cr.figures:
            lines.append(image_md(fig, image_prefix))

    lines.append("## 6. Train/Test/Original Distribution Comparison / データセット間の分布比較\n")
    if compare_figs:
        lines.append(
            "train/test/original の分布が大きく違うカラムは、Public LBとCVのズレや、外部データ追加時の悪化につながる可能性があります。\n"
        )
        for fig in compare_figs:
            lines.append(image_md(fig, image_prefix))
    else:
        lines.append("比較可能なデータセットが2種類以上ない、または重複カラムが少ないため、分布比較グラフは作成されませんでした。\n")

    lines.append("## 7. Notes for Feature Engineering / 特徴量作成への示唆\n")
    lines.append(
        "EDAで重要なのは、単にグラフを見ることではなく、`なぜ次のラップでピットするのか` という発生メカニズムに結びつけることです。"
        "たとえば、TyreLife、LapNumber、RaceProgress、Stint、Compound、Race、Position系の変数は、ピット戦略の理由と直接つながりやすい候補です。"
        "一方で、相関が低い変数でも、LightGBMやCatBoostのような木系モデルでは非線形な条件分岐として効く場合があります。\n"
    )

    return "\n".join(lines)


def inject_into_report(report_path: Path, eda_markdown: str, start_marker: str, end_marker: str) -> None:
    if not report_path.exists():
        log(f"REPORT.md not found, creating new file: {report_path}")
        report_path.write_text(f"{start_marker}\n{eda_markdown}\n{end_marker}\n", encoding="utf-8")
        return

    text = report_path.read_text(encoding="utf-8")
    if start_marker in text and end_marker in text:
        pattern = re.compile(re.escape(start_marker) + r".*?" + re.escape(end_marker), flags=re.DOTALL)
        replacement = f"{start_marker}\n{eda_markdown}\n{end_marker}"
        new_text = pattern.sub(replacement, text)
    else:
        new_text = text.rstrip() + f"\n\n{start_marker}\n{eda_markdown}\n{end_marker}\n"
    report_path.write_text(new_text, encoding="utf-8")
    log(f"Injected EDA section into {report_path}")


# ==============================
# Main pipeline
# ==============================

def main() -> int:
    args = parse_args()

    input_dir = Path(args.input_dir) if args.input_dir else None
    output_dir = ensure_dir(Path(args.output_dir))
    table_dir = ensure_dir(output_dir / "eda_summary_tables")
    fig_dir = ensure_dir(output_dir / "figures" / "eda")

    train_path = Path(args.train) if args.train else find_first_file(input_dir, ["train.csv"])
    test_path = Path(args.test) if args.test else find_first_file(input_dir, ["test.csv"])
    sub_path = Path(args.sample_submission) if args.sample_submission else find_first_file(input_dir, ["sample_submission.csv", "sample.csv"])

    train = safe_read_csv(train_path, "train")
    if train is None:
        log("train.csv is required. Use --train or put train.csv under --input-dir.")
        return 1
    test = safe_read_csv(test_path, "test")
    sample_submission = safe_read_csv(sub_path, "sample_submission")

    original_dfs = load_original_csvs(args)
    main_original_name = None
    main_original = None
    chosen = choose_main_original(original_dfs, train, args.target)
    if chosen is not None:
        main_original_name, main_original = chosen
        log(f"Main original dataset selected: {main_original_name} shape={main_original.shape}")
    else:
        log("No original dataset loaded. If needed, pass --original-dir or --original-csv.")

    # Save basic summaries
    datasets_overview_input: Dict[str, Optional[pd.DataFrame]] = {
        "train": train,
        "test": test,
        "sample_submission": sample_submission,
    }
    if main_original is not None:
        datasets_overview_input[f"original:{main_original_name}"] = main_original
    dataset_overview = build_dataset_overview(datasets_overview_input)
    dataset_overview.to_csv(table_dir / "dataset_overview.csv", index=False)

    column_summary = build_column_summary(train, args.target, args.id_col)
    column_summary.to_csv(table_dir / "column_summary_train.csv", index=False)

    corr_with_target = build_corr_with_target(train, args.target, args.id_col)
    if not corr_with_target.empty:
        corr_with_target.to_csv(table_dir / "numeric_corr_with_target.csv", index=False)

    category_rates = build_category_target_rates(train, args.target, args.id_col, args.top_n)
    if not category_rates.empty:
        category_rates.to_csv(table_dir / "category_pit_rates.csv", index=False)

    # Global figures
    log("Creating global figures...")
    target_fig = plot_target_distribution(train, args.target, fig_dir, output_dir, args.dpi)
    missing_fig = plot_missing_values(train, fig_dir, output_dir, args.dpi)
    corr_figs = plot_correlation_heatmap(train, args.target, args.id_col, fig_dir, output_dir, args.dpi, args.max_corr_cols)

    # Per-column EDA
    log("Creating per-column EDA figures...")
    feature_cols = [c for c in train.columns if c != args.target and c.lower() != args.id_col.lower()]
    col_reports: List[ColumnReport] = []

    for i, col in enumerate(feature_cols, start=1):
        log(f"Column {i}/{len(feature_cols)}: {col}")
        kind = infer_kind(train, col, args.max_categories)
        s = train[col]
        figures: List[FigureRecord] = []

        if pd.api.types.is_numeric_dtype(s):
            fig = plot_numeric_column(train, col, args.target, fig_dir, output_dir, args.dpi, args.seed, args.max_plot_rows)
            if fig is not None:
                figures.append(fig)
            summary = numeric_column_summary_text(train, col, args.target)
        else:
            fig = plot_categorical_column(train, col, args.target, fig_dir, output_dir, args.dpi, args.top_n, args.seed, args.max_plot_rows)
            if fig is not None:
                figures.append(fig)
            summary = categorical_column_summary_text(train, col, args.target)

        col_reports.append(ColumnReport(
            column=col,
            kind=kind,
            dtype=str(s.dtype),
            missing=int(s.isna().sum()),
            missing_pct=float(s.isna().mean()),
            unique=int(s.nunique(dropna=True)),
            summary_jp=summary,
            figures=figures,
        ))

    # Dataset comparison figures
    log("Creating train/test/original comparison figures...")
    compare_inputs: Dict[str, pd.DataFrame] = {"train": train}
    if test is not None:
        compare_inputs["test"] = test
    if main_original is not None:
        compare_inputs["original"] = main_original

    compare_figs: List[FigureRecord] = []
    for col in feature_cols:
        fig = plot_dataset_comparison(
            compare_inputs,
            col,
            args.target,
            fig_dir,
            output_dir,
            args.dpi,
            args.top_n,
            args.seed,
            args.max_plot_rows,
        )
        if fig is not None:
            compare_figs.append(fig)

    # Save machine-readable column report
    pd.DataFrame([
        {
            "column": cr.column,
            "kind": cr.kind,
            "dtype": cr.dtype,
            "missing": cr.missing,
            "missing_pct": cr.missing_pct,
            "unique": cr.unique,
            "summary_jp": cr.summary_jp,
            "figures": "; ".join(f.rel_path for f in cr.figures),
        }
        for cr in col_reports
    ]).to_csv(table_dir / "column_eda_report.csv", index=False)

    # Render markdown for standalone eda_section.md
    log("Rendering markdown...")
    eda_md = render_markdown(
        dataset_overview=dataset_overview,
        column_summary=column_summary,
        corr_with_target=corr_with_target,
        category_rates=category_rates,
        target_fig=target_fig,
        missing_fig=missing_fig,
        corr_figs=corr_figs,
        col_reports=col_reports,
        compare_figs=compare_figs,
        target=args.target,
        id_col=args.id_col,
        image_prefix="",
    )
    eda_path = output_dir / "eda_section.md"
    eda_path.write_text(eda_md, encoding="utf-8")
    log(f"Saved {eda_path}")

    # Optional injection into REPORT.md
    if args.report_md and not args.no_inject:
        report_path = Path(args.report_md)
        # If REPORT.md is outside output_dir, image links need a prefix from report parent to output_dir
        try:
            image_prefix = os.path.relpath(output_dir, start=report_path.parent).replace("\\", "/")
            if image_prefix == ".":
                image_prefix = ""
        except Exception:
            image_prefix = output_dir.as_posix()

        eda_md_for_report = render_markdown(
            dataset_overview=dataset_overview,
            column_summary=column_summary,
            corr_with_target=corr_with_target,
            category_rates=category_rates,
            target_fig=target_fig,
            missing_fig=missing_fig,
            corr_figs=corr_figs,
            col_reports=col_reports,
            compare_figs=compare_figs,
            target=args.target,
            id_col=args.id_col,
            image_prefix=image_prefix,
        )
        inject_into_report(report_path, eda_md_for_report, args.eda_start_marker, args.eda_end_marker)

    # Summary message
    log("Done.")
    log(f"Markdown: {eda_path}")
    log(f"Figures:  {fig_dir}")
    log(f"Tables:   {table_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
