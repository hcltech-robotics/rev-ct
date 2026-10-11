"""Render monochrome publication figures from the recorded comparison data."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets"
INK, GREY, PAPER = "#222222", "#aaaaaa", "#faf9f6"
plt.rcParams.update({"font.family": "STIXGeneral", "font.size": 12, "axes.edgecolor": "#bbb9b4",
                     "text.color": INK, "axes.labelcolor": INK, "xtick.color": "#777777",
                     "ytick.color": INK, "figure.facecolor": PAPER, "axes.facecolor": PAPER,
                     "svg.fonttype": "none", "svg.hashsalt": "rev-ct-comparisons", "pdf.fonttype": 42})


def finish(fig, name):
    OUTPUT.mkdir(exist_ok=True)
    for extension in ["png", "svg", "pdf"]:
        metadata = {"Date": None} if extension == "svg" else (
            {"CreationDate": None, "ModDate": None} if extension == "pdf" else None
        )
        fig.savefig(OUTPUT / f"{name}.{extension}", dpi=180, facecolor=PAPER, metadata=metadata)
    plt.close(fig)


def style(ax):
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.tick_params(axis="x", length=3)
    ax.set_axisbelow(True)


def accuracy(comparison):
    rows = comparison["models"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True)
    fig.subplots_adjust(left=.19, right=.92, top=.79, bottom=.14, wspace=.32)
    y = np.arange(len(rows))
    for ax, key, title in zip(axes, ["macro_f1", "macro_auroc"], ["Macro-F1", "Macro-AUROC"], strict=True):
        origin = .3 if key == "macro_f1" else .6
        for j, row in enumerate(rows):
            value = row[key]
            if value is not None:
                ax.barh(j, value - origin, left=origin, height=.36, color=INK if j == 0 else GREY)
                ax.text(value + .022, j, f"{value:.4f}" if j == 0 else f"{value:.3f}", va="center", fontsize=11)
            else:
                ax.text(origin + .008, j, "not reported", va="center", color="#888888", fontsize=11)
        ax.set_xlim(origin, 1.02)
        ax.set_xticks(np.round(np.arange(origin, 1.01, .1), 1))
        ax.set_title(title, loc="left", fontsize=15, pad=12)
        style(ax)
    axes[0].set_yticks(y, [r["name"] for r in rows])
    axes[0].invert_yaxis()
    fig.text(.07, .92, "RevCT is SOTA on CT-RATE.", fontsize=25)
    finish(fig, "classification-comparison")


def timing(benchmark, pillar):
    r = benchmark["results"]
    groups = [
        ("18-finding decisions or scores", [
            ("Rev · fast preprocessing", r["rev_fast_preprocessing"]["summary"]),
            ("Rev · reference preprocessing", r["rev"]["summary"]),
            ("COLIPRI · zero-shot scores", r["colipri_zero_shot"]["summary"]),
            ("NV-Reason-CT · 18 Yes/No prompts", r["nv_reason_ct"]["per_label"]["summary"])]),
        ("Other output contracts", [
            ("NV-Reason-CT · one free-text list", r["nv_reason_ct"]["single_list"]["summary"]),
            ("Pillar-0 · fp32 vision features", pillar["results"]["fp32"]["summary"]),
            ("Pillar-0 · AMP bf16 vision features", pillar["results"]["amp_bf16"]["summary"])])]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.2), gridspec_kw={"height_ratios": [4, 3]})
    fig.subplots_adjust(left=.36, right=.94, top=.75, bottom=.25, hspace=.55)
    for ax, (title, rows) in zip(axes, groups, strict=True):
        for j, (name, summary) in enumerate(rows):
            mean, p95 = summary["mean_s"], summary["p95_s"]
            ax.barh(j, mean, height=.3, color=INK if name.startswith("Rev · fast") else GREY)
            ax.plot(p95, j, "o", ms=3, color=INK)
            ax.text(max(mean, p95) + .5, j, f"{mean:.2f} / {p95:.2f}", va="center", fontsize=10)
        ax.set_yticks(range(len(rows)), [name for name, _ in rows])
        ax.invert_yaxis()
        ax.set_xlim(0, 31)
        ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
        ax.set_title(title, loc="left", fontsize=14, pad=9)
        style(ax)
    axes[1].set_xlabel("Seconds per scan · lower is faster", labelpad=10)
    fig.text(.06, .93, "Inference from a CT volume", fontsize=25)
    fig.text(.06, .875, "64 distinct patients · A100 80 GB · 12 vCPU · batch one", fontsize=12)
    fig.text(.06, .82, "Bar = mean   ·   dot = p95   ·   annotation = mean / p95 seconds", fontsize=11, color="#666666")
    fig.text(.06, .065, "Warm local NIfTI → CPU-visible output; downloads and model loading excluded. Same scans in all rows.\nRev uses fp32; NV and COLIPRI use bf16. Pillar rows produce features, with no trained finding head.\nThe NV list protocol produced 0/64 requested 18-key JSON mappings. See the benchmark report for protocols.", fontsize=10, color="#666666", linespacing=1.5)
    finish(fig, "inference-comparison")


def size(comparison):
    p = comparison["parameters"]
    rev = p["rev_complete_billions_approx"]
    fig, ax = plt.subplots(figsize=(12, 4.9))
    fig.subplots_adjust(left=.32, right=.93, top=.69, bottom=.31)
    names = ["Rev · complete model", "NV-Reason-CT · active CT pathway", "NV-Reason-CT · complete checkpoint"]
    values = [rev, p["nv_active_ct_path_billions_published"], p["nv_checkpoint_billions_published"]]
    for j, value in enumerate(values):
        ax.barh(j, value, height=.3, color=INK if j == 0 else GREY)
        ax.text(value + .07, j, f"{value:.3f}B" if j == 0 else f"{value:.2f}B", va="center", fontsize=12)
    ax.set_yticks(range(3), names)
    ax.invert_yaxis()
    ax.set_xlim(0, 5.35)
    ax.set_xticks(range(6))
    ax.set_xlabel("Billions of parameter values", labelpad=10)
    style(ax)
    fig.text(.06, .90, "A compact CT decision model", fontsize=25)
    fig.text(.06, .82, "589 MB complete fp32 checkpoint", fontsize=13)
    fig.text(.06, .075, "Complete-model counts are approximate. NV sizes are published rounded counts.\nThe checkpoint includes the encoder and the calibrated decision model.", fontsize=10, color="#666666", linespacing=1.6)
    finish(fig, "model-size-comparison")


if __name__ == "__main__":
    comparison = json.loads((ROOT / "benchmarks/model_comparison.json").read_text())
    accuracy(comparison)
    timing(json.loads((ROOT / "reports/benchmark_64.json").read_text()),
           json.loads((ROOT / "reports/pillar_volumes_64.json").read_text()))
    size(comparison)
