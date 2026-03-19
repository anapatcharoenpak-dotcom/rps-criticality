from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["graph", "N", "K", "beta"], as_index=False)
        .agg(
            mean_Text_mcs=("Text_mcs", "mean"),
            std_Text_mcs=("Text_mcs", "std"),
            n_runs=("Text_mcs", "size"),
            censored_fraction=("censored", "mean"),
        )
        .sort_values(["graph", "K", "beta", "N"])
    )
    return grouped


def plot_lattice(df: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["N"], df["mean_Text_mcs"], marker="o")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("Mean extinction time (MCS)")
    ax.set_title("Phase 3 lattice2D reference")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "phase3_lattice_loglog.png", dpi=200)
    plt.close(fig)


def plot_smallworld(df: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for beta, sub in df.groupby("beta"):
        sub = sub.sort_values("N")
        ax.plot(sub["N"], sub["mean_Text_mcs"], marker="o", label=f"beta={beta:g}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("Mean extinction time (MCS)")
    ax.set_title("Phase 3 small-world crossover scan (K=4)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "phase3_smallworld_loglog.png", dpi=200)
    plt.close(fig)


def plot_scalefree(df: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for m, sub in df.groupby("K"):
        sub = sub.sort_values("N")
        ax.plot(sub["N"], sub["mean_Text_mcs"], marker="o", label=f"m={int(m)}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("Mean extinction time (MCS)")
    ax.set_title("Phase 3 scale-free crossover scan")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "phase3_scalefree_loglog.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize and plot Phase 3 CSV outputs.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing Phase 3 CSV files")
    args = parser.parse_args()

    data_dir = args.data_dir
    outdir = data_dir

    lattice_file = data_dir / "phase3_lattice.csv"
    sw_file = data_dir / "phase3_smallworld_K4.csv"
    sf_file = data_dir / "phase3_scalefree.csv"

    summary_lines = []

    if lattice_file.exists():
        df = pd.read_csv(lattice_file)
        s = summarize(df)
        s.to_csv(outdir / "phase3_lattice_summary.csv", index=False)
        plot_lattice(s, outdir)
        summary_lines.append(f"Lattice rows: {len(df)} -> {outdir / 'phase3_lattice_summary.csv'}")

    if sw_file.exists():
        df = pd.read_csv(sw_file)
        s = summarize(df)
        s.to_csv(outdir / "phase3_smallworld_summary.csv", index=False)
        plot_smallworld(s, outdir)
        summary_lines.append(f"Small-world rows: {len(df)} -> {outdir / 'phase3_smallworld_summary.csv'}")

    if sf_file.exists():
        df = pd.read_csv(sf_file)
        s = summarize(df)
        s.to_csv(outdir / "phase3_scalefree_summary.csv", index=False)
        plot_scalefree(s, outdir)
        summary_lines.append(f"Scale-free rows: {len(df)} -> {outdir / 'phase3_scalefree_summary.csv'}")

    report = outdir / "phase3_plot_report.txt"
    report.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
