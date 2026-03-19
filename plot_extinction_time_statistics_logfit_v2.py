import math
from pathlib import Path
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path("data")
OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    (
        DATA_DIR / "extinction_lattice2D.csv",
        "2D lattice extinction time",
        OUTPUT_DIR / "extinction_lattice2D_hist.png",
        OUTPUT_DIR / "extinction_lattice2D_logx_hist.png",
        OUTPUT_DIR / "extinction_lattice2D_log10_fit.png",
        OUTPUT_DIR / "extinction_lattice2D_log10_fit_summary.txt",
        {100: "L=10 (N=100)", 196: "L=14 (N=196)", 256: "L=16 (N=256)"},
    ),
    (
        DATA_DIR / "extinction_smallworld.csv",
        "Small-world extinction time",
        OUTPUT_DIR / "extinction_smallworld_hist.png",
        OUTPUT_DIR / "extinction_smallworld_logx_hist.png",
        OUTPUT_DIR / "extinction_smallworld_log10_fit.png",
        OUTPUT_DIR / "extinction_smallworld_log10_fit_summary.txt",
        {100: "N=100", 196: "N=196", 256: "N=256"},
    ),
    (
        DATA_DIR / "extinction_scalefree.csv",
        "Scale-free extinction time",
        OUTPUT_DIR / "extinction_scalefree_hist.png",
        OUTPUT_DIR / "extinction_scalefree_logx_hist.png",
        OUTPUT_DIR / "extinction_scalefree_log10_fit.png",
        OUTPUT_DIR / "extinction_scalefree_log10_fit_summary.txt",
        {100: "N=100", 196: "N=196", 256: "N=256"},
    ),
]

STD_NORMAL = NormalDist(mu=0.0, sigma=1.0)


def normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.zeros_like(x, dtype=float)
    z = (x - mu) / sigma
    return np.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))


def histogram_fit_r2(values: np.ndarray, mu: float, sigma: float, bins: int = 30):
    counts, edges = np.histogram(values, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fitted = normal_pdf(centers, mu, sigma)
    mask = np.isfinite(counts) & np.isfinite(fitted)
    counts = counts[mask]
    fitted = fitted[mask]
    centers = centers[mask]
    if counts.size < 2:
        return float("nan"), centers, counts, fitted
    ss_res = float(np.sum((counts - fitted) ** 2))
    ss_tot = float(np.sum((counts - counts.mean()) ** 2))
    r2 = float("nan") if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    return r2, centers, counts, fitted


def qq_correlation_r(values: np.ndarray, mu: float, sigma: float) -> float:
    n = len(values)
    if n < 3 or sigma <= 0:
        return float("nan")
    sorted_values = np.sort(values)
    probs = (np.arange(1, n + 1) - 0.5) / n
    theoretical = np.array([mu + sigma * STD_NORMAL.inv_cdf(float(p)) for p in probs])
    corr = np.corrcoef(sorted_values, theoretical)[0, 1]
    return float(corr)


def jarque_bera_test(values: np.ndarray):
    n = len(values)
    if n < 4:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean = float(np.mean(values))
    centered = values - mean
    m2 = float(np.mean(centered ** 2))
    if m2 <= 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    m3 = float(np.mean(centered ** 3))
    m4 = float(np.mean(centered ** 4))
    skew = m3 / (m2 ** 1.5)
    kurt = m4 / (m2 ** 2)
    excess_kurt = kurt - 3.0
    jb = (n / 6.0) * (skew ** 2 + 0.25 * excess_kurt ** 2)
    # For df=2, chi-square survival function is exactly exp(-x/2).
    p_value = math.exp(-0.5 * jb)
    return float(skew), float(excess_kurt), float(jb), float(p_value)


for csv_path, title, output_hist, output_logx, output_fit, output_summary, labels in CONFIGS:
    if not csv_path.exists():
        print(f"Skip missing file: {csv_path}")
        continue

    df = pd.read_csv(csv_path)
    df = df[df["censored"] == 0].copy()
    df = df[df["Text_mcs"] > 0].copy()

    if df.empty:
        print(f"Skip empty file after filtering: {csv_path}")
        continue

    plt.figure(figsize=(8, 5))
    for N, label in labels.items():
        subset = df[df["N"] == N]
        if subset.empty:
            continue
        plt.hist(subset["Text_mcs"], bins=30, density=True, alpha=0.45, label=label)
    plt.xlabel("Extinction time (MCS)")
    plt.ylabel("Probability density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_hist, dpi=200)
    plt.close()
    print(f"Wrote: {output_hist}")

    plt.figure(figsize=(8, 5))
    for N, label in labels.items():
        subset = df[df["N"] == N]
        if subset.empty:
            continue
        values = subset["Text_mcs"].to_numpy(dtype=float)
        vmin = values.min()
        vmax = values.max()
        if vmin <= 0 or vmax <= 0 or vmin == vmax:
            continue
        bins = np.logspace(np.log10(vmin), np.log10(vmax), 30)
        plt.hist(values, bins=bins, density=True, alpha=0.45, label=label)
    plt.xscale("log")
    plt.xlabel("Extinction time (MCS)")
    plt.ylabel("Probability density")
    plt.title(title + " (log x-axis)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_logx, dpi=200)
    plt.close()
    print(f"Wrote: {output_logx}")

    summary_lines = []
    plt.figure(figsize=(8, 5))
    for N, label in labels.items():
        subset = df[df["N"] == N]
        if subset.empty:
            continue

        log_values = np.log10(subset["Text_mcs"].to_numpy(dtype=float))
        if len(log_values) < 4:
            continue

        mu = float(log_values.mean())
        sigma = float(log_values.std(ddof=1))

        plt.hist(log_values, bins=30, density=True, alpha=0.35, label=f"{label} data")
        x_min = float(log_values.min())
        x_max = float(log_values.max())
        x_grid = np.linspace(x_min, x_max, 400)
        y_grid = normal_pdf(x_grid, mu, sigma)
        plt.plot(x_grid, y_grid, linewidth=2, label=f"{label} Gaussian fit")

        r2, centers, empirical_pdf, fitted_pdf = histogram_fit_r2(log_values, mu, sigma, bins=30)
        qq_r = qq_correlation_r(log_values, mu, sigma)
        qq_r2 = float("nan") if math.isnan(qq_r) else qq_r ** 2
        skew, excess_kurt, jb, jb_p = jarque_bera_test(log_values)

        summary_lines.append(f"Topology: {title}")
        summary_lines.append(f"Subset: {label}")
        summary_lines.append(f"Samples: {len(log_values)}")
        summary_lines.append(f"mean(log10 T_ext): {mu:.6f}")
        summary_lines.append(f"std(log10 T_ext): {sigma:.6f}")
        summary_lines.append(f"Histogram-fit R^2 (Gaussian on log10 scale): {r2:.6f}")
        summary_lines.append(f"Q-Q correlation r: {qq_r:.6f}")
        summary_lines.append(f"Q-Q correlation R^2: {qq_r2:.6f}")
        summary_lines.append(f"Skewness of log10(T_ext): {skew:.6f}")
        summary_lines.append(f"Excess kurtosis of log10(T_ext): {excess_kurt:.6f}")
        summary_lines.append(f"Jarque-Bera statistic: {jb:.6f}")
        summary_lines.append(f"Jarque-Bera p-value: {jb_p:.6e}")
        summary_lines.append("Interpretation notes:")
        summary_lines.append("- Histogram-fit R^2 close to 1 means the Gaussian curve follows the binned log10-data well.")
        summary_lines.append("- Q-Q correlation R^2 close to 1 means the ordered log10-data closely follows a straight-line Gaussian Q-Q plot.")
        summary_lines.append("- Jarque-Bera p-value is a normality diagnostic on log10(T_ext); small p-values argue against exact Gaussianity.")
        summary_lines.append("")

    plt.xlabel("log10(Extinction time in MCS)")
    plt.ylabel("Probability density")
    plt.title(title + " (histogram of log10 T_ext with Gaussian fit)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_fit, dpi=200)
    plt.close()
    print(f"Wrote: {output_fit}")

    output_summary.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Wrote: {output_summary}")
