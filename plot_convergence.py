import csv
from collections import defaultdict
from pathlib import Path
import math
import matplotlib.pyplot as plt

INPUT_CSV = Path("convergence_errors.csv")
OUTPUT_PNG_UNCOND = Path("convergence_unconditional_percent_error.png")
OUTPUT_PNG_COND = Path("convergence_conditional_percent_error.png")
OUTPUT_SUMMARY = Path("convergence_slope_summary.txt")


def load_rows(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["reps"] = int(row["reps"])
            row["log10_reps"] = float(row["log10_reps"])
            row["ensemble_size"] = int(row["ensemble_size"])
            row["pct_error_unconditional"] = float(row["pct_error_unconditional"])
            row["std_pct_error_unconditional"] = float(row["std_pct_error_unconditional"])
            row["pct_error_conditional"] = float(row["pct_error_conditional"])
            row["std_pct_error_conditional"] = float(row["std_pct_error_conditional"])
            row["elapsed_seconds"] = float(row["elapsed_seconds"])
            rows.append(row)
    return rows


def group_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["case_key"]].append(row)
    for key in grouped:
        grouped[key].sort(key=lambda r: r["reps"])
    return grouped


def fit_slope(rows, ykey):
    xs, ys = [], []
    for row in rows:
        y = row[ykey]
        if row["reps"] > 0 and y > 0.0:
            xs.append(math.log10(row["reps"]))
            ys.append(math.log10(y))
    if len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0.0:
        return None
    slope = sxy / sxx
    intercept = my - slope * mx
    return slope, intercept, len(xs)


def make_plot(grouped, ykey, skey, ylabel, title, output_path):
    plt.figure(figsize=(8.8, 5.8))
    for case_key, rows in grouped.items():
        xs = [r["reps"] for r in rows if r[ykey] > 0.0]
        ys = [r[ykey] for r in rows if r[ykey] > 0.0]
        stds = [r[skey] for r in rows if r[ykey] > 0.0]
        plt.plot(xs, ys, marker="o", linewidth=1.7, label=case_key)

        lower = [max(1e-12, y - s) for y, s in zip(ys, stds)]
        upper = [max(1e-12, y + s) for y, s in zip(ys, stds)]
        plt.fill_between(xs, lower, upper, alpha=0.18)

        fit = fit_slope(rows, ykey)
        if fit is not None:
            slope, intercept, _ = fit
            xfit = xs
            yfit = [10 ** (intercept + slope * math.log10(x)) for x in xfit]
            plt.plot(xfit, yfit, linestyle="--", linewidth=1.2, label=f"{case_key} fit ({slope:.3f})")

    plt.xscale("log")
    plt.yscale("log")

    # restrict axis window
    plt.xlim(1e3, 1e5)
    plt.ylim(1e-1, 1e1)
    plt.xticks([1e3,1e4,1e5])
    plt.yticks([1e-1,1e0,1e1])

    plt.xlabel("Monte Carlo repetitions")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path.with_suffix(".pdf"))
    plt.savefig(output_path.with_suffix(".png"), dpi=300)
    plt.close()


def write_summary(grouped, path: Path):
    lines = []
    lines.append("Convergence slope summary")
    lines.append("=========================")
    lines.append("Each point is smoothed over the same fixed ensemble of 10 realizations.")
    lines.append("Reference Monte Carlo scaling: percent error ~ reps^(-1/2), so slope ~ -0.5 on a log-log plot.")
    lines.append("")
    for case_key, rows in grouped.items():
        lines.append(f"Case: {case_key}")
        lines.append(f"  ensemble size : {rows[0]['ensemble_size'] if rows else 'NA'}")
        for ykey, label in [("pct_error_unconditional", "unconditional"), ("pct_error_conditional", "conditional")]:
            fit = fit_slope(rows, ykey)
            if fit is None:
                lines.append(f"  {label:14s}: not enough positive-error points")
            else:
                slope, intercept, n_used = fit
                lines.append(f"  {label:14s}: slope = {slope:.6f}, intercept = {intercept:.6f}, points used = {n_used}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    rows = load_rows(INPUT_CSV)
    grouped = group_rows(rows)

    make_plot(
        grouped,
        "pct_error_unconditional",
        "std_pct_error_unconditional",
        "Mean percent error (%)",
        "Smoothed convergence of unconditional extinction-time error",
        OUTPUT_PNG_UNCOND,
    )

    make_plot(
        grouped,
        "pct_error_conditional",
        "std_pct_error_conditional",
        "Mean percent error (%)",
        "Smoothed convergence of conditional extinction-time error",
        OUTPUT_PNG_COND,
    )

    write_summary(grouped, OUTPUT_SUMMARY)

    print(f"Read: {INPUT_CSV.resolve()}")
    print(f"Wrote: {OUTPUT_PNG_UNCOND.resolve()}")
    print(f"Wrote: {OUTPUT_PNG_COND.resolve()}")
    print(f"Wrote: {OUTPUT_SUMMARY.resolve()}")


if __name__ == "__main__":
    main()
