from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path("data")
OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    (
        DATA_DIR / "extinction_lattice2D.csv",
        "2D lattice extinction time",
        OUTPUT_DIR / "extinction_lattice2D_hist.png",
        {100: "L=10 (N=100)", 196: "L=14 (N=196)", 256: "L=16 (N=256)"},
    ),
    (
        DATA_DIR / "extinction_smallworld.csv",
        "Small-world extinction time",
        OUTPUT_DIR / "extinction_smallworld_hist.png",
        {100: "N=100", 196: "N=196", 256: "N=256"},
    ),
    (
        DATA_DIR / "extinction_scalefree.csv",
        "Scale-free extinction time",
        OUTPUT_DIR / "extinction_scalefree_hist.png",
        {100: "N=100", 196: "N=196", 256: "N=256"},
    ),
]

for csv_path, title, output_path, labels in CONFIGS:
    df = pd.read_csv(csv_path)
    df = df[df["censored"] == 0].copy()

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
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Wrote: {output_path}")
