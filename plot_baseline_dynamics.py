from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path("data")
OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    (DATA_DIR / "baseline_lattice2D_N100.csv", "2D lattice (N=100)", OUTPUT_DIR / "baseline_lattice2D_N100.png"),
    (DATA_DIR / "baseline_smallworld_N100.csv", "Small-world (N=100, K=4, beta=0.10)", OUTPUT_DIR / "baseline_smallworld_N100.png"),
    (DATA_DIR / "baseline_scalefree_N100.csv", "Scale-free (N=100, m=2)", OUTPUT_DIR / "baseline_scalefree_N100.png"),
]

for csv_path, title, output_path in FILES:
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(8, 5))
    plt.plot(df["mcs"], df["fR"], label="Rock")
    plt.plot(df["mcs"], df["fP"], label="Paper")
    plt.plot(df["mcs"], df["fS"], label="Scissors")
    plt.xlabel("Monte Carlo steps (MCS)")
    plt.ylabel("Species fraction")
    plt.title(title)
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Wrote: {output_path}")
