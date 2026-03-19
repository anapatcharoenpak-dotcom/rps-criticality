from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Iterable


def q(path: Path | str) -> str:
    s = str(path)
    return f'"{s}"' if ' ' in s else s


def run_command(cmd: str, dry_run: bool) -> None:
    print(cmd)
    if not dry_run:
        subprocess.run(cmd, shell=True, check=True)


def compile_rps(project_root: Path, exe_name: str, dry_run: bool) -> None:
    cmd = (
        f"g++ -O2 -std=c++17 "
        f"{q(project_root / 'src' / 'main.cpp')} "
        f"{q(project_root / 'src' / 'graph_builders.cpp')} "
        f"{q(project_root / 'src' / 'rps_sim.cpp')} "
        f"-o {q(project_root / exe_name)}"
    )
    run_command(cmd, dry_run)


def build_run_command(
    exe_path: Path,
    graph: str,
    size_param: int,
    degree_param: int,
    beta: float,
    k: float,
    reps: int,
    seed: int,
    out_csv: Path,
    max_mcs: int,
    threads: int,
) -> str:
    return (
        f"{q(exe_path)} {graph} {size_param} {degree_param} {beta:.3f} {k:.6f} "
        f"{reps} {seed} {q(out_csv)} {max_mcs} {threads}"
    )


def lattice_scan(project_root: Path, exe_name: str, max_mcs: int, threads: int, dry_run: bool) -> None:
    exe_path = project_root / exe_name
    out_csv = project_root / "data" / "phase3_lattice.csv"
    for L in range(4, 14):
        seed = 1000 + L
        cmd = build_run_command(exe_path, "lattice2D", L, 0, 0.0, 1.0, 2000, seed, out_csv, max_mcs, threads)
        run_command(cmd, dry_run)


def smallworld_scan(project_root: Path, exe_name: str, max_mcs: int, threads: int, dry_run: bool) -> None:
    exe_path = project_root / exe_name
    out_csv = project_root / "data" / "phase3_smallworld_K4.csv"
    Ns = [16, 25, 36, 49, 64, 81, 100, 121, 144, 169]
    betas = [0.00, 0.001, 0.003, 0.01, 0.03, 0.10, 0.30, 1.00]
    K = 4
    reps = 100
    graph_realizations = 200
    seed0 = 500000

    for N in Ns:
        for beta in betas:
            beta_code = round(1000 * beta)
            for g in range(graph_realizations):
                seed = seed0 + 10000 * N + 1000 * beta_code + g
                cmd = build_run_command(exe_path, "smallworld", N, K, beta, 1.0, reps, seed, out_csv, max_mcs, threads)
                run_command(cmd, dry_run)


def scalefree_scan(project_root: Path, exe_name: str, max_mcs: int, threads: int, dry_run: bool) -> None:
    exe_path = project_root / exe_name
    out_csv = project_root / "data" / "phase3_scalefree.csv"
    Ns = [16, 25, 36, 49, 64, 81, 100, 121, 144, 169]
    ms = [1, 2, 3, 4, 5]
    reps = 100
    graph_realizations = 200
    seed0 = 900000

    for N in Ns:
        for m in ms:
            for g in range(graph_realizations):
                seed = seed0 + 10000 * N + 1000 * m + g
                cmd = build_run_command(exe_path, "scalefree", N, m, 0.0, 1.0, reps, seed, out_csv, max_mcs, threads)
                run_command(cmd, dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3 scans for the RPS criticality project.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Project root directory")
    parser.add_argument("--exe", default="rps.exe", help="Executable name")
    parser.add_argument("--max-mcs", type=int, default=100000, help="Cap in Monte Carlo steps")
    parser.add_argument("--threads", type=int, default=0, help="Threads passed to rps.exe")
    parser.add_argument("--run", action="store_true", help="Actually execute commands")
    parser.add_argument("--no-compile", action="store_true", help="Skip compilation")
    parser.add_argument("--lattice-only", action="store_true", help="Run only lattice scan")
    parser.add_argument("--smallworld-only", action="store_true", help="Run only small-world scan")
    parser.add_argument("--scalefree-only", action="store_true", help="Run only scale-free scan")
    args = parser.parse_args()

    project_root = args.root
    (project_root / "data").mkdir(parents=True, exist_ok=True)

    dry_run = not args.run

    if not args.no_compile:
        compile_rps(project_root, args.exe, dry_run)

    selected = [args.lattice_only, args.smallworld_only, args.scalefree_only]
    if any(selected):
        if args.lattice_only:
            lattice_scan(project_root, args.exe, args.max_mcs, args.threads, dry_run)
        if args.smallworld_only:
            smallworld_scan(project_root, args.exe, args.max_mcs, args.threads, dry_run)
        if args.scalefree_only:
            scalefree_scan(project_root, args.exe, args.max_mcs, args.threads, dry_run)
    else:
        lattice_scan(project_root, args.exe, args.max_mcs, args.threads, dry_run)
        smallworld_scan(project_root, args.exe, args.max_mcs, args.threads, dry_run)
        scalefree_scan(project_root, args.exe, args.max_mcs, args.threads, dry_run)


if __name__ == "__main__":
    main()
