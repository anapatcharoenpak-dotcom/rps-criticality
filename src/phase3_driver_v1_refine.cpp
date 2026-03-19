#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct RunConfig {
    std::string project_root = ".";
    std::string exe_name = "rps.exe";
    std::string data_dir = "data";
    double k = 1.0;
    long long max_mcs = 0; // 0 = uncapped
    unsigned int threads = 0;
    bool dry_run = true;
    bool do_compile = true;
    bool do_lattice = false;
    bool do_smallworld = true;
    bool do_scalefree = false;
};

static std::string quote_if_needed(const std::string& s) {
    if (s.find(' ') != std::string::npos) return '"' + s + '"';
    return s;
}

static int run_command(const std::string& cmd, bool dry_run) {
    std::cout << cmd << "\n";
    if (dry_run) return 0;
    return std::system(cmd.c_str());
}

static std::string build_compile_command(const RunConfig& cfg) {
    std::ostringstream oss;
    const std::string root = quote_if_needed(cfg.project_root);
    oss << "g++ -O2 -std=c++17 "
        << root << "\\src\\main.cpp "
        << root << "\\src\\graph_builders.cpp "
        << root << "\\src\\rps_sim.cpp "
        << "-o " << root << "\\" << cfg.exe_name;
    return oss.str();
}

static std::string build_run_command(const RunConfig& cfg,
                                     const std::string& graph,
                                     int size_param,
                                     int degree_param,
                                     double beta,
                                     int reps,
                                     unsigned long long seed,
                                     const std::string& out_csv) {
    std::ostringstream oss;
    oss << quote_if_needed(cfg.project_root + "\\" + cfg.exe_name) << ' '
        << graph << ' '
        << size_param << ' '
        << degree_param << ' '
        << std::fixed << std::setprecision(3) << beta << ' '
        << std::setprecision(6) << cfg.k << ' '
        << reps << ' '
        << seed << ' '
        << quote_if_needed(cfg.project_root + "\\" + out_csv) << ' '
        << cfg.max_mcs << ' '
        << cfg.threads;
    return oss.str();
}

static void ensure_data_dir(const RunConfig& cfg) {
    fs::create_directories(fs::path(cfg.project_root) / cfg.data_dir);
}

static void run_lattice(const RunConfig& cfg) {
    const std::vector<int> Ls{4,5,6,7,8,9,10,11,12,13};
    const int reps = 2000;
    const std::string out_csv = cfg.data_dir + "\\phase3_lattice.csv";
    const unsigned long long seed0 = 1000ULL;

    std::cout << "\n=== Phase 3 lattice2D reference scan ===\n";
    for (int L : Ls) {
        const unsigned long long seed = seed0 + static_cast<unsigned long long>(L);
        const std::string cmd = build_run_command(cfg, "lattice2D", L, 0, 0.0, reps, seed, out_csv);
        const int rc = run_command(cmd, cfg.dry_run);
        if (rc != 0) {
            std::cerr << "Command failed with exit code " << rc << "\n";
            std::exit(rc);
        }
    }
}

static void run_smallworld(const RunConfig& cfg) {
    const std::vector<int> Ns{16,25,36,49,64,81,100,121,144,169,196,225,256,289,324,361,400,441,484,529,576,625,676,729,784,841,900,961};
    const std::vector<double> betas{
        0.00, 0.02, 0.04, 0.06, 0.08,
        0.09, 0.10, 0.11, 0.12, 0.14,
        0.16, 0.18, 0.20
    };
    const int K = 4;
    const int graph_realizations = 200;
    const int reps = 100;
    const std::string out_csv = cfg.data_dir + "\\phase3_smallworld_K4_refine.csv";
    const unsigned long long seed0 = 500000ULL;

    std::cout << "\n=== Phase 3 small-world refined scan (K=4, uncapped extinction time) ===\n";
    for (int N : Ns) {
        for (double beta : betas) {
            const int beta_code = static_cast<int>(beta * 1000.0 + 0.5);
            for (int g = 0; g < graph_realizations; ++g) {
                const unsigned long long seed = seed0
                    + 10000ULL * static_cast<unsigned long long>(N)
                    + 1000ULL * static_cast<unsigned long long>(beta_code)
                    + static_cast<unsigned long long>(g);
                const std::string cmd = build_run_command(cfg, "smallworld", N, K, beta, reps, seed, out_csv);
                const int rc = run_command(cmd, cfg.dry_run);
                if (rc != 0) {
                    std::cerr << "Command failed with exit code " << rc << "\n";
                    std::exit(rc);
                }
            }
        }
    }
}

static void run_scalefree(const RunConfig& cfg) {
    const std::vector<int> Ns{16,25,36,49,64,81,100,121,144,169};
    const std::vector<int> ms{1,2,3,4,5};
    const int graph_realizations = 200;
    const int reps = 100;
    const std::string out_csv = cfg.data_dir + "\\phase3_scalefree.csv";
    const unsigned long long seed0 = 900000ULL;

    std::cout << "\n=== Phase 3 scale-free scan ===\n";
    for (int N : Ns) {
        for (int m : ms) {
            for (int g = 0; g < graph_realizations; ++g) {
                const unsigned long long seed = seed0
                    + 10000ULL * static_cast<unsigned long long>(N)
                    + 1000ULL * static_cast<unsigned long long>(m)
                    + static_cast<unsigned long long>(g);
                const std::string cmd = build_run_command(cfg, "scalefree", N, m, 0.0, reps, seed, out_csv);
                const int rc = run_command(cmd, cfg.dry_run);
                if (rc != 0) {
                    std::cerr << "Command failed with exit code " << rc << "\n";
                    std::exit(rc);
                }
            }
        }
    }
}

static void print_usage(const char* prog) {
    std::cout
        << "Usage: " << prog << " [options]\n\n"
        << "Default behavior of this refined v1 file: compile + small-world only + uncapped extinction time.\n\n"
        << "Options:\n"
        << "  --root PATH          Project root directory (default: .)\n"
        << "  --exe NAME           Executable name (default: rps.exe)\n"
        << "  --run                Actually run commands (default is dry-run)\n"
        << "  --no-compile         Skip compilation\n"
        << "  --lattice-only       Run only lattice scan\n"
        << "  --smallworld-only    Run only small-world scan\n"
        << "  --scalefree-only     Run only scale-free scan\n"
        << "  --threads N          Threads passed to rps.exe (default: 0 = auto)\n"
        << "  --max-mcs M          max_mcs cap (default: 0 = uncapped)\n"
        << "  --help               Show this help\n\n"
        << "Examples:\n"
        << "  phase3_driver_v1_refine.exe\n"
        << "  phase3_driver_v1_refine.exe --run\n"
        << "  phase3_driver_v1_refine.exe --run --smallworld-only --threads 8\n";
}

int main(int argc, char** argv) {
    RunConfig cfg;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--root" && i + 1 < argc) {
            cfg.project_root = argv[++i];
        } else if (arg == "--exe" && i + 1 < argc) {
            cfg.exe_name = argv[++i];
        } else if (arg == "--run") {
            cfg.dry_run = false;
        } else if (arg == "--no-compile") {
            cfg.do_compile = false;
        } else if (arg == "--lattice-only") {
            cfg.do_lattice = true;
            cfg.do_smallworld = false;
            cfg.do_scalefree = false;
        } else if (arg == "--smallworld-only") {
            cfg.do_lattice = false;
            cfg.do_smallworld = true;
            cfg.do_scalefree = false;
        } else if (arg == "--scalefree-only") {
            cfg.do_lattice = false;
            cfg.do_smallworld = false;
            cfg.do_scalefree = true;
        } else if (arg == "--threads" && i + 1 < argc) {
            cfg.threads = static_cast<unsigned int>(std::stoul(argv[++i]));
        } else if (arg == "--max-mcs" && i + 1 < argc) {
            cfg.max_mcs = std::stoll(argv[++i]);
        } else if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown or incomplete option: " << arg << "\n\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    ensure_data_dir(cfg);

    if (cfg.do_compile) {
        std::cout << "\n=== Compile rps.exe ===\n";
        const int rc = run_command(build_compile_command(cfg), cfg.dry_run);
        if (rc != 0) {
            std::cerr << "Compilation failed with exit code " << rc << "\n";
            return rc;
        }
    }

    if (cfg.do_lattice) run_lattice(cfg);
    if (cfg.do_smallworld) run_smallworld(cfg);
    if (cfg.do_scalefree) run_scalefree(cfg);

    std::cout << "\nDone.\n";
    return 0;
}
