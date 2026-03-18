#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <cstdint>
#include <thread>
#include <vector>
#include <algorithm>
#include <cstdlib>

#include "graph.hpp"
#include "rng.hpp"
#include "rps_sim.hpp"

// Forward declarations from graph_builders.cpp
Graph build_lattice2D(int L);
Graph build_watts_strogatz(int N, int K, double beta, RNG& rng);
Graph build_barabasi_albert(int N, int m0, int m, RNG& rng);

static inline const char* species_name(int s) {
    if (s == (int)ROCK) return "R";
    if (s == (int)PAPER) return "P";
    if (s == (int)SCISSORS) return "S";
    return "NA"; // includes s == -1 (censored)
}

void print_graph(const Graph& g) {
    std::cout << "Graph: " << g.name << ", N = " << g.N << "\n";

    for (int i = 0; i < g.N; ++i) {
        std::cout << i << " : ";

        for (int v : g.adj[i]) {
            std::cout << v << " ";
        }

        std::cout << "(deg=" << g.adj[i].size() << ")\n";
    }
}

struct RepRecord {
    uint64_t seed = 0;
    SimResult result;
};

static bool should_print_graph() {
    const char* env = std::getenv("RPS_PRINT_GRAPH");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

static unsigned int pick_thread_count(int reps, unsigned int requested_threads) {
    const unsigned int rep_limit = (reps > 0) ? static_cast<unsigned int>(reps) : 1u;

    if (requested_threads > 0) {
        return std::max(1u, std::min(requested_threads, rep_limit));
    }

    const unsigned int hw = std::thread::hardware_concurrency();
    const unsigned int fallback = (hw == 0) ? 1u : hw;
    return std::max(1u, std::min(fallback, rep_limit));
}

int main(int argc, char** argv) {
    try {
        if (argc < 9 || argc > 11) {
            std::cerr
              << "Usage:\n"
              << "  " << argv[0] << " graph_type size_param degree_param beta k reps seed out_csv [max_mcs] [threads]\n"
              << "Examples:\n"
              << "  " << argv[0] << " lattice2D 50 0 0.0 1.0 200 12345 data/out.csv\n"
              << "  " << argv[0] << " smallworld 2500 4 0.10 1.0 200 12345 data/out.csv\n"
              << "  " << argv[0] << " scalefree 2500 2 0.0 1.0 200 12345 data/out.csv\n"
              << "Optional:\n"
              << "  max_mcs <= 0 means uncapped; threads = 0 means auto\n";
            return 1;
        }

        const std::string graph_type = argv[1];
        const int size_param = std::stoi(argv[2]);
        const int degree_param = std::stoi(argv[3]);
        const double beta = std::stod(argv[4]);
        const double k = std::stod(argv[5]);
        const int reps = std::stoi(argv[6]);
        const uint64_t base_seed = (uint64_t)std::stoull(argv[7]);
        const std::string out_csv = argv[8];
        const long long max_mcs = (argc >= 10) ? std::stoll(argv[9]) : 0LL;
        const unsigned int requested_threads = (argc >= 11) ? (unsigned int)std::stoul(argv[10]) : 0u;

        if (reps < 0) {
            throw std::runtime_error("reps must be >= 0.");
        }

        Graph g;
        int L = 0;
        int K = 0;
        double used_beta = 0.0;

        if (graph_type == "lattice2D") {
            L = size_param;
            g = build_lattice2D(L);
            K = 4;
            used_beta = 0.0;

        } else if (graph_type == "smallworld") {
            const int N = size_param;
            K = degree_param;
            used_beta = beta;
            RNG rng_graph(base_seed ^ 0x9e3779b97f4a7c15ULL);
            g = build_watts_strogatz(N, K, used_beta, rng_graph);

        } else if (graph_type == "scalefree") {
            const int N = size_param;
            const int m = degree_param; // interpret degree_param as "m"
            const int m0 = 2 * m;       // simple default
            used_beta = 0.0;            // not used
            K = m;                      // store m in K column for reference
            RNG rng_graph(base_seed ^ 0xD1B54A32D192ED03ULL);
            g = build_barabasi_albert(N, m0, m, rng_graph);

        } else {
            throw std::runtime_error("Unknown graph_type. Use lattice2D, smallworld, or scalefree.");
        }

        if (should_print_graph()) {
            print_graph(g);
        }


        // ---------- CAP SETTINGS ----------
        // Cap in Monte Carlo steps (MCS). If <=0, no cap.
        const long long max_attempts = (max_mcs > 0) ? max_mcs * (long long)g.N : -1;
        // ----------------------------------

        const int outL = (graph_type == "lattice2D") ? L : 0;
        const int N = g.N;
        const unsigned int thread_count = pick_thread_count(reps, requested_threads);

        // Check if file exists to write header once
        bool file_exists = false;
        {
            std::ifstream fin(out_csv);
            file_exists = fin.good();
        }

        std::ofstream fout(out_csv, std::ios::app);
        if (!fout) throw std::runtime_error("Cannot open output file.");

        // Set numeric formatting ONCE (clean + slightly faster)
        fout.setf(std::ios::fixed);
        fout << std::setprecision(6);

        if (!file_exists) {
            fout << "graph,L,N,K,beta,k,rep,seed,max_mcs,censored,Text_attempts,Text_mcs,extinct\n";
        }

        std::cout << "Graph built: " << g.name
                  << " (N=" << N << ")"
                  << ", reps=" << reps
                  << ", max_mcs=" << ((max_mcs > 0) ? std::to_string(max_mcs) : std::string("uncapped"))
                  << ", threads=" << thread_count
                  << "\n";

        std::vector<RepRecord> records(reps);
        auto run_rep = [&](int rep) {
            const uint64_t seed =
                (base_seed ^ 0xA5A5A5A5A5A5A5A5ULL) + (uint64_t)rep * 1315423911ULL;

            RPS_Sim sim(g, seed, k);
            sim.init_random_uniform();

            RepRecord record;
            record.seed = seed;
            record.result = sim.run_until_extinction(max_attempts);
            records[rep] = record;
        };

        std::vector<std::thread> workers;
        workers.reserve(thread_count);
        for (unsigned int t = 0; t < thread_count; ++t) {
            workers.emplace_back([&, t]() {
                for (int rep = static_cast<int>(t); rep < reps; rep += static_cast<int>(thread_count)) {
                    run_rep(rep);
                }
            });
        }
        for (auto& worker : workers) worker.join();

        for (int rep = 0; rep < reps; ++rep) {
            const SimResult& res = records[rep].result;
            const int censored = (res.extinct == -1) ? 1 : 0;
            fout << g.name << ","
                 << outL << ","
                 << N << ","
                 << K << ","
                 << used_beta << ","
                 << k << ","
                 << rep << ","
                 << records[rep].seed << ","
                 << max_mcs << ","
                 << censored << ","
                 << res.Text_attempts << ","
                 << res.Text_mcs << ","
                 << species_name(res.extinct)
                 << "\n";
        }

        std::cout << "Done. Appended " << reps << " runs to: " << out_csv << "\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
