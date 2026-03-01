#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>

#include "graph.hpp"
#include "rng.hpp"
#include "rps_sim.hpp"

// Forward declarations from graph_builders.cpp
Graph build_lattice2D(int L);
Graph build_watts_strogatz(int N, int K, double beta, RNG& rng);

static inline const char* species_name(int s) {
    if (s == (int)ROCK) return "R";
    if (s == (int)PAPER) return "P";
    if (s == (int)SCISSORS) return "S";
    return "NA";
}

int main(int argc, char** argv) {
    try {
        if (argc < 9) {
            std::cerr
              << "Usage:\n"
              << "  " << argv[0] << " graph_type size_param degree_param beta k reps seed out_csv\n"
              << "Examples:\n"
              << "  " << argv[0] << " lattice2D 50 0 0.0 1.0 200 12345 data/out.csv\n"
              << "  " << argv[0] << " smallworld 2500 4 0.10 1.0 200 12345 data/out.csv\n";
            return 1;
        }

        std::string graph_type = argv[1];
        int size_param = std::stoi(argv[2]);
        int degree_param = std::stoi(argv[3]);
        double beta = std::stod(argv[4]);
        double k = std::stod(argv[5]);
        int reps = std::stoi(argv[6]);
        uint64_t base_seed = (uint64_t)std::stoull(argv[7]);
        std::string out_csv = argv[8];

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
            int N = size_param;
            K = degree_param;
            used_beta = beta;
            RNG rng_graph(base_seed ^ 0x9e3779b97f4a7c15ULL);
            g = build_watts_strogatz(N, K, used_beta, rng_graph);
        } else {
            throw std::runtime_error("Unknown graph_type. Use lattice2D or smallworld.");
        }

        // Check if file exists to write header once
        bool file_exists = false;
        {
            std::ifstream fin(out_csv);
            file_exists = fin.good();
        }

        std::ofstream fout(out_csv, std::ios::app);
        if (!fout) throw std::runtime_error("Cannot open output file.");

        if (!file_exists) {
            fout << "graph,L,N,K,beta,k,rep,seed,Text_attempts,Text_mcs,extinct\n";
        }

        for (int rep = 0; rep < reps; ++rep) {
            uint64_t seed = base_seed + (uint64_t)rep * 1315423911ULL;

            RPS_Sim sim(g, seed, k);
            sim.init_random_equal();
            SimResult res = sim.run_until_extinction();

            fout << g.name << ","
                 << ((graph_type == "lattice2D") ? L : 0) << ","
                 << g.N << ","
                 << K << ","
                 << std::fixed << std::setprecision(6) << used_beta << ","
                 << std::fixed << std::setprecision(6) << k << ","
                 << rep << ","
                 << seed << ","
                 << res.Text_attempts << ","
                 << std::fixed << std::setprecision(6) << res.Text_mcs << ","
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