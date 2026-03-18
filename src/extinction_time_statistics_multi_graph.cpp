#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "graph.hpp"
#include "rng.hpp"
#include "rps_sim.hpp"

// Forward declarations from graph_builders.cpp
Graph build_watts_strogatz(int N, int K, double beta, RNG& rng);
Graph build_barabasi_albert(int N, int m0, int m, RNG& rng);

namespace {

Graph build_lattice_rectangular(int width, int height) {
    const int N = width * height;
    Graph g(N);
    g.name = "lattice2D";

    auto idx = [width, height](int x, int y) {
        x = (x % width + width) % width;
        y = (y % height + height) % height;
        return y * width + x;
    };

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int u = idx(x, y);
            auto& nb = g.adj[u];
            nb.reserve(4);
            nb.push_back(idx(x + 1, y));
            nb.push_back(idx(x - 1, y));
            nb.push_back(idx(x, y + 1));
            nb.push_back(idx(x, y - 1));
        }
    }

    g.finalize_simple();
    return g;
}

std::pair<int, int> choose_lattice_shape(int N) {
    if (N <= 0) throw std::runtime_error("lattice2D requires N > 0.");

    int width = 1;
    for (int d = 1; d * d <= N; ++d) {
        if (N % d == 0) width = d;
    }
    const int height = N / width;
    return {width, height};
}

const char* species_name(int s) {
    if (s == static_cast<int>(ROCK)) return "R";
    if (s == static_cast<int>(PAPER)) return "P";
    if (s == static_cast<int>(SCISSORS)) return "S";
    return "NA";
}

Graph build_graph(const std::string& graph_type,
                  int size_param,
                  int degree_param,
                  double beta,
                  uint64_t graph_seed) {
    if (graph_type == "lattice2D") {
        const auto [width, height] = choose_lattice_shape(size_param);
        return build_lattice_rectangular(width, height);
    }

    if (graph_type == "smallworld") {
        RNG graph_rng(graph_seed ^ 0x9e3779b97f4a7c15ULL);
        return build_watts_strogatz(size_param, degree_param, beta, graph_rng);
    }

    if (graph_type == "scalefree") {
        const int m = degree_param;
        const int m0 = 2 * m;
        RNG graph_rng(graph_seed ^ 0xD1B54A32D192ED03ULL);
        return build_barabasi_albert(size_param, m0, m, graph_rng);
    }

    throw std::runtime_error("Unknown graph_type. Use lattice2D, smallworld, or scalefree.");
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 12) {
            std::cerr
                << "Usage:\n"
                << "  " << argv[0]
                << " graph_type size_param degree_param beta k reps_per_graph num_graphs base_seed max_mcs out_csv append\n\n"
                << "Notes:\n"
                << "  - lattice2D expects size_param = N (must factor into width x height; perfect squares give square lattices).\n"
                << "  - smallworld and scalefree are averaged over num_graphs independent graph realizations.\n"
                << "  - lattice2D is deterministic, so num_graphs is ignored in practice and treated as 1.\n\n"
                << "Examples:\n"
                << "  " << argv[0] << " lattice2D 100 0 0.0 1.0 1000 1 12345 0 data/extinction_lattice.csv 0\n"
                << "  " << argv[0] << " smallworld 100 4 0.10 1.0 200 20 12345 0 data/extinction_smallworld.csv 0\n"
                << "  " << argv[0] << " scalefree 100 2 0.0 1.0 200 20 12345 0 data/extinction_scalefree.csv 0\n";
            return 1;
        }

        const std::string graph_type = argv[1];
        const int size_param = std::stoi(argv[2]);
        const int degree_param = std::stoi(argv[3]);
        const double beta = std::stod(argv[4]);
        const double k = std::stod(argv[5]);
        const int reps_per_graph = std::stoi(argv[6]);
        const int num_graphs_input = std::stoi(argv[7]);
        const uint64_t base_seed = static_cast<uint64_t>(std::stoull(argv[8]));
        const long long max_mcs = std::stoll(argv[9]);
        const std::string out_csv = argv[10];
        const bool append_mode = (std::stoi(argv[11]) != 0);

        if (reps_per_graph <= 0) throw std::runtime_error("reps_per_graph must be > 0.");
        if (num_graphs_input <= 0) throw std::runtime_error("num_graphs must be > 0.");

        const int num_graphs = (graph_type == "lattice2D") ? 1 : num_graphs_input;

        std::ios::openmode mode = std::ios::out;
        if (append_mode) mode |= std::ios::app;
        std::ofstream fout(out_csv, mode);
        if (!fout) throw std::runtime_error("Cannot open output CSV: " + out_csv);

        if (!append_mode) {
            fout << "graph,N,size_param,degree_param,beta,k,graph_id,rep,graph_seed,sim_seed,max_mcs,censored,Text_attempts,Text_mcs,extinct\n";
        }
        fout << std::fixed << std::setprecision(6);

        for (int graph_id = 0; graph_id < num_graphs; ++graph_id) {
            const uint64_t graph_seed = (base_seed ^ 0xC6A4A7935BD1E995ULL)
                                      + static_cast<uint64_t>(graph_id) * 0x9E3779B97F4A7C15ULL;

            Graph g = build_graph(graph_type, size_param, degree_param, beta, graph_seed);
            const long long max_attempts = (max_mcs > 0) ? max_mcs * static_cast<long long>(g.N) : -1;

            for (int rep = 0; rep < reps_per_graph; ++rep) {
                const uint64_t sim_seed = (graph_seed ^ 0xA5A5A5A5A5A5A5A5ULL)
                                        + static_cast<uint64_t>(rep) * 1315423911ULL;

                RPS_Sim sim(g, sim_seed, k);
                sim.init_random_uniform();
                const SimResult res = sim.run_until_extinction(max_attempts);
                const int censored = (res.extinct == -1) ? 1 : 0;

                fout << graph_type << ','
                     << g.N << ','
                     << size_param << ','
                     << degree_param << ','
                     << beta << ','
                     << k << ','
                     << graph_id << ','
                     << rep << ','
                     << graph_seed << ','
                     << sim_seed << ','
                     << max_mcs << ','
                     << censored << ','
                     << res.Text_attempts << ','
                     << res.Text_mcs << ','
                     << species_name(res.extinct) << '\n';
            }
        }

        std::cout << "Wrote extinction statistics with graph averaging: " << out_csv << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
