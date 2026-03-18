#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "graph.hpp"
#include "rng.hpp"

// Forward declarations from graph_builders.cpp
Graph build_watts_strogatz(int N, int K, double beta, RNG& rng);
Graph build_barabasi_albert(int N, int m0, int m, RNG& rng);

enum Species : uint8_t { ROCK = 0, PAPER = 1, SCISSORS = 2 };

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
    int height = N / width;
    return {width, height};
}

struct TrajectoryPoint {
    long long attempts = 0;
    double mcs = 0.0;
    int nR = 0;
    int nP = 0;
    int nS = 0;
};

struct SimpleRPS {
    const Graph& g;
    RNG rng;
    std::vector<uint8_t> state;
    int nR = 0;
    int nP = 0;
    int nS = 0;
    double kR = 1.0;
    double kP = 1.0;
    double kS = 1.0;

    SimpleRPS(const Graph& graph, uint64_t seed, double k)
        : g(graph), rng(seed), state(graph.N, 0), kR(k), kP(k), kS(k) {}

    void init_random_uniform() {
        nR = nP = nS = 0;
        for (int i = 0; i < g.N; ++i) {
            const int x = rng.randint(0, 2);
            state[i] = static_cast<uint8_t>(x);
            if (x == ROCK) ++nR;
            else if (x == PAPER) ++nP;
            else ++nS;
        }
    }

    bool extinct() const {
        return (nR == 0 || nP == 0 || nS == 0);
    }

    double invasion_rate(uint8_t winner) const {
        if (winner == ROCK) return kR;
        if (winner == PAPER) return kP;
        return kS;
    }

    bool step_attempt() {
        static constexpr uint8_t prey_of[3] = {SCISSORS, ROCK, PAPER};

        const int i = rng.randint(0, g.N - 1);
        const auto& nb = g.neighbors(i);
        const int j = nb[rng.randint(0, static_cast<int>(nb.size()) - 1)];

        const uint8_t si = state[i];
        const uint8_t sj = state[j];
        if (si == sj) return false;

        const bool i_wins = (prey_of[si] == sj);
        const uint8_t winner = i_wins ? si : sj;
        const uint8_t loser = i_wins ? sj : si;
        const int loser_idx = i_wins ? j : i;
        const double rate = invasion_rate(winner);

        if (rate <= 0.0) return false;
        if (rate < 1.0 && rng.u01() >= rate) return false;

        if (loser == ROCK) --nR;
        else if (loser == PAPER) --nP;
        else --nS;

        if (winner == ROCK) ++nR;
        else if (winner == PAPER) ++nP;
        else ++nS;

        state[loser_idx] = winner;
        return extinct();
    }
};

Graph build_graph(const std::string& graph_type, int size_param, int degree_param, double beta, uint64_t seed) {
    if (graph_type == "lattice2D") {
        const auto [width, height] = choose_lattice_shape(size_param);
        return build_lattice_rectangular(width, height);
    }

    RNG graph_rng(seed ^ 0x9e3779b97f4a7c15ULL);

    if (graph_type == "smallworld") {
        return build_watts_strogatz(size_param, degree_param, beta, graph_rng);
    }
    if (graph_type == "scalefree") {
        const int m = degree_param;
        const int m0 = 2 * m;
        return build_barabasi_albert(size_param, m0, m, graph_rng);
    }

    throw std::runtime_error("Unknown graph_type. Use lattice2D, smallworld, or scalefree.");
}

void write_trajectory_csv(const std::string& filename,
                          const std::string& graph_type,
                          int size_param,
                          int degree_param,
                          double beta,
                          double k,
                          uint64_t seed,
                          long long max_mcs,
                          int record_every_mcs) {
    Graph g = build_graph(graph_type, size_param, degree_param, beta, seed);
    SimpleRPS sim(g, seed, k);
    sim.init_random_uniform();

    const long long max_attempts = max_mcs * static_cast<long long>(g.N);
    const long long record_every_attempts = static_cast<long long>(record_every_mcs) * g.N;

    std::ofstream fout(filename);
    if (!fout) throw std::runtime_error("Cannot open output CSV: " + filename);

    fout << "graph,N,size_param,degree_param,beta,k,seed,attempts,mcs,nR,nP,nS,fR,fP,fS\n";
    fout << std::fixed << std::setprecision(6);

    auto write_row = [&](long long attempts) {
        fout << graph_type << ','
             << g.N << ','
             << size_param << ','
             << degree_param << ','
             << beta << ','
             << k << ','
             << seed << ','
             << attempts << ','
             << (static_cast<double>(attempts) / static_cast<double>(g.N)) << ','
             << sim.nR << ','
             << sim.nP << ','
             << sim.nS << ','
             << (static_cast<double>(sim.nR) / g.N) << ','
             << (static_cast<double>(sim.nP) / g.N) << ','
             << (static_cast<double>(sim.nS) / g.N) << '\n';
    };

    long long attempts = 0;
    long long next_record = 0;
    write_row(0);
    next_record += record_every_attempts;

    while (attempts < max_attempts && !sim.extinct()) {
        ++attempts;
        sim.step_attempt();
        if (attempts >= next_record) {
            write_row(attempts);
            next_record += record_every_attempts;
        }
    }

    if ((attempts % record_every_attempts) != 0 || attempts == 0) {
        // Ensure the final state is always present.
        write_row(attempts);
    }

    std::cout << "Wrote trajectory: " << filename << "\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 10) {
            std::cerr
                << "Usage:\n"
                << "  " << argv[0]
                << " graph_type size_param degree_param beta k seed max_mcs record_every_mcs out_csv\n\n"
                << "Examples:\n"
                << "  " << argv[0] << " lattice2D 10 0 0.0 1.0 12345 200 1 baseline_lattice.csv\n"
                << "  " << argv[0] << " smallworld 100 4 0.10 1.0 12345 200 1 baseline_smallworld.csv\n"
                << "  " << argv[0] << " scalefree 100 2 0.0 1.0 12345 200 1 baseline_scalefree.csv\n";
            return 1;
        }

        const std::string graph_type = argv[1];
        const int size_param = std::stoi(argv[2]);
        const int degree_param = std::stoi(argv[3]);
        const double beta = std::stod(argv[4]);
        const double k = std::stod(argv[5]);
        const uint64_t seed = static_cast<uint64_t>(std::stoull(argv[6]));
        const long long max_mcs = std::stoll(argv[7]);
        const int record_every_mcs = std::stoi(argv[8]);
        const std::string out_csv = argv[9];

        if (max_mcs <= 0) throw std::runtime_error("max_mcs must be > 0.");
        if (record_every_mcs <= 0) throw std::runtime_error("record_every_mcs must be > 0.");

        write_trajectory_csv(out_csv, graph_type, size_param, degree_param, beta, k, seed, max_mcs, record_every_mcs);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
