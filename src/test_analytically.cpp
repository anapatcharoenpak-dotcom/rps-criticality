#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <iomanip>
#include <stdexcept>
#include <cstdint>
#include <cmath>
#include <algorithm>

#include "graph.hpp"
#include "rng.hpp"
#include "rps_sim.hpp"

// Forward declaration from graph_builders.cpp
Graph build_lattice2D(int L);

namespace {

constexpr int NUM_SITES = 4;
constexpr int NUM_STATES = 81; // 3^4

struct Stats {
    int nR = 0;
    int nP = 0;
    int nS = 0;
};

static std::array<Species, NUM_SITES> decode_state(int code) {
    std::array<Species, NUM_SITES> s{};
    for (int i = 0; i < NUM_SITES; ++i) {
        s[i] = static_cast<Species>(code % 3);
        code /= 3;
    }
    return s;
}

static int encode_state(const std::array<Species, NUM_SITES>& s) {
    int code = 0;
    int mul = 1;
    for (int i = 0; i < NUM_SITES; ++i) {
        code += static_cast<int>(s[i]) * mul;
        mul *= 3;
    }
    return code;
}

static Stats count_species(const std::array<Species, NUM_SITES>& s) {
    Stats out;
    for (Species x : s) {
        if (x == ROCK) out.nR++;
        else if (x == PAPER) out.nP++;
        else out.nS++;
    }
    return out;
}

static bool is_extinct_state(int code) {
    const auto s = decode_state(code);
    const Stats c = count_species(s);
    return (c.nR == 0 || c.nP == 0 || c.nS == 0);
}

static std::string state_to_string(int code) {
    const auto s = decode_state(code);
    std::string out = "[";
    for (int i = 0; i < NUM_SITES; ++i) {
        char ch = '?';
        if (s[i] == ROCK) ch = 'R';
        else if (s[i] == PAPER) ch = 'P';
        else ch = 'S';
        out += ch;
        if (i + 1 != NUM_SITES) out += ' ';
    }
    out += "]";
    return out;
}

static std::vector<std::vector<double>> build_transition_matrix(const Graph& g, double k) {
    if (g.N != NUM_SITES) {
        throw std::runtime_error("This test expects a 2x2 lattice with exactly 4 sites.");
    }

    std::vector<std::vector<double>> P(NUM_STATES, std::vector<double>(NUM_STATES, 0.0));

    for (int code = 0; code < NUM_STATES; ++code) {
        const auto s = decode_state(code);

        for (int i = 0; i < g.N; ++i) {
            const auto& nb = g.neighbors(i);
            const double pick_ij = 1.0 / static_cast<double>(g.N) / static_cast<double>(nb.size());

            for (int j : nb) {
                const Species si = s[i];
                const Species sj = s[j];

                if (si == sj) {
                    P[code][code] += pick_ij;
                    continue;
                }

                if (beats(si, sj)) {
                    auto t = s;
                    t[j] = si;
                    const int next_code = encode_state(t);
                    P[code][next_code] += pick_ij * k;
                    P[code][code] += pick_ij * (1.0 - k);
                } else {
                    auto t = s;
                    t[i] = sj;
                    const int next_code = encode_state(t);
                    P[code][next_code] += pick_ij * k;
                    P[code][code] += pick_ij * (1.0 - k);
                }
            }
        }
    }

    return P;
}

// Solve A x = b by Gaussian elimination with partial pivoting.
static std::vector<double> solve_linear_system(std::vector<std::vector<double>> A,
                                               std::vector<double> b) {
    const int n = static_cast<int>(A.size());
    if (static_cast<int>(b.size()) != n) {
        throw std::runtime_error("Dimension mismatch in linear solver.");
    }

    for (int col = 0; col < n; ++col) {
        int pivot = col;
        double best = std::fabs(A[col][col]);
        for (int r = col + 1; r < n; ++r) {
            const double cand = std::fabs(A[r][col]);
            if (cand > best) {
                best = cand;
                pivot = r;
            }
        }

        if (best < 1e-14) {
            throw std::runtime_error("Singular matrix in analytic extinction-time solve.");
        }

        if (pivot != col) {
            std::swap(A[pivot], A[col]);
            std::swap(b[pivot], b[col]);
        }

        const double diag = A[col][col];
        for (int j = col; j < n; ++j) A[col][j] /= diag;
        b[col] /= diag;

        for (int r = 0; r < n; ++r) {
            if (r == col) continue;
            const double factor = A[r][col];
            if (std::fabs(factor) < 1e-18) continue;
            for (int j = col; j < n; ++j) A[r][j] -= factor * A[col][j];
            b[r] -= factor * b[col];
        }
    }

    return b;
}

struct AnalyticResult {
    std::vector<double> hitting_attempts_by_state; // size 81; extinct states = 0
    double mean_attempts_unconditional = 0.0;
    double mean_attempts_conditional = 0.0;
    int transient_count = 0;
    int extinct_count = 0;
};

static AnalyticResult compute_analytic_extinction_times(const std::vector<std::vector<double>>& P) {
    std::vector<int> transient_states;
    std::vector<int> extinct_states;
    transient_states.reserve(NUM_STATES);
    extinct_states.reserve(NUM_STATES);

    for (int code = 0; code < NUM_STATES; ++code) {
        if (is_extinct_state(code)) extinct_states.push_back(code);
        else transient_states.push_back(code);
    }

    const int m = static_cast<int>(transient_states.size());
    std::vector<int> pos(NUM_STATES, -1);
    for (int i = 0; i < m; ++i) pos[transient_states[i]] = i;

    // (I - Q) t = 1
    std::vector<std::vector<double>> A(m, std::vector<double>(m, 0.0));
    std::vector<double> rhs(m, 1.0);

    for (int row = 0; row < m; ++row) {
        const int s = transient_states[row];
        A[row][row] = 1.0;
        for (int next = 0; next < NUM_STATES; ++next) {
            const int col = pos[next];
            if (col != -1) {
                A[row][col] -= P[s][next];
            }
        }
    }

    const std::vector<double> t = solve_linear_system(A, rhs);

    AnalyticResult out;
    out.hitting_attempts_by_state.assign(NUM_STATES, 0.0);
    out.transient_count = m;
    out.extinct_count = static_cast<int>(extinct_states.size());

    double sum_cond = 0.0;
    for (int i = 0; i < m; ++i) {
        out.hitting_attempts_by_state[transient_states[i]] = t[i];
        sum_cond += t[i];
    }

    double sum_all = 0.0;
    for (double x : out.hitting_attempts_by_state) sum_all += x;

    out.mean_attempts_conditional = sum_cond / static_cast<double>(m);
    out.mean_attempts_unconditional = sum_all / static_cast<double>(NUM_STATES);
    return out;
}

static bool nearly_equal(double a, double b, double tol = 1e-10) {
    return std::fabs(a - b) <= tol;
}

static void validate_transition_matrix(const std::vector<std::vector<double>>& P) {
    for (int i = 0; i < NUM_STATES; ++i) {
        double row_sum = 0.0;
        for (double x : P[i]) row_sum += x;
        if (!nearly_equal(row_sum, 1.0, 1e-10)) {
            throw std::runtime_error("Transition matrix row does not sum to 1 for state code " + std::to_string(i));
        }
    }
}

struct NumericResult {
    double mean_attempts_unconditional = 0.0;
    double mean_attempts_conditional = 0.0;
    long long accepted_conditional_runs = 0;
};

static NumericResult run_monte_carlo(const Graph& g, double k, long long reps, uint64_t base_seed) {
    NumericResult out;

    double sum_unconditional = 0.0;
    double sum_conditional = 0.0;
    long long n_conditional = 0;

    for (long long rep = 0; rep < reps; ++rep) {
        const uint64_t seed = (base_seed ^ 0xA5A5A5A5A5A5A5A5ULL) + static_cast<uint64_t>(rep) * 1315423911ULL;

        RPS_Sim sim(g, seed, k);
        sim.init_random_uniform();
        SimResult res = sim.run_until_extinction(-1);
        sum_unconditional += static_cast<double>(res.Text_attempts);

        if (sim.nR > 0 && sim.nP > 0 && sim.nS > 0) {
            // impossible here, because run_until_extinction ended at extinction
        }

        // Recover whether the initial state contained all three species.
        // After initialization and before running, counts were available in sim.nR/nP/nS,
        // but run_until_extinction mutates them. So rebuild using a second sim with same seed.
        RPS_Sim sim_init_only(g, seed, k);
        sim_init_only.init_random_uniform();
        if (sim_init_only.nR > 0 && sim_init_only.nP > 0 && sim_init_only.nS > 0) {
            sum_conditional += static_cast<double>(res.Text_attempts);
            n_conditional++;
        }
    }

    out.mean_attempts_unconditional = sum_unconditional / static_cast<double>(reps);
    out.accepted_conditional_runs = n_conditional;
    out.mean_attempts_conditional = (n_conditional > 0)
        ? (sum_conditional / static_cast<double>(n_conditional))
        : 0.0;
    return out;
}

static void print_top_transient_examples(const AnalyticResult& ar, int how_many = 8) {
    std::vector<std::pair<double, int>> items;
    for (int code = 0; code < NUM_STATES; ++code) {
        if (!is_extinct_state(code)) {
            items.push_back({ar.hitting_attempts_by_state[code], code});
        }
    }
    std::sort(items.begin(), items.end(), [](const auto& a, const auto& b) {
        if (a.first != b.first) return a.first > b.first;
        return a.second < b.second;
    });

    std::cout << "\nSome transient states with the largest analytic mean extinction times\n";
    std::cout << "---------------------------------------------------------------\n";
    for (int i = 0; i < std::min<int>(how_many, static_cast<int>(items.size())); ++i) {
        std::cout << std::setw(2) << (i + 1) << ". code=" << std::setw(2) << items[i].second
                  << "  state=" << state_to_string(items[i].second)
                  << "  T=" << std::fixed << std::setprecision(6) << items[i].first << " attempts\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        double k = 1.0;
        long long reps = 200000;
        uint64_t seed = 12345ULL;

        if (argc >= 2) k = std::stod(argv[1]);
        if (argc >= 3) reps = std::stoll(argv[2]);
        if (argc >= 4) seed = static_cast<uint64_t>(std::stoull(argv[3]));
        if (argc > 4) {
            throw std::runtime_error("Usage: test_analytically.exe [k] [reps] [seed]");
        }

        if (k < 0.0 || k > 1.0) {
            throw std::runtime_error("This test expects 0 <= k <= 1.");
        }
        if (reps <= 0) {
            throw std::runtime_error("reps must be positive.");
        }

        Graph g = build_lattice2D(2);

        std::cout << "2x2 lattice analytic vs numerical extinction-time test\n";
        std::cout << "=====================================================\n";
        std::cout << "Interpretation of extinction: stop when at least one species count becomes zero.\n";
        std::cout << "This matches run_until_extinction() in your current simulation code.\n\n";

        std::cout << "Graph summary\n";
        std::cout << "-------------\n";
        std::cout << "name      : " << g.name << "\n";
        std::cout << "N         : " << g.N << "\n";
        std::cout << "degrees   : ";
        for (int i = 0; i < g.N; ++i) {
            std::cout << g.adj[i].size();
            if (i + 1 != g.N) std::cout << ", ";
        }
        std::cout << "\n\n";

        const auto P = build_transition_matrix(g, k);
        validate_transition_matrix(P);
        const AnalyticResult analytic = compute_analytic_extinction_times(P);
        const NumericResult numeric = run_monte_carlo(g, k, reps, seed);

        const double analytic_mcs_uncond = analytic.mean_attempts_unconditional / static_cast<double>(g.N);
        const double analytic_mcs_cond = analytic.mean_attempts_conditional / static_cast<double>(g.N);
        const double numeric_mcs_uncond = numeric.mean_attempts_unconditional / static_cast<double>(g.N);
        const double numeric_mcs_cond = numeric.mean_attempts_conditional / static_cast<double>(g.N);

        std::cout << "State counting\n";
        std::cout << "-------------\n";
        std::cout << "total states                  : " << NUM_STATES << "\n";
        std::cout << "transient states (all 3 alive): " << analytic.transient_count << "\n";
        std::cout << "extinction states             : " << analytic.extinct_count << "\n\n";

        std::cout << std::fixed << std::setprecision(8);
        std::cout << "Extinction time comparison\n";
        std::cout << "--------------------------\n";
        std::cout << "Parameter k                  : " << k << "\n";
        std::cout << "Monte Carlo reps             : " << reps << "\n";
        std::cout << "Base seed                    : " << seed << "\n\n";

        std::cout << "A) Unconditional average over init_random_uniform()\n";
        std::cout << "   analytic mean attempts    : " << analytic.mean_attempts_unconditional << "\n";
        std::cout << "   numeric  mean attempts    : " << numeric.mean_attempts_unconditional << "\n";
        std::cout << "   absolute difference       : " << std::fabs(analytic.mean_attempts_unconditional - numeric.mean_attempts_unconditional) << "\n";
        std::cout << "   analytic mean MCS         : " << analytic_mcs_uncond << "\n";
        std::cout << "   numeric  mean MCS         : " << numeric_mcs_uncond << "\n\n";

        std::cout << "B) Conditional average given all three species are present initially\n";
        std::cout << "   analytic mean attempts    : " << analytic.mean_attempts_conditional << "\n";
        std::cout << "   numeric  mean attempts    : " << numeric.mean_attempts_conditional << "\n";
        std::cout << "   absolute difference       : " << std::fabs(analytic.mean_attempts_conditional - numeric.mean_attempts_conditional) << "\n";
        std::cout << "   analytic mean MCS         : " << analytic_mcs_cond << "\n";
        std::cout << "   numeric  mean MCS         : " << numeric_mcs_cond << "\n";
        std::cout << "   accepted conditional runs : " << numeric.accepted_conditional_runs << " / " << reps << "\n";

        print_top_transient_examples(analytic);

        std::cout << "\nSuggested compile command\n";
        std::cout << "-------------------------\n";
        std::cout << "g++ -O2 -std=c++17 src/test_analytically.cpp src/graph_builders.cpp src/rps_sim.cpp -o test_analytically.exe\n";

        std::cout << "\nSuggested run command\n";
        std::cout << "---------------------\n";
        std::cout << ".\\test_analytically.exe 1.0 200000 12345\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
