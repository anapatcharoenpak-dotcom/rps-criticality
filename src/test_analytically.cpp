#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <stdexcept>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <sstream>
#include <chrono>

#include "graph.hpp"
#include "rng.hpp"
#include "rps_sim.hpp"

Graph build_lattice2D(int L);
Graph build_watts_strogatz(int N, int K, double beta, RNG& rng);
Graph build_barabasi_albert(int N, int m0, int m, RNG& rng);

namespace {

struct SpeciesCount {
    int nR = 0;
    int nP = 0;
    int nS = 0;
};

struct AnalyticResult {
    std::vector<double> hitting_attempts_by_state;
    double mean_attempts_unconditional = 0.0;
    double mean_attempts_conditional = 0.0;
    int total_states = 0;
    int transient_count = 0;
    int extinct_count = 0;
};

struct NumericResult {
    double mean_attempts_unconditional = 0.0;
    double mean_attempts_conditional = 0.0;
    long long accepted_conditional_runs = 0;
};

struct ConvergenceRow {
    std::string case_key;
    long long reps = 0;
    double log10_reps = 0.0;
    int ensemble_size = 0;
    double pct_error_unconditional = 0.0;
    double std_pct_error_unconditional = 0.0;
    double pct_error_conditional = 0.0;
    double std_pct_error_conditional = 0.0;
    double elapsed_seconds = 0.0;
};

struct CaseMember {
    Graph graph;
    AnalyticResult analytic;
    uint64_t member_seed = 0;
};

struct CaseEnsemble {
    std::string case_key;
    std::string title;
    std::string parameter_text;
    std::vector<CaseMember> members;
};

static int ipow3(int n) {
    int out = 1;
    for (int i = 0; i < n; ++i) out *= 3;
    return out;
}

static std::vector<Species> decode_state(int code, int n_sites) {
    std::vector<Species> s(n_sites, ROCK);
    for (int i = 0; i < n_sites; ++i) {
        s[i] = static_cast<Species>(code % 3);
        code /= 3;
    }
    return s;
}

static int encode_state(const std::vector<Species>& s) {
    int code = 0;
    int mul = 1;
    for (Species x : s) {
        code += static_cast<int>(x) * mul;
        mul *= 3;
    }
    return code;
}

static SpeciesCount count_species(const std::vector<Species>& s) {
    SpeciesCount c;
    for (Species x : s) {
        if (x == ROCK) c.nR++;
        else if (x == PAPER) c.nP++;
        else c.nS++;
    }
    return c;
}

static bool is_extinct_state(int code, int n_sites) {
    const auto s = decode_state(code, n_sites);
    const auto c = count_species(s);
    return (c.nR == 0 || c.nP == 0 || c.nS == 0);
}

static std::string state_to_string(int code, int n_sites) {
    const auto s = decode_state(code, n_sites);
    std::string out = "[";
    for (int i = 0; i < n_sites; ++i) {
        char ch = '?';
        if (s[i] == ROCK) ch = 'R';
        else if (s[i] == PAPER) ch = 'P';
        else ch = 'S';
        out += ch;
        if (i + 1 != n_sites) out += ' ';
    }
    out += "]";
    return out;
}

static uint64_t mix_seed(uint64_t a, uint64_t b, uint64_t c = 0) {
    uint64_t x = a + 0x9e3779b97f4a7c15ULL;
    x ^= b + 0x9e3779b97f4a7c15ULL + (x << 6) + (x >> 2);
    x ^= c + 0x9e3779b97f4a7c15ULL + (x << 6) + (x >> 2);
    return x;
}

static std::vector<std::vector<double>> build_transition_matrix(const Graph& g, double k) {
    const int total_states = ipow3(g.N);
    std::vector<std::vector<double>> P(total_states, std::vector<double>(total_states, 0.0));

    for (int code = 0; code < total_states; ++code) {
        const auto s = decode_state(code, g.N);

        for (int i = 0; i < g.N; ++i) {
            const auto& nb = g.neighbors(i);
            if (nb.empty()) {
                throw std::runtime_error("Graph contains an isolated node; analytic benchmark expects no isolated nodes.");
            }
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

static void validate_transition_matrix(const std::vector<std::vector<double>>& P) {
    for (int i = 0; i < static_cast<int>(P.size()); ++i) {
        double row_sum = 0.0;
        for (double x : P[i]) row_sum += x;
        if (std::fabs(row_sum - 1.0) > 1e-10) {
            throw std::runtime_error("Transition matrix row does not sum to 1 for state code " + std::to_string(i));
        }
    }
}

static std::vector<double> solve_linear_system(std::vector<std::vector<double>> A,
                                               std::vector<double> b) {
    const int n = static_cast<int>(A.size());
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

static AnalyticResult compute_analytic_extinction_times(const Graph& g,
                                                        const std::vector<std::vector<double>>& P) {
    const int total_states = ipow3(g.N);
    std::vector<int> transient_states;
    std::vector<int> extinct_states;

    for (int code = 0; code < total_states; ++code) {
        if (is_extinct_state(code, g.N)) extinct_states.push_back(code);
        else transient_states.push_back(code);
    }

    const int m = static_cast<int>(transient_states.size());
    std::vector<int> pos(total_states, -1);
    for (int i = 0; i < m; ++i) pos[transient_states[i]] = i;

    std::vector<std::vector<double>> A(m, std::vector<double>(m, 0.0));
    std::vector<double> rhs(m, 1.0);

    for (int row = 0; row < m; ++row) {
        const int s = transient_states[row];
        A[row][row] = 1.0;
        for (int next = 0; next < total_states; ++next) {
            const int col = pos[next];
            if (col != -1) A[row][col] -= P[s][next];
        }
    }

    const std::vector<double> t = solve_linear_system(A, rhs);

    AnalyticResult out;
    out.hitting_attempts_by_state.assign(total_states, 0.0);
    out.total_states = total_states;
    out.transient_count = m;
    out.extinct_count = static_cast<int>(extinct_states.size());

    double sum_cond = 0.0;
    for (int i = 0; i < m; ++i) {
        out.hitting_attempts_by_state[transient_states[i]] = t[i];
        sum_cond += t[i];
    }

    double sum_all = 0.0;
    for (double x : out.hitting_attempts_by_state) sum_all += x;

    out.mean_attempts_conditional = (m > 0) ? (sum_cond / static_cast<double>(m)) : 0.0;
    out.mean_attempts_unconditional = sum_all / static_cast<double>(total_states);
    return out;
}

static NumericResult run_monte_carlo(const Graph& g, double k, long long reps, uint64_t base_seed) {
    NumericResult out;
    double sum_unconditional = 0.0;
    double sum_conditional = 0.0;
    long long n_conditional = 0;

    for (long long rep = 0; rep < reps; ++rep) {
        const uint64_t seed = mix_seed(base_seed, static_cast<uint64_t>(rep));

        RPS_Sim sim(g, seed, k);
        sim.init_random_uniform();
        const bool all_three_initial = (sim.nR > 0 && sim.nP > 0 && sim.nS > 0);
        const SimResult res = sim.run_until_extinction();

        sum_unconditional += static_cast<double>(res.Text_attempts);
        if (all_three_initial) {
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

static void print_graph_summary(const Graph& g) {
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

    std::cout << "Graph connections (adjacency list)\n";
    std::cout << "--------------------------------\n";
    for (int i = 0; i < g.N; ++i) {
        std::cout << "node " << i << " -> ";
        for (std::size_t j = 0; j < g.adj[i].size(); ++j) {
            std::cout << g.adj[i][j];
            if (j + 1 != g.adj[i].size()) std::cout << ", ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

static void print_top_transient_examples(const AnalyticResult& ar, int n_sites, int how_many = 5) {
    std::vector<std::pair<double, int>> items;
    for (int code = 0; code < ar.total_states; ++code) {
        if (!is_extinct_state(code, n_sites)) items.push_back({ar.hitting_attempts_by_state[code], code});
    }
    std::sort(items.begin(), items.end(), [](const auto& a, const auto& b) {
        if (a.first != b.first) return a.first > b.first;
        return a.second < b.second;
    });

    std::cout << "Some transient states with the largest analytic mean extinction times\n";
    std::cout << "---------------------------------------------------------------\n";
    for (int i = 0; i < std::min<int>(how_many, static_cast<int>(items.size())); ++i) {
        std::cout << std::setw(2) << (i + 1)
                  << ". code=" << std::setw(4) << items[i].second
                  << "  state=" << state_to_string(items[i].second, n_sites)
                  << "  T=" << std::fixed << std::setprecision(6) << items[i].first << " attempts\n";
    }
    std::cout << "\n";
}

static double percent_error(double numeric, double analytic) {
    if (analytic == 0.0) return 0.0;
    return 100.0 * std::fabs(numeric - analytic) / std::fabs(analytic);
}

static double mean_of(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

static double stdev_of(const std::vector<double>& v) {
    if (v.size() <= 1) return 0.0;
    const double m = mean_of(v);
    double acc = 0.0;
    for (double x : v) acc += (x - m) * (x - m);
    return std::sqrt(acc / static_cast<double>(v.size()));
}

static void write_csv_header(std::ofstream& out) {
    out << "case_key,reps,log10_reps,ensemble_size,pct_error_unconditional,std_pct_error_unconditional,"
           "pct_error_conditional,std_pct_error_conditional,elapsed_seconds\n";
}

static std::vector<std::pair<double, long long>> build_rep_schedule(double start_log10 = 3.0,
                                                                    double stop_log10 = 6.0,
                                                                    double step_log10 = 0.05) {
    std::vector<std::pair<double, long long>> schedule;
    double e = start_log10;
    long long previous = -1;
    while (e <= stop_log10 + 1e-12) {
        long long reps = static_cast<long long>(std::llround(std::pow(10.0, e)));
        if (reps <= previous) {
            e += step_log10;
            continue;
        }
        schedule.push_back({e, reps});
        previous = reps;
        e += step_log10;
    }
    return schedule;
}

static std::vector<ConvergenceRow> run_convergence_case(const CaseEnsemble& ensemble,
                                                        double k,
                                                        uint64_t sim_seed,
                                                        const std::vector<std::pair<double, long long>>& schedule,
                                                        double time_budget_seconds,
                                                        std::ostream& log_stream) {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    std::vector<ConvergenceRow> rows;

    for (std::size_t p = 0; p < schedule.size(); ++p) {
        const double elapsed = std::chrono::duration<double>(clock::now() - t0).count();
        if (elapsed >= time_budget_seconds) break;

        const double log10_reps = schedule[p].first;
        const long long reps = schedule[p].second;

        std::vector<double> err_uncond;
        std::vector<double> err_cond;
        err_uncond.reserve(ensemble.members.size());
        err_cond.reserve(ensemble.members.size());

        for (std::size_t m = 0; m < ensemble.members.size(); ++m) {
            const auto& member = ensemble.members[m];
            const uint64_t member_run_seed = mix_seed(sim_seed, member.member_seed, static_cast<uint64_t>(p));
            const NumericResult nr = run_monte_carlo(member.graph, k, reps, member_run_seed);
            err_uncond.push_back(percent_error(nr.mean_attempts_unconditional, member.analytic.mean_attempts_unconditional));
            err_cond.push_back(percent_error(nr.mean_attempts_conditional, member.analytic.mean_attempts_conditional));
        }

        ConvergenceRow row;
        row.case_key = ensemble.case_key;
        row.reps = reps;
        row.log10_reps = log10_reps;
        row.ensemble_size = static_cast<int>(ensemble.members.size());
        row.pct_error_unconditional = mean_of(err_uncond);
        row.std_pct_error_unconditional = stdev_of(err_uncond);
        row.pct_error_conditional = mean_of(err_cond);
        row.std_pct_error_conditional = stdev_of(err_cond);
        row.elapsed_seconds = std::chrono::duration<double>(clock::now() - t0).count();
        rows.push_back(row);

        log_stream << "  reps=10^" << std::fixed << std::setprecision(2) << log10_reps
                   << " (~" << reps << ")"
                   << "  mean err uncond=" << std::setprecision(6) << row.pct_error_unconditional << "%"
                   << "  mean err cond=" << row.pct_error_conditional << "%"
                   << "  elapsed=" << std::setprecision(1) << row.elapsed_seconds << " s\n";
    }

    return rows;
}

static std::pair<double, double> fit_loglog_slope(const std::vector<ConvergenceRow>& rows,
                                                  bool conditional) {
    std::vector<double> xs, ys;
    for (const auto& row : rows) {
        const double y = conditional ? row.pct_error_conditional : row.pct_error_unconditional;
        if (row.reps > 0 && y > 0.0) {
            xs.push_back(std::log10(static_cast<double>(row.reps)));
            ys.push_back(std::log10(y));
        }
    }
    if (xs.size() < 2) return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};

    const double mx = mean_of(xs);
    const double my = mean_of(ys);
    double sxx = 0.0, sxy = 0.0;
    for (std::size_t i = 0; i < xs.size(); ++i) {
        sxx += (xs[i] - mx) * (xs[i] - mx);
        sxy += (xs[i] - mx) * (ys[i] - my);
    }
    if (sxx == 0.0) return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
    const double slope = sxy / sxx;
    const double intercept = my - slope * mx;
    return {slope, intercept};
}

static void append_rows_to_csv(std::ofstream& out, const std::vector<ConvergenceRow>& rows) {
    out << std::fixed << std::setprecision(8);
    for (const auto& row : rows) {
        out << row.case_key << ','
            << row.reps << ','
            << row.log10_reps << ','
            << row.ensemble_size << ','
            << row.pct_error_unconditional << ','
            << row.std_pct_error_unconditional << ','
            << row.pct_error_conditional << ','
            << row.std_pct_error_conditional << ','
            << row.elapsed_seconds << '\n';
    }
}

static CaseEnsemble build_lattice_ensemble(int ensemble_size, double k, uint64_t graph_seed_unused) {
    (void)graph_seed_unused;
    CaseEnsemble out;
    out.case_key = "lattice2D";
    out.title = "2D lattice analytic vs numerical extinction-time test";
    out.parameter_text = "L=2 (so N=4, total state space = 3^4 = 81).";

    Graph g = build_lattice2D(2);
    const auto P = build_transition_matrix(g, k);
    validate_transition_matrix(P);
    const AnalyticResult ar = compute_analytic_extinction_times(g, P);

    out.members.reserve(ensemble_size);
    for (int i = 0; i < ensemble_size; ++i) {
        out.members.push_back({g, ar, static_cast<uint64_t>(i + 1)});
    }
    return out;
}

static CaseEnsemble build_smallworld_ensemble(int ensemble_size, double k, uint64_t graph_seed_base) {
    CaseEnsemble out;
    out.case_key = "smallworld";
    out.title = "Small-world analytic vs numerical extinction-time test";
    out.parameter_text = "N=6, K=2, beta=0.5 (fixed ensemble reused across all convergence points).";

    out.members.reserve(ensemble_size);
    for (int i = 0; i < ensemble_size; ++i) {
        const uint64_t seed = mix_seed(graph_seed_base, static_cast<uint64_t>(i), 11ULL);
        RNG rng(seed);
        Graph g = build_watts_strogatz(6, 2, 0.5, rng);
        const auto P = build_transition_matrix(g, k);
        validate_transition_matrix(P);
        const AnalyticResult ar = compute_analytic_extinction_times(g, P);
        out.members.push_back({g, ar, seed});
    }
    return out;
}

static CaseEnsemble build_scalefree_ensemble(int ensemble_size, double k, uint64_t graph_seed_base) {
    CaseEnsemble out;
    out.case_key = "scalefree";
    out.title = "Scale-free analytic vs numerical extinction-time test";
    out.parameter_text = "N=6, m0=3, m=2 (fixed ensemble reused across all convergence points).";

    out.members.reserve(ensemble_size);
    for (int i = 0; i < ensemble_size; ++i) {
        const uint64_t seed = mix_seed(graph_seed_base, static_cast<uint64_t>(i), 29ULL);
        RNG rng(seed);
        Graph g = build_barabasi_albert(6, 3, 2, rng);
        const auto P = build_transition_matrix(g, k);
        validate_transition_matrix(P);
        const AnalyticResult ar = compute_analytic_extinction_times(g, P);
        out.members.push_back({g, ar, seed});
    }
    return out;
}

static void print_representative_benchmark(const CaseEnsemble& ensemble,
                                           double k,
                                           long long reps,
                                           uint64_t sim_seed) {
    const auto& rep = ensemble.members.front();
    const NumericResult numeric = run_monte_carlo(rep.graph, k, reps, sim_seed);
    const double analytic_mcs_uncond = rep.analytic.mean_attempts_unconditional / static_cast<double>(rep.graph.N);
    const double analytic_mcs_cond = rep.analytic.mean_attempts_conditional / static_cast<double>(rep.graph.N);
    const double numeric_mcs_uncond = numeric.mean_attempts_unconditional / static_cast<double>(rep.graph.N);
    const double numeric_mcs_cond = numeric.mean_attempts_conditional / static_cast<double>(rep.graph.N);

    std::cout << "=============================================================\n";
    std::cout << ensemble.title << "\n";
    std::cout << "=============================================================\n";
    std::cout << "Topology parameters: " << ensemble.parameter_text << "\n\n";
    std::cout << "Interpretation of extinction: stop when at least one species count becomes zero.\n";
    std::cout << "This matches run_until_extinction() in your current simulation code.\n\n";

    print_graph_summary(rep.graph);

    std::cout << "State counting\n";
    std::cout << "-------------\n";
    std::cout << "total states                  : " << rep.analytic.total_states << "\n";
    std::cout << "transient states (all 3 alive): " << rep.analytic.transient_count << "\n";
    std::cout << "extinction states             : " << rep.analytic.extinct_count << "\n\n";

    std::cout << std::fixed << std::setprecision(8);
    std::cout << "Extinction time comparison\n";
    std::cout << "--------------------------\n";
    std::cout << "Parameter k                  : " << k << "\n";
    std::cout << "Monte Carlo reps             : " << reps << "\n";
    std::cout << "Base seed                    : " << sim_seed << "\n\n";

    std::cout << "A) Unconditional average over init_random_uniform()\n";
    std::cout << "   analytic mean attempts    : " << rep.analytic.mean_attempts_unconditional << "\n";
    std::cout << "   numeric  mean attempts    : " << numeric.mean_attempts_unconditional << "\n";
    std::cout << "   absolute difference       : "
              << std::fabs(rep.analytic.mean_attempts_unconditional - numeric.mean_attempts_unconditional) << "\n";
    std::cout << "   analytic mean MCS         : " << analytic_mcs_uncond << "\n";
    std::cout << "   numeric  mean MCS         : " << numeric_mcs_uncond << "\n\n";

    std::cout << "B) Conditional average given all three species are present initially\n";
    std::cout << "   analytic mean attempts    : " << rep.analytic.mean_attempts_conditional << "\n";
    std::cout << "   numeric  mean attempts    : " << numeric.mean_attempts_conditional << "\n";
    std::cout << "   absolute difference       : "
              << std::fabs(rep.analytic.mean_attempts_conditional - numeric.mean_attempts_conditional) << "\n";
    std::cout << "   analytic mean MCS         : " << analytic_mcs_cond << "\n";
    std::cout << "   numeric  mean MCS         : " << numeric_mcs_cond << "\n";
    std::cout << "   accepted conditional runs : " << numeric.accepted_conditional_runs << " / " << reps << "\n\n";

    print_top_transient_examples(rep.analytic, rep.graph.N, 5);
}

} // namespace

int main(int argc, char** argv) {
    try {
        double k = 1.0;
        long long benchmark_reps = 200000;
        uint64_t graph_seed = 12345ULL;
        uint64_t sim_seed = 67890ULL;
        std::string csv_path = "convergence_errors.csv";
        double time_budget_seconds = 7200.0; // 2 hours
        int ensemble_size = 100;

        if (argc >= 2) k = std::stod(argv[1]);
        if (argc >= 3) benchmark_reps = std::stoll(argv[2]);
        if (argc >= 4) graph_seed = static_cast<uint64_t>(std::stoull(argv[3]));
        if (argc >= 5) sim_seed = static_cast<uint64_t>(std::stoull(argv[4]));
        if (argc >= 6) csv_path = argv[5];
        if (argc >= 7) time_budget_seconds = std::stod(argv[6]);
        if (argc >= 8) ensemble_size = std::stoi(argv[7]);
        if (argc > 8) {
            throw std::runtime_error(
                "Usage: test_analytically.exe [k] [benchmark_reps] [graph_seed] [sim_seed] [csv_path] [time_budget_seconds] [ensemble_size]"
            );
        }
        if (k < 0.0 || k > 1.0) throw std::runtime_error("Expected 0 <= k <= 1.");
        if (benchmark_reps <= 0) throw std::runtime_error("benchmark_reps must be positive.");
        if (time_budget_seconds <= 0.0) throw std::runtime_error("time_budget_seconds must be positive.");
        if (ensemble_size <= 0) throw std::runtime_error("ensemble_size must be positive.");

        std::cout << "=========================================================\n";
        std::cout << "This program runs three exact Markov-chain benchmarks:\n";
        std::cout << "  1. 2x2 periodic lattice builder output (deduplicated to degree-2 graph at L=2)\n";
        std::cout << "  2. a small small-world graph ensemble\n";
        std::cout << "  3. a small scale-free graph ensemble\n\n";
        std::cout << "Convergence modification in this version:\n";
        std::cout << "  - x-axis schedule: 10^3, 10^3.05, ..., 10^6\n";
        std::cout << "  - smoothed with the same fixed ensemble at every x-axis point\n";
        std::cout << "  - default ensemble size = " << ensemble_size << "\n";
        std::cout << "  - default total time budget = " << time_budget_seconds << " s\n";
        std::cout << "=========================================================\n\n";

        const auto lattice = build_lattice_ensemble(ensemble_size, k, graph_seed);
        const auto smallworld = build_smallworld_ensemble(ensemble_size, k, mix_seed(graph_seed, 101ULL));
        const auto scalefree = build_scalefree_ensemble(ensemble_size, k, mix_seed(graph_seed, 202ULL));

        print_representative_benchmark(lattice, k, benchmark_reps, mix_seed(sim_seed, 1ULL));
        print_representative_benchmark(smallworld, k, benchmark_reps, mix_seed(sim_seed, 2ULL));
        print_representative_benchmark(scalefree, k, benchmark_reps, mix_seed(sim_seed, 3ULL));

        const auto schedule = build_rep_schedule(3.0, 6.0, 0.05);
        const double per_case_budget = time_budget_seconds / 3.0;

        std::ofstream csv(csv_path);
        if (!csv) throw std::runtime_error("Cannot open CSV output: " + csv_path);
        write_csv_header(csv);

        std::cout << "Convergence test\n";
        std::cout << "----------------\n";
        std::cout << "CSV output          : " << csv_path << "\n";
        std::cout << "Schedule            : 10^3, 10^3.05, ..., 10^6\n";
        std::cout << "Ensemble size       : " << ensemble_size << "\n";
        std::cout << "Time budget per case: " << per_case_budget << " s\n\n";

        std::cout << "[lattice2D]\n";
        const auto rows_lattice = run_convergence_case(lattice, k, mix_seed(sim_seed, 11ULL), schedule, per_case_budget, std::cout);
        append_rows_to_csv(csv, rows_lattice);
        std::cout << '\n';

        std::cout << "[smallworld]\n";
        const auto rows_sw = run_convergence_case(smallworld, k, mix_seed(sim_seed, 22ULL), schedule, per_case_budget, std::cout);
        append_rows_to_csv(csv, rows_sw);
        std::cout << '\n';

        std::cout << "[scalefree]\n";
        const auto rows_sf = run_convergence_case(scalefree, k, mix_seed(sim_seed, 33ULL), schedule, per_case_budget, std::cout);
        append_rows_to_csv(csv, rows_sf);
        std::cout << '\n';

        csv.close();

        std::vector<std::pair<std::string, std::vector<ConvergenceRow>>> summaries = {
            {"lattice2D", rows_lattice},
            {"smallworld", rows_sw},
            {"scalefree", rows_sf}
        };

        std::cout << "Slope summary (log10 percent error vs log10 reps)\n";
        std::cout << "-----------------------------------------------\n";
        for (const auto& item : summaries) {
            const auto fit_u = fit_loglog_slope(item.second, false);
            const auto fit_c = fit_loglog_slope(item.second, true);
            std::cout << item.first << "\n";
            std::cout << "  unconditional slope : " << fit_u.first << "\n";
            std::cout << "  conditional slope   : " << fit_c.first << "\n";
        }

        std::cout << "\nSuggested compile command\n";
        std::cout << "-------------------------\n";
        std::cout << "g++ -O2 -std=c++17 src/test_analytically.cpp src/graph_builders.cpp src/rps_sim.cpp -o test_analytically.exe\n\n";

        std::cout << "Suggested run command\n";
        std::cout << "---------------------\n";
        std::cout << ".\\test_analytically.exe 1.0 200000 12345 67890 convergence_errors.csv 7200 100\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
