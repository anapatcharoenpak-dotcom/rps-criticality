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

// Forward declarations from graph_builders.cpp
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

struct CaseSpec {
    std::string key;
    std::string label;
    Graph graph;
    std::string parameter_text;
    std::vector<Graph> convergence_graphs;
};

struct ConvergenceRow {
    std::string case_key;
    std::string case_label;
    long long reps = 0;
    double log10_reps = 0.0;
    int ensemble_size = 0;
    double analytic_unconditional = 0.0;
    double numeric_unconditional = 0.0;
    double abs_error_unconditional = 0.0;
    double pct_error_unconditional = 0.0;
    double std_pct_error_unconditional = 0.0;
    double analytic_conditional = 0.0;
    double numeric_conditional = 0.0;
    double abs_error_conditional = 0.0;
    double pct_error_conditional = 0.0;
    double std_pct_error_conditional = 0.0;
    double avg_accepted_conditional_runs = 0.0;
    double elapsed_seconds = 0.0;
};

struct SlopeInfo {
    bool ok = false;
    double slope = 0.0;
    double intercept = 0.0;
    int n_used = 0;
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

static std::vector<std::vector<double>> build_transition_matrix(const Graph& g, double k) {
    const int total_states = ipow3(g.N);
    std::vector<std::vector<double>> P(total_states, std::vector<double>(total_states, 0.0));

    for (int code = 0; code < total_states; ++code) {
        const auto s = decode_state(code, g.N);

        for (int i = 0; i < g.N; ++i) {
            const auto& nb = g.neighbors(i);
            if (nb.empty()) {
                throw std::runtime_error("Graph contains an isolated node; analytic test expects no isolated nodes.");
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

static bool nearly_equal(double a, double b, double tol = 1e-10) {
    return std::fabs(a - b) <= tol;
}

static void validate_transition_matrix(const std::vector<std::vector<double>>& P) {
    for (int i = 0; i < static_cast<int>(P.size()); ++i) {
        double row_sum = 0.0;
        for (double x : P[i]) row_sum += x;
        if (!nearly_equal(row_sum, 1.0, 1e-10)) {
            throw std::runtime_error("Transition matrix row does not sum to 1 for state code " + std::to_string(i));
        }
    }
}

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

static AnalyticResult compute_analytic_extinction_times(const Graph& g,
                                                        const std::vector<std::vector<double>>& P) {
    const int total_states = ipow3(g.N);
    std::vector<int> transient_states;
    std::vector<int> extinct_states;
    transient_states.reserve(total_states);
    extinct_states.reserve(total_states);

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
            if (col != -1) {
                A[row][col] -= P[s][next];
            }
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
        const uint64_t seed = (base_seed ^ 0xA5A5A5A5A5A5A5A5ULL) + static_cast<uint64_t>(rep) * 1315423911ULL;

        RPS_Sim sim(g, seed, k);
        sim.init_random_uniform();
        const bool initially_all_three_present = (sim.nR > 0 && sim.nP > 0 && sim.nS > 0);

        SimResult res = sim.run_until_extinction(-1);
        sum_unconditional += static_cast<double>(res.Text_attempts);

        if (initially_all_three_present) {
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
}

static void print_graph_connections(const Graph& g) {
    std::cout << "Graph connections (adjacency list)\n";
    std::cout << "--------------------------------\n";
    for (int i = 0; i < g.N; ++i) {
        std::cout << "node " << i << " -> ";
        for (size_t j = 0; j < g.adj[i].size(); ++j) {
            std::cout << g.adj[i][j];
            if (j + 1 != g.adj[i].size()) std::cout << ", ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

static void print_top_transient_examples(const Graph& g,
                                         const AnalyticResult& ar,
                                         int how_many = 5) {
    std::vector<std::pair<double, int>> items;
    for (int code = 0; code < ar.total_states; ++code) {
        if (!is_extinct_state(code, g.N)) {
            items.push_back({ar.hitting_attempts_by_state[code], code});
        }
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
                  << "  state=" << state_to_string(items[i].second, g.N)
                  << "  T=" << std::fixed << std::setprecision(6) << items[i].first
                  << " attempts\n";
    }
    std::cout << "\n";
}

static double percent_error(double analytic, double numeric) {
    if (std::fabs(analytic) < 1e-14) return 0.0;
    return 100.0 * std::fabs(numeric - analytic) / std::fabs(analytic);
}

static double mean_of(const std::vector<double>& xs) {
    if (xs.empty()) return 0.0;
    return std::accumulate(xs.begin(), xs.end(), 0.0) / static_cast<double>(xs.size());
}

static double stddev_of(const std::vector<double>& xs) {
    if (xs.size() < 2) return 0.0;
    const double m = mean_of(xs);
    double acc = 0.0;
    for (double x : xs) {
        const double d = x - m;
        acc += d * d;
    }
    return std::sqrt(acc / static_cast<double>(xs.size() - 1));
}

static std::vector<long long> build_log_reps_schedule(double exp_start, double exp_step, long long min_reps, long long max_reps_cap);

static ConvergenceRow build_smoothed_convergence_row(const CaseSpec& spec,
                                                     const std::vector<AnalyticResult>& analytics,
                                                     const std::vector<NumericResult>& numerics,
                                                     long long reps,
                                                     double elapsed_seconds) {
    if (analytics.size() != numerics.size() || analytics.empty()) {
        throw std::runtime_error("Convergence ensemble size mismatch.");
    }

    std::vector<double> analytic_uncond_vals, analytic_cond_vals;
    std::vector<double> numeric_uncond_vals, numeric_cond_vals;
    std::vector<double> abs_err_uncond_vals, abs_err_cond_vals;
    std::vector<double> pct_err_uncond_vals, pct_err_cond_vals;
    double accepted_sum = 0.0;

    for (size_t i = 0; i < analytics.size(); ++i) {
        const double a_u = analytics[i].mean_attempts_unconditional;
        const double a_c = analytics[i].mean_attempts_conditional;
        const double n_u = numerics[i].mean_attempts_unconditional;
        const double n_c = numerics[i].mean_attempts_conditional;

        analytic_uncond_vals.push_back(a_u);
        analytic_cond_vals.push_back(a_c);
        numeric_uncond_vals.push_back(n_u);
        numeric_cond_vals.push_back(n_c);
        abs_err_uncond_vals.push_back(std::fabs(n_u - a_u));
        abs_err_cond_vals.push_back(std::fabs(n_c - a_c));
        pct_err_uncond_vals.push_back(percent_error(a_u, n_u));
        pct_err_cond_vals.push_back(percent_error(a_c, n_c));
        accepted_sum += static_cast<double>(numerics[i].accepted_conditional_runs);
    }

    ConvergenceRow row;
    row.case_key = spec.key;
    row.case_label = spec.label;
    row.reps = reps;
    row.log10_reps = std::log10(static_cast<double>(reps));
    row.ensemble_size = static_cast<int>(analytics.size());
    row.analytic_unconditional = mean_of(analytic_uncond_vals);
    row.numeric_unconditional = mean_of(numeric_uncond_vals);
    row.abs_error_unconditional = mean_of(abs_err_uncond_vals);
    row.pct_error_unconditional = mean_of(pct_err_uncond_vals);
    row.std_pct_error_unconditional = stddev_of(pct_err_uncond_vals);
    row.analytic_conditional = mean_of(analytic_cond_vals);
    row.numeric_conditional = mean_of(numeric_cond_vals);
    row.abs_error_conditional = mean_of(abs_err_cond_vals);
    row.pct_error_conditional = mean_of(pct_err_cond_vals);
    row.std_pct_error_conditional = stddev_of(pct_err_cond_vals);
    row.avg_accepted_conditional_runs = accepted_sum / static_cast<double>(analytics.size());
    row.elapsed_seconds = elapsed_seconds;
    return row;
}

static std::vector<ConvergenceRow> run_convergence_log_sweep(const CaseSpec& spec,
                                                             double k,
                                                             uint64_t base_seed,
                                                             double time_budget_seconds) {
    const int ensemble_size = static_cast<int>(spec.convergence_graphs.size());
    if (ensemble_size <= 0) {
        throw std::runtime_error("No convergence graphs supplied for case: " + spec.key);
    }

    std::vector<AnalyticResult> analytics;
    analytics.reserve(ensemble_size);
    for (const auto& g : spec.convergence_graphs) {
        const auto P = build_transition_matrix(g, k);
        validate_transition_matrix(P);
        analytics.push_back(compute_analytic_extinction_times(g, P));
    }

    const auto reps_list = build_log_reps_schedule(3.0, 0.1, 1000, 2000000000LL);
    std::vector<ConvergenceRow> rows;
    rows.reserve(reps_list.size());

    const auto t0 = std::chrono::steady_clock::now();
    for (long long reps_i : reps_list) {
        std::vector<NumericResult> numerics;
        numerics.reserve(ensemble_size);

        for (int gi = 0; gi < ensemble_size; ++gi) {
            uint64_t run_seed = base_seed
                ^ (0x9E3779B97F4A7C15ULL * static_cast<uint64_t>(reps_i + 17 * (gi + 1)))
                ^ (0xBF58476D1CE4E5B9ULL * static_cast<uint64_t>(gi + 1));
            numerics.push_back(run_monte_carlo(spec.convergence_graphs[gi], k, reps_i, run_seed));
        }

        const auto now = std::chrono::steady_clock::now();
        const double elapsed_after = std::chrono::duration<double>(now - t0).count();
        rows.push_back(build_smoothed_convergence_row(spec, analytics, numerics, reps_i, elapsed_after));

        if (elapsed_after >= time_budget_seconds) {
            break;
        }
    }
    return rows;
}

static SlopeInfo fit_loglog_slope(const std::vector<ConvergenceRow>& rows, bool conditional) {
    std::vector<double> xs;
    std::vector<double> ys;
    for (const auto& row : rows) {
        const double y = conditional ? row.pct_error_conditional : row.pct_error_unconditional;
        if (row.reps > 0 && y > 0.0) {
            xs.push_back(std::log10(static_cast<double>(row.reps)));
            ys.push_back(std::log10(y));
        }
    }

    SlopeInfo out;
    out.n_used = static_cast<int>(xs.size());
    if (xs.size() < 2) return out;

    const double mean_x = std::accumulate(xs.begin(), xs.end(), 0.0) / xs.size();
    const double mean_y = std::accumulate(ys.begin(), ys.end(), 0.0) / ys.size();

    double sxx = 0.0;
    double sxy = 0.0;
    for (size_t i = 0; i < xs.size(); ++i) {
        const double dx = xs[i] - mean_x;
        const double dy = ys[i] - mean_y;
        sxx += dx * dx;
        sxy += dx * dy;
    }

    if (sxx <= 0.0) return out;

    out.ok = true;
    out.slope = sxy / sxx;
    out.intercept = mean_y - out.slope * mean_x;
    return out;
}

static std::vector<long long> build_log_reps_schedule(double exp_start,
                                                      double exp_step,
                                                      long long min_reps,
                                                      long long max_reps_cap) {
    std::vector<long long> reps_list;
    double exponent = exp_start;
    while (true) {
        long long reps = static_cast<long long>(std::llround(std::pow(10.0, exponent)));
        if (reps < min_reps) reps = min_reps;
        if (reps > max_reps_cap) break;
        if (reps_list.empty() || reps != reps_list.back()) {
            reps_list.push_back(reps);
        }
        exponent += exp_step;
    }
    return reps_list;
}

static void print_convergence_table(const std::vector<ConvergenceRow>& rows, const std::string& case_key) {
    if (rows.empty()) return;

    std::cout << "Convergence sweep (smoothed over fixed ensemble) for " << case_key << "\n";
    std::cout << "---------------------------------------------------\n";
    std::cout << "Each x-axis point uses the same " << rows.front().ensemble_size
              << " fixed graph/replicate realizations.\n";
    std::cout << std::fixed << std::setprecision(6);
    for (const auto& row : rows) {
        std::cout << "reps=" << std::setw(8) << row.reps
                  << "  uncond %err=" << std::setw(10) << row.pct_error_unconditional
                  << " ± " << std::setw(9) << row.std_pct_error_unconditional
                  << "  cond %err=" << std::setw(10) << row.pct_error_conditional
                  << " ± " << std::setw(9) << row.std_pct_error_conditional
                  << "  avg accepted=" << std::setw(10) << row.avg_accepted_conditional_runs
                  << "  elapsed=" << row.elapsed_seconds << " s\n";
    }

    const SlopeInfo s_u = fit_loglog_slope(rows, false);
    const SlopeInfo s_c = fit_loglog_slope(rows, true);
    std::cout << "\nSmoothed log-log slope estimates (percent error vs reps)\n";
    std::cout << "--------------------------------------------------------\n";
    if (s_u.ok) {
        std::cout << "unconditional slope : " << s_u.slope << " (using " << s_u.n_used << " points)\n";
    } else {
        std::cout << "unconditional slope : not enough positive-error points\n";
    }
    if (s_c.ok) {
        std::cout << "conditional slope   : " << s_c.slope << " (using " << s_c.n_used << " points)\n";
    } else {
        std::cout << "conditional slope   : not enough positive-error points\n";
    }
    std::cout << "Reference slope     : about -0.5 for Monte Carlo error\n\n";
}

static void write_convergence_csv(const std::string& path,
                                  const std::vector<ConvergenceRow>& rows) {
    std::ofstream fout(path);
    if (!fout) {
        throw std::runtime_error("Cannot open convergence CSV for writing: " + path);
    }

    fout << "case_key,case_label,reps,log10_reps,ensemble_size,analytic_unconditional,numeric_unconditional,abs_error_unconditional,pct_error_unconditional,std_pct_error_unconditional,";
    fout << "analytic_conditional,numeric_conditional,abs_error_conditional,pct_error_conditional,std_pct_error_conditional,avg_accepted_conditional_runs,elapsed_seconds\n";
    fout << std::fixed << std::setprecision(10);

    for (const auto& row : rows) {
        fout << row.case_key << ",";
        fout << '"' << row.case_label << '"' << ",";
        fout << row.reps << ",";
        fout << row.log10_reps << ",";
        fout << row.ensemble_size << ",";
        fout << row.analytic_unconditional << ",";
        fout << row.numeric_unconditional << ",";
        fout << row.abs_error_unconditional << ",";
        fout << row.pct_error_unconditional << ",";
        fout << row.std_pct_error_unconditional << ",";
        fout << row.analytic_conditional << ",";
        fout << row.numeric_conditional << ",";
        fout << row.abs_error_conditional << ",";
        fout << row.pct_error_conditional << ",";
        fout << row.std_pct_error_conditional << ",";
        fout << row.avg_accepted_conditional_runs << ",";
        fout << row.elapsed_seconds << "\n";
    }
}

static void run_case(const CaseSpec& spec,
                     double k,
                     long long reps,
                     uint64_t base_seed,
                     double convergence_time_budget_seconds,
                     std::vector<ConvergenceRow>* all_rows) {
    std::cout << "=============================================================\n";
    std::cout << spec.label << "\n";
    std::cout << "=============================================================\n";
    std::cout << spec.parameter_text << "\n\n";
    std::cout << "Interpretation of extinction: stop when at least one species count becomes zero.\n";
    std::cout << "This matches run_until_extinction() in your current simulation code.\n\n";

    print_graph_summary(spec.graph);
    print_graph_connections(spec.graph);
    const auto P = build_transition_matrix(spec.graph, k);
    validate_transition_matrix(P);
    const AnalyticResult analytic = compute_analytic_extinction_times(spec.graph, P);
    const NumericResult numeric = run_monte_carlo(spec.graph, k, reps, base_seed);

    const double analytic_mcs_uncond = analytic.mean_attempts_unconditional / static_cast<double>(spec.graph.N);
    const double analytic_mcs_cond = analytic.mean_attempts_conditional / static_cast<double>(spec.graph.N);
    const double numeric_mcs_uncond = numeric.mean_attempts_unconditional / static_cast<double>(spec.graph.N);
    const double numeric_mcs_cond = numeric.mean_attempts_conditional / static_cast<double>(spec.graph.N);

    std::cout << "State counting\n";
    std::cout << "-------------\n";
    std::cout << "total states                  : " << analytic.total_states << "\n";
    std::cout << "transient states (all 3 alive): " << analytic.transient_count << "\n";
    std::cout << "extinction states             : " << analytic.extinct_count << "\n\n";

    std::cout << std::fixed << std::setprecision(8);
    std::cout << "Extinction time comparison\n";
    std::cout << "--------------------------\n";
    std::cout << "Parameter k                  : " << k << "\n";
    std::cout << "Monte Carlo reps             : " << reps << "\n";
    std::cout << "Base seed                    : " << base_seed << "\n\n";

    std::cout << "A) Unconditional average over init_random_uniform()\n";
    std::cout << "   analytic mean attempts    : " << analytic.mean_attempts_unconditional << "\n";
    std::cout << "   numeric  mean attempts    : " << numeric.mean_attempts_unconditional << "\n";
    std::cout << "   absolute difference       : "
              << std::fabs(analytic.mean_attempts_unconditional - numeric.mean_attempts_unconditional) << "\n";
    std::cout << "   percent error             : "
              << percent_error(analytic.mean_attempts_unconditional, numeric.mean_attempts_unconditional) << "%\n";
    std::cout << "   analytic mean MCS         : " << analytic_mcs_uncond << "\n";
    std::cout << "   numeric  mean MCS         : " << numeric_mcs_uncond << "\n\n";

    std::cout << "B) Conditional average given all three species are present initially\n";
    std::cout << "   analytic mean attempts    : " << analytic.mean_attempts_conditional << "\n";
    std::cout << "   numeric  mean attempts    : " << numeric.mean_attempts_conditional << "\n";
    std::cout << "   absolute difference       : "
              << std::fabs(analytic.mean_attempts_conditional - numeric.mean_attempts_conditional) << "\n";
    std::cout << "   percent error             : "
              << percent_error(analytic.mean_attempts_conditional, numeric.mean_attempts_conditional) << "%\n";
    std::cout << "   analytic mean MCS         : " << analytic_mcs_cond << "\n";
    std::cout << "   numeric  mean MCS         : " << numeric_mcs_cond << "\n";
    std::cout << "   accepted conditional runs : " << numeric.accepted_conditional_runs << " / " << reps << "\n\n";

    print_top_transient_examples(spec.graph, analytic);

    if (all_rows != nullptr && convergence_time_budget_seconds > 0.0) {
        std::vector<ConvergenceRow> local_rows = run_convergence_log_sweep(
            spec, k, base_seed ^ 0x0F0F0F0F0F0F0F0FULL, convergence_time_budget_seconds);
        for (const auto& row : local_rows) {
            all_rows->push_back(row);
        }
        print_convergence_table(local_rows, spec.key);
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        double k = 1.0;
        long long reps = 200000;
        uint64_t graph_seed = 12345ULL;
        uint64_t sim_seed = 67890ULL;
        std::string convergence_csv = "convergence_errors.csv";
        double total_convergence_budget_seconds = 30.0;

        if (argc >= 2) k = std::stod(argv[1]);
        if (argc >= 3) reps = std::stoll(argv[2]);
        if (argc >= 4) graph_seed = static_cast<uint64_t>(std::stoull(argv[3]));
        if (argc >= 5) sim_seed = static_cast<uint64_t>(std::stoull(argv[4]));
        if (argc >= 6) convergence_csv = argv[5];
        if (argc >= 7) total_convergence_budget_seconds = std::stod(argv[6]);
        if (argc > 7) {
            throw std::runtime_error("Usage: test_analytically.exe [k] [reps] [graph_seed] [sim_seed] [convergence_csv] [convergence_seconds]");
        }

        if (k < 0.0 || k > 1.0) {
            throw std::runtime_error("This test expects 0 <= k <= 1.");
        }
        if (reps <= 0) {
            throw std::runtime_error("reps must be positive.");
        }
        if (total_convergence_budget_seconds <= 0.0) {
            throw std::runtime_error("convergence_seconds must be positive.");
        }

        RNG rng_sw(graph_seed ^ 0x1111111111111111ULL);
        RNG rng_ba(graph_seed ^ 0x2222222222222222ULL);

        std::vector<Graph> lattice_conv_graphs(10, build_lattice2D(2));

        std::vector<Graph> smallworld_conv_graphs;
        smallworld_conv_graphs.reserve(10);
        for (int i = 0; i < 10; ++i) {
            RNG rg(graph_seed ^ 0x1111111111111111ULL ^ (0x9E3779B97F4A7C15ULL * static_cast<uint64_t>(i + 1)));
            smallworld_conv_graphs.push_back(build_watts_strogatz(6, 2, 0.5, rg));
        }

        std::vector<Graph> scalefree_conv_graphs;
        scalefree_conv_graphs.reserve(10);
        for (int i = 0; i < 10; ++i) {
            RNG rg(graph_seed ^ 0x2222222222222222ULL ^ (0xBF58476D1CE4E5B9ULL * static_cast<uint64_t>(i + 1)));
            scalefree_conv_graphs.push_back(build_barabasi_albert(6, 3, 2, rg));
        }

        const CaseSpec lattice_case{
            "lattice2D",
            "1) 2D lattice analytic vs numerical extinction-time test",
            build_lattice2D(2),
            "Topology parameters: L=2 (so N=4, total state space = 3^4 = 81).",
            lattice_conv_graphs
        };

        const CaseSpec smallworld_case{
            "smallworld",
            "2) Small-world analytic vs numerical extinction-time test",
            build_watts_strogatz(6, 2, 0.5, rng_sw),
            "Topology parameters: N=6, K=2, beta=0.5 (one concrete graph instance generated from graph_seed).",
            smallworld_conv_graphs
        };

        const CaseSpec scalefree_case{
            "scalefree",
            "3) Scale-free analytic vs numerical extinction-time test",
            build_barabasi_albert(6, 3, 2, rng_ba),
            "Topology parameters: N=6, m0=3, m=2 (one concrete graph instance generated from graph_seed).",
            scalefree_conv_graphs
        };

        const double per_case_budget = total_convergence_budget_seconds / 3.0;

        std::cout << "Analytic vs numerical extinction-time tests on small graphs\n";
        std::cout << "=========================================================\n";
        std::cout << "This program runs three exact Markov-chain benchmarks:\n";
        std::cout << "  1. 2x2 lattice\n";
        std::cout << "  2. a small small-world graph\n";
        std::cout << "  3. a small scale-free graph\n\n";
        std::cout << "For the random-graph cases, the analytic result is for the exact graph instance\n";
        std::cout << "generated by graph_seed, and the Monte Carlo comparison uses that same graph.\n\n";
        std::cout << "The convergence test now uses log-spaced sample sizes: reps ~ 10^(3 + 0.1 i).\n";
        std::cout << "Percent error is intended for log-log plotting, and the program also estimates\n";
        std::cout << "the slope of log10(percent error) vs log10(reps).\n";
        std::cout << "Total convergence time budget  : " << total_convergence_budget_seconds << " seconds\n";
        std::cout << "Per-case convergence budget    : " << per_case_budget << " seconds\n";
        std::cout << "Reference Monte Carlo slope    : about -0.5\n\n";

        std::vector<ConvergenceRow> all_rows;
        run_case(lattice_case, k, reps, sim_seed ^ 0xABCDEF0000000001ULL, per_case_budget, &all_rows);
        run_case(smallworld_case, k, reps, sim_seed ^ 0xABCDEF0000000002ULL, per_case_budget, &all_rows);
        run_case(scalefree_case, k, reps, sim_seed ^ 0xABCDEF0000000003ULL, per_case_budget, &all_rows);

        write_convergence_csv(convergence_csv, all_rows);
        std::cout << "Convergence CSV written to: " << convergence_csv << "\n\n";

        std::cout << "Suggested compile command\n";
        std::cout << "-------------------------\n";
        std::cout << "g++ -O2 -std=c++17 src/test_analytically.cpp src/graph_builders.cpp src/rps_sim.cpp -o test_analytically.exe\n\n";

        std::cout << "Suggested run command\n";
        std::cout << "---------------------\n";
        std::cout << ".\\test_analytically.exe 1.0 200000 12345 67890 convergence_errors.csv 30\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
