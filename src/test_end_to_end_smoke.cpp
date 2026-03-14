#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>
#include <stdexcept>
#include <cstdint>
#include <cstdio>

#include "graph.hpp"
#include "rng.hpp"
#include "rps_sim.hpp"

// Forward declarations from graph_builders.cpp
Graph build_lattice2D(int L);
Graph build_watts_strogatz(int N, int K, double beta, RNG& rng);
Graph build_barabasi_albert(int N, int m0, int m, RNG& rng);

struct TestResult {
    std::string name;
    bool passed;
    std::string detail;
};

static inline const char* species_name(int s) {
    if (s == (int)ROCK) return "R";
    if (s == (int)PAPER) return "P";
    if (s == (int)SCISSORS) return "S";
    return "NA";
}

static int count_csv_columns(const std::string& line) {
    if (line.empty()) return 0;
    int commas = 0;
    for (char c : line) if (c == ',') commas++;
    return commas + 1;
}

static bool is_valid_extinct_label(const std::string& s) {
    return s == "R" || s == "P" || s == "S" || s == "NA";
}

static std::vector<std::string> split_csv_simple(const std::string& line) {
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, ',')) out.push_back(item);
    return out;
}

static bool counts_match_state(const RPS_Sim& sim) {
    int cR = 0, cP = 0, cS = 0;
    for (uint8_t x : sim.state) {
        if (x == (uint8_t)ROCK) cR++;
        else if (x == (uint8_t)PAPER) cP++;
        else if (x == (uint8_t)SCISSORS) cS++;
    }
    return cR == sim.nR && cP == sim.nP && cS == sim.nS && (cR + cP + cS == sim.g.N);
}

static TestResult run_single_case(const std::string& graph_type,
                                  int size_param,
                                  int degree_param,
                                  double beta,
                                  double k,
                                  uint64_t base_seed,
                                  long long max_mcs) {
    try {
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
            const int m = degree_param;
            const int m0 = 2 * m;
            K = m;
            used_beta = 0.0;
            RNG rng_graph(base_seed ^ 0xD1B54A32D192ED03ULL);
            g = build_barabasi_albert(N, m0, m, rng_graph);
        } else {
            return {"Unknown case", false, "Unknown graph type: " + graph_type};
        }

        const long long max_attempts = (max_mcs > 0) ? max_mcs * (long long)g.N : -1;
        const uint64_t seed = (base_seed ^ 0xA5A5A5A5A5A5A5A5ULL);

        RPS_Sim sim(g, seed, k);
        sim.init_random_uniform();
        if (!counts_match_state(sim)) {
            return {"Smoke - " + graph_type, false,
                    "Initialization bookkeeping mismatch before run"};
        }

        SimResult res = sim.run_until_extinction(max_attempts);

        if (!counts_match_state(sim)) {
            return {"Smoke - " + graph_type, false,
                    "Population bookkeeping mismatch after run"};
        }

        const bool extinct_now = (sim.nR == 0 || sim.nP == 0 || sim.nS == 0);
        const bool censored = (res.extinct == -1);
        const bool attempt_ok = (res.Text_attempts >= 0);
        const bool mcs_ok = (res.Text_mcs >= 0.0);
        const bool relation_ok = (res.Text_mcs == (double)res.Text_attempts / (double)g.N);

        if (!attempt_ok || !mcs_ok || !relation_ok) {
            return {"Smoke - " + graph_type, false,
                    "Invalid time bookkeeping: attempts=" + std::to_string(res.Text_attempts) +
                    ", mcs=" + std::to_string(res.Text_mcs)};
        }

        if (!censored && !extinct_now) {
            return {"Smoke - " + graph_type, false,
                    "Reported extinction, but final counts still have all three species"};
        }

        if (censored && max_attempts > 0 && res.Text_attempts != max_attempts) {
            return {"Smoke - " + graph_type, false,
                    "Censored run should stop exactly at cap"};
        }

        std::string detail =
            "N=" + std::to_string(g.N) +
            ", K=" + std::to_string(K) +
            ", beta=" + std::to_string(used_beta) +
            ", attempts=" + std::to_string(res.Text_attempts) +
            ", mcs=" + std::to_string(res.Text_mcs) +
            ", extinct=" + species_name(res.extinct);

        return {"Smoke - " + graph_type, true, detail};

    } catch (const std::exception& e) {
        return {"Smoke - " + graph_type, false, e.what()};
    }
}

static TestResult test_csv_pipeline() {
    const std::string out_csv = "smoke_test_output.csv";
    std::remove(out_csv.c_str());

    try {
        const std::string graph_type = "lattice2D";
        const int L = 3;
        const double k = 1.0;
        const int reps = 2;
        const uint64_t base_seed = 123456789ULL;
        const long long MAX_MCS = 50;

        Graph g = build_lattice2D(L);
        const long long max_attempts = MAX_MCS * (long long)g.N;

        bool file_exists = false;
        {
            std::ifstream fin(out_csv);
            file_exists = fin.good();
        }

        std::ofstream fout(out_csv, std::ios::app);
        if (!fout) {
            return {"Test 4 - CSV pipeline", false, "Cannot open temporary CSV file"};
        }

        fout.setf(std::ios::fixed);
        fout << std::setprecision(6);

        if (!file_exists) {
            fout << "graph,L,N,K,beta,k,rep,seed,max_mcs,censored,Text_attempts,Text_mcs,extinct\n";
        }

        for (int rep = 0; rep < reps; ++rep) {
            const uint64_t seed =
                (base_seed ^ 0xA5A5A5A5A5A5A5A5ULL) + (uint64_t)rep * 1315423911ULL;

            RPS_Sim sim(g, seed, k);
            sim.init_random_uniform();
            SimResult res = sim.run_until_extinction(max_attempts);
            const int censored = (res.extinct == -1) ? 1 : 0;

            fout << g.name << ","
                 << L << ","
                 << g.N << ","
                 << 4 << ","
                 << 0.0 << ","
                 << k << ","
                 << rep << ","
                 << seed << ","
                 << MAX_MCS << ","
                 << censored << ","
                 << res.Text_attempts << ","
                 << res.Text_mcs << ","
                 << species_name(res.extinct)
                 << "\n";
        }
        fout.close();

        std::ifstream fin(out_csv);
        if (!fin) {
            return {"Test 4 - CSV pipeline", false, "Failed to reopen temporary CSV"};
        }

        std::vector<std::string> lines;
        std::string line;
        while (std::getline(fin, line)) {
            if (!line.empty()) lines.push_back(line);
        }

        if ((int)lines.size() != reps + 1) {
            return {"Test 4 - CSV pipeline", false,
                    "Expected " + std::to_string(reps + 1) +
                    " non-empty lines, got " + std::to_string(lines.size())};
        }

        const std::string expected_header =
            "graph,L,N,K,beta,k,rep,seed,max_mcs,censored,Text_attempts,Text_mcs,extinct";
        if (lines[0] != expected_header) {
            return {"Test 4 - CSV pipeline", false, "CSV header mismatch"};
        }

        for (size_t i = 1; i < lines.size(); ++i) {
            if (count_csv_columns(lines[i]) != 13) {
                return {"Test 4 - CSV pipeline", false,
                        "Row " + std::to_string(i) + " does not have 13 columns"};
            }

            auto cols = split_csv_simple(lines[i]);
            if (cols.size() != 13) {
                return {"Test 4 - CSV pipeline", false,
                        "Row split failure at row " + std::to_string(i)};
            }

            if (cols[0] != graph_type) {
                return {"Test 4 - CSV pipeline", false,
                        "Unexpected graph label in CSV row " + std::to_string(i)};
            }

            if (!is_valid_extinct_label(cols[12])) {
                return {"Test 4 - CSV pipeline", false,
                        "Invalid extinction label in row " + std::to_string(i)};
            }
        }

        std::remove(out_csv.c_str());
        return {"Test 4 - CSV pipeline", true, "Header + rows + extinction labels are valid"};

    } catch (const std::exception& e) {
        std::remove(out_csv.c_str());
        return {"Test 4 - CSV pipeline", false, e.what()};
    }
}

int main() {
    std::vector<TestResult> results;

    results.push_back(run_single_case("lattice2D", 3, 0, 0.0, 1.0, 123ULL, 50));
    results.push_back(run_single_case("smallworld", 20, 4, 0.10, 1.0, 456ULL, 50));
    results.push_back(run_single_case("scalefree", 20, 2, 0.0, 1.0, 789ULL, 50));
    results.push_back(test_csv_pipeline());

    bool all_ok = true;

    std::cout << "End-to-end smoke test results\n";
    std::cout << "------------------------------\n";
    for (const auto& r : results) {
        std::cout << (r.passed ? "[PASS] " : "[FAIL] ")
                  << r.name << " -> " << r.detail << "\n";
        all_ok = all_ok && r.passed;
    }

    std::cout << "\nOverall: " << (all_ok ? "PASS" : "FAIL") << "\n";
    return all_ok ? 0 : 2;
}
