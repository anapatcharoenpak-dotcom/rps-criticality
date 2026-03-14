#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <iomanip>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <set>

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
    return "NA"; // includes s == -1 (censored)
}

struct CsvRow {
    std::vector<std::string> cells;
};

static std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) out.push_back(cell);
    if (!line.empty() && line.back() == ',') out.push_back("");
    return out;
}

static std::vector<CsvRow> read_csv(const std::string& path) {
    std::ifstream fin(path);
    if (!fin) throw std::runtime_error("Cannot open CSV file: " + path);

    std::vector<CsvRow> rows;
    std::string line;
    while (std::getline(fin, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        rows.push_back({split_csv_line(line)});
    }
    return rows;
}

static std::string join_cells(const std::vector<std::string>& cells) {
    std::string s;
    for (size_t i = 0; i < cells.size(); ++i) {
        if (i) s += ",";
        s += cells[i];
    }
    return s;
}

static long long to_ll(const std::string& s) { return std::stoll(s); }
static int to_int(const std::string& s) { return std::stoi(s); }
static double to_double(const std::string& s) { return std::stod(s); }

static bool approx_equal(double a, double b, double eps = 1e-9) {
    return std::fabs(a - b) <= eps;
}

static std::string expected_header() {
    return "graph,L,N,K,beta,k,rep,seed,max_mcs,censored,Text_attempts,Text_mcs,extinct";
}

static void remove_if_exists(const std::string& path) {
    std::remove(path.c_str());
}

static void write_csv_runs(const Graph& g,
                           int outL,
                           int K,
                           double used_beta,
                           double k,
                           int reps,
                           uint64_t base_seed,
                           long long MAX_MCS,
                           const std::string& out_csv) {
    const long long max_attempts = (MAX_MCS > 0) ? MAX_MCS * (long long)g.N : -1;
    const int N = g.N;

    bool file_exists = false;
    {
        std::ifstream fin(out_csv);
        file_exists = fin.good();
    }

    std::ofstream fout(out_csv, std::ios::app);
    if (!fout) throw std::runtime_error("Cannot open output file: " + out_csv);

    fout.setf(std::ios::fixed);
    fout << std::setprecision(6);

    if (!file_exists) {
        fout << expected_header() << "\n";
    }

    for (int rep = 0; rep < reps; ++rep) {
        const uint64_t seed =
            (base_seed ^ 0xA5A5A5A5A5A5A5A5ULL) + (uint64_t)rep * 1315423911ULL;

        RPS_Sim sim(g, seed, k);
        sim.init_random_uniform();

        SimResult res = sim.run_until_extinction(max_attempts);
        const int censored = (res.extinct == -1) ? 1 : 0;

        fout << g.name << ","
             << outL << ","
             << N << ","
             << K << ","
             << used_beta << ","
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
}

// Test 1: header correctness
static TestResult test_header_correctness(const std::string& path) {
    auto rows = read_csv(path);
    if (rows.empty()) {
        return {"Test 1 - Header correctness", false, "CSV file is empty"};
    }

    const std::string got = join_cells(rows[0].cells);
    const bool ok = (got == expected_header());
    return {
        "Test 1 - Header correctness",
        ok,
        ok ? "Header matches expected schema"
           : ("Expected: " + expected_header() + " | Got: " + got)
    };
}

// Test 2: header appears exactly once even after append
static TestResult test_header_written_once(const std::string& path) {
    auto rows = read_csv(path);
    int count = 0;
    for (const auto& row : rows) {
        if (join_cells(row.cells) == expected_header()) count++;
    }

    const bool ok = (count == 1);
    return {
        "Test 2 - Header written once",
        ok,
        ok ? "Header appears exactly once"
           : ("Header appears " + std::to_string(count) + " times")
    };
}

// Test 3: every row has 13 columns
static TestResult test_column_count_consistency(const std::string& path) {
    auto rows = read_csv(path);
    for (size_t i = 0; i < rows.size(); ++i) {
        if (rows[i].cells.size() != 13) {
            return {
                "Test 3 - Column count consistency",
                false,
                "Row " + std::to_string(i) + " has " + std::to_string(rows[i].cells.size()) +
                " columns (expected 13)"
            };
        }
    }
    return {"Test 3 - Column count consistency", true, "All rows have 13 columns"};
}

// Test 4: parameter metadata are recorded correctly
static TestResult test_parameter_consistency(const std::string& path,
                                             const std::string& graph_name,
                                             int outL,
                                             int N,
                                             int K,
                                             double beta,
                                             double k,
                                             const std::vector<int>& expected_reps,
                                             uint64_t base_seed,
                                             long long MAX_MCS) {
    auto rows = read_csv(path);
    if ((int)rows.size() != (int)expected_reps.size() + 1) {
        return {
            "Test 4 - Parameter consistency",
            false,
            "Expected " + std::to_string((int)expected_reps.size()) + " data rows, found " +
            std::to_string((int)rows.size() - 1)
        };
    }

    for (size_t row_idx = 0; row_idx < expected_reps.size(); ++row_idx) {
        const auto& c = rows[row_idx + 1].cells;
        const int rep = expected_reps[row_idx];

        const uint64_t expected_seed =
            (base_seed ^ 0xA5A5A5A5A5A5A5A5ULL) + (uint64_t)rep * 1315423911ULL;

        if (c[0] != graph_name) {
            return {"Test 4 - Parameter consistency", false,
                    "Row " + std::to_string(row_idx + 1) + " graph mismatch"};
        }
        if (to_int(c[1]) != outL || to_int(c[2]) != N || to_int(c[3]) != K) {
            return {"Test 4 - Parameter consistency", false,
                    "Row " + std::to_string(row_idx + 1) + " L/N/K mismatch"};
        }
        if (!approx_equal(to_double(c[4]), beta) || !approx_equal(to_double(c[5]), k)) {
            return {"Test 4 - Parameter consistency", false,
                    "Row " + std::to_string(row_idx + 1) + " beta/k mismatch"};
        }
        if (to_int(c[6]) != rep) {
            return {"Test 4 - Parameter consistency", false,
                    "Row " + std::to_string(row_idx + 1) + " rep mismatch"};
        }
        if ((uint64_t)std::stoull(c[7]) != expected_seed) {
            return {"Test 4 - Parameter consistency", false,
                    "Row " + std::to_string(row_idx + 1) + " seed mismatch"};
        }
        if (to_ll(c[8]) != MAX_MCS) {
            return {"Test 4 - Parameter consistency", false,
                    "Row " + std::to_string(row_idx + 1) + " max_mcs mismatch"};
        }
    }

    return {"Test 4 - Parameter consistency", true, "Metadata columns match expected values"};
}

// Test 5: time bookkeeping must satisfy Text_mcs = Text_attempts / N
static TestResult test_time_bookkeeping(const std::string& path, int N) {
    auto rows = read_csv(path);
    for (size_t i = 1; i < rows.size(); ++i) {
        const auto& c = rows[i].cells;
        const long long attempts = to_ll(c[10]);
        const double mcs = to_double(c[11]);
        const double expected = (double)attempts / (double)N;

        if (attempts < 0 || mcs < 0.0) {
            return {"Test 5 - Time bookkeeping", false,
                    "Negative time value at row " + std::to_string(i)};
        }
        if (!approx_equal(mcs, expected, 1e-6)) {
            return {"Test 5 - Time bookkeeping", false,
                    "Row " + std::to_string(i) + " has Text_mcs=" + c[11] +
                    " but expected " + std::to_string(expected)};
        }
    }
    return {"Test 5 - Time bookkeeping", true, "Text_mcs matches Text_attempts / N for all rows"};
}

// Test 6: censored/extinct values must be valid and consistent
static TestResult test_censor_and_species_labels(const std::string& path,
                                                 bool expect_all_censored,
                                                 bool expect_all_not_censored) {
    auto rows = read_csv(path);
    const std::set<std::string> valid_species = {"R", "P", "S", "NA"};

    for (size_t i = 1; i < rows.size(); ++i) {
        const auto& c = rows[i].cells;
        const int censored = to_int(c[9]);
        const std::string extinct = c[12];

        if (!valid_species.count(extinct)) {
            return {"Test 6 - Censored/extinct consistency", false,
                    "Invalid extinct label at row " + std::to_string(i) + ": " + extinct};
        }

        if (censored == 1 && extinct != "NA") {
            return {"Test 6 - Censored/extinct consistency", false,
                    "Row " + std::to_string(i) + " is censored but extinct != NA"};
        }
        if (censored == 0 && extinct == "NA") {
            return {"Test 6 - Censored/extinct consistency", false,
                    "Row " + std::to_string(i) + " is not censored but extinct = NA"};
        }

        if (expect_all_censored && censored != 1) {
            return {"Test 6 - Censored/extinct consistency", false,
                    "Expected all rows censored, but row " + std::to_string(i) + " is not"};
        }
        if (expect_all_not_censored && censored != 0) {
            return {"Test 6 - Censored/extinct consistency", false,
                    "Expected all rows non-censored, but row " + std::to_string(i) + " is censored"};
        }
    }

    return {"Test 6 - Censored/extinct consistency", true,
            expect_all_censored ? "All rows correctly marked as censored with extinct=NA"
            : expect_all_not_censored ? "All rows correctly marked as extinction events"
            : "Censored and extinct columns are valid"};
}

static bool run_and_print(const std::vector<TestResult>& results) {
    bool all_ok = true;
    std::cout << "CSV output sanity results\n";
    std::cout << "-------------------------\n";
    for (const auto& r : results) {
        std::cout << (r.passed ? "[PASS] " : "[FAIL] ")
                  << r.name << " -> " << r.detail << "\n";
        all_ok = all_ok && r.passed;
    }
    return all_ok;
}

int main() {
    try {
        const std::string normal_csv = "csv_output_sanity_normal.csv";
        const std::string censored_csv = "csv_output_sanity_censored.csv";
        remove_if_exists(normal_csv);
        remove_if_exists(censored_csv);

        // Use the same graph family and metadata conventions as main.cpp.
        const int L = 3;
        Graph g = build_lattice2D(L);
        const int outL = L;
        const int N = g.N;
        const int K = 4;
        const double beta = 0.0;

        // File A: normal extinction runs, written twice to test append-without-duplicate-header.
        const double k_normal = 1.0;
        const int reps_first = 3;
        const int reps_second = 2;
        const uint64_t base_seed_normal = 12345ULL;
        const long long max_mcs_normal = 5000;

        write_csv_runs(g, outL, K, beta, k_normal, reps_first,  base_seed_normal, max_mcs_normal, normal_csv);
        write_csv_runs(g, outL, K, beta, k_normal, reps_second, base_seed_normal, max_mcs_normal, normal_csv);

        // File B: guaranteed censored runs with k=0 and tiny cap.
        const double k_censored = 0.0;
        const int reps_censored = 3;
        const uint64_t base_seed_censored = 98765ULL;
        const long long max_mcs_censored = 2;

        write_csv_runs(g, outL, K, beta, k_censored, reps_censored, base_seed_censored, max_mcs_censored, censored_csv);

        std::vector<TestResult> results;
        results.push_back(test_header_correctness(normal_csv));
        results.push_back(test_header_written_once(normal_csv));
        results.push_back(test_column_count_consistency(normal_csv));
        results.push_back(test_parameter_consistency(normal_csv, g.name, outL, N, K, beta,
                                                     k_normal, {0, 1, 2, 0, 1},
                                                     base_seed_normal, max_mcs_normal));
        results.push_back(test_time_bookkeeping(normal_csv, N));
        results.push_back(test_censor_and_species_labels(censored_csv, true, false));

        bool ok = run_and_print(results);

        std::cout << "\nFiles generated during test:\n";
        std::cout << "  " << normal_csv << "\n";
        std::cout << "  " << censored_csv << "\n";
        std::cout << "\nOverall: " << (ok ? "PASS" : "FAIL") << "\n";
        return ok ? 0 : 2;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
