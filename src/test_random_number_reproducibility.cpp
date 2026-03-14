#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <cstdint>
#include <algorithm>

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

struct RunSnapshot {
    std::vector<uint8_t> init_state;
    std::vector<uint8_t> final_state;
    int init_nR = 0, init_nP = 0, init_nS = 0;
    int final_nR = 0, final_nP = 0, final_nS = 0;
    SimResult result;
};

static uint64_t rep_seed(uint64_t base_seed, int rep) {
    return (base_seed ^ 0xA5A5A5A5A5A5A5A5ULL) + (uint64_t)rep * 1315423911ULL;
}

static std::string vec_signature(const std::vector<uint8_t>& v) {
    std::ostringstream oss;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) oss << ',';
        oss << (int)v[i];
    }
    return oss.str();
}

static std::string sim_signature(const RunSnapshot& s) {
    std::ostringstream oss;
    oss << "init=[" << vec_signature(s.init_state) << "]"
        << " final=[" << vec_signature(s.final_state) << "]"
        << " init_counts=(" << s.init_nR << "," << s.init_nP << "," << s.init_nS << ")"
        << " final_counts=(" << s.final_nR << "," << s.final_nP << "," << s.final_nS << ")"
        << " Text_attempts=" << s.result.Text_attempts
        << " Text_mcs=" << std::fixed << std::setprecision(6) << s.result.Text_mcs
        << " extinct=" << s.result.extinct;
    return oss.str();
}

static RunSnapshot run_once(const Graph& g, uint64_t seed, double k, long long max_attempts) {
    RPS_Sim sim(g, seed, k);
    sim.init_random_uniform();

    RunSnapshot snap;
    snap.init_state = sim.state;
    snap.init_nR = sim.nR;
    snap.init_nP = sim.nP;
    snap.init_nS = sim.nS;

    snap.result = sim.run_until_extinction(max_attempts);

    snap.final_state = sim.state;
    snap.final_nR = sim.nR;
    snap.final_nP = sim.nP;
    snap.final_nS = sim.nS;
    return snap;
}

static bool same_snapshot(const RunSnapshot& a, const RunSnapshot& b) {
    return a.init_state == b.init_state &&
           a.final_state == b.final_state &&
           a.init_nR == b.init_nR && a.init_nP == b.init_nP && a.init_nS == b.init_nS &&
           a.final_nR == b.final_nR && a.final_nP == b.final_nP && a.final_nS == b.final_nS &&
           a.result.Text_attempts == b.result.Text_attempts &&
           std::fabs(a.result.Text_mcs - b.result.Text_mcs) < 1e-12 &&
           a.result.extinct == b.result.extinct;
}

static TestResult test_same_seed_same_simulation() {
    Graph g = build_lattice2D(3);
    const uint64_t seed = 42;
    const double k = 1.0;
    const long long max_attempts = 200;

    RunSnapshot a = run_once(g, seed, k, max_attempts);
    RunSnapshot b = run_once(g, seed, k, max_attempts);

    const bool ok = same_snapshot(a, b);
    std::string detail = ok
        ? ("Same seed reproduced identical initialization and trajectory | " + sim_signature(a))
        : ("Mismatch detected\nA: " + sim_signature(a) + "\nB: " + sim_signature(b));

    return {"Test 1 - Same seed reproducibility", ok, detail};
}

static TestResult test_different_seed_changes_simulation() {
    Graph g = build_lattice2D(3);
    const double k = 1.0;
    const long long max_attempts = 200;
    const std::vector<uint64_t> seeds = {101, 102, 103, 104, 105};

    std::vector<std::string> signatures;
    signatures.reserve(seeds.size());
    for (uint64_t s : seeds) {
        signatures.push_back(sim_signature(run_once(g, s, k, max_attempts)));
    }

    bool all_same = true;
    for (size_t i = 1; i < signatures.size(); ++i) {
        if (signatures[i] != signatures[0]) {
            all_same = false;
            break;
        }
    }

    std::ostringstream oss;
    for (size_t i = 0; i < seeds.size(); ++i) {
        oss << "seed " << seeds[i] << " -> " << signatures[i];
        if (i + 1 < seeds.size()) oss << " | ";
    }

    return {
        "Test 2 - Different seed independence",
        !all_same,
        !all_same ? ("At least one realization differs as expected | " + oss.str())
                  : ("All tested seeds produced identical signatures (unexpected) | " + oss.str())
    };
}

static TestResult test_rep_seed_logic() {
    const uint64_t base_seed = 123456789ULL;
    std::vector<uint64_t> seeds_a;
    std::vector<uint64_t> seeds_b;

    for (int rep = 0; rep < 10; ++rep) {
        seeds_a.push_back(rep_seed(base_seed, rep));
        seeds_b.push_back(rep_seed(base_seed, rep));
    }

    bool reproducible = (seeds_a == seeds_b);

    std::vector<uint64_t> sorted = seeds_a;
    std::sort(sorted.begin(), sorted.end());
    bool unique = std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end();

    std::ostringstream oss;
    oss << "base_seed=" << base_seed << " -> ";
    for (size_t i = 0; i < seeds_a.size(); ++i) {
        if (i) oss << ", ";
        oss << "rep " << i << ": " << seeds_a[i];
    }

    return {
        "Test 3 - Rep seed logic",
        reproducible && unique,
        (reproducible && unique)
            ? ("Rep-derived seeds are deterministic and unique | " + oss.str())
            : ("Rep-derived seeds failed reproducibility or uniqueness | " + oss.str())
    };
}

static TestResult test_rng_distribution_sanity() {
    const uint64_t seed = 987654321ULL;
    const int n = 200000;
    const int bins = 10;

    RNG rng(seed);
    double sum = 0.0;
    double sumsq = 0.0;
    std::vector<int> hist(bins, 0);

    for (int i = 0; i < n; ++i) {
        double x = rng.u01();
        sum += x;
        sumsq += x * x;

        int b = (int)(x * bins);
        if (b == bins) b = bins - 1;
        hist[b]++;
    }

    const double mean = sum / (double)n;
    const double var = sumsq / (double)n - mean * mean;
    const double expected_mean = 0.5;
    const double expected_var = 1.0 / 12.0;
    const double expected_bin = (double)n / (double)bins;

    const bool mean_ok = std::fabs(mean - expected_mean) < 0.01;
    const bool var_ok = std::fabs(var - expected_var) < 0.01;

    bool bins_ok = true;
    for (int c : hist) {
        if (std::fabs((double)c - expected_bin) > 0.05 * expected_bin) {
            bins_ok = false;
            break;
        }
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6)
        << "mean=" << mean << " (expected ~0.5), "
        << "var=" << var << " (expected ~0.083333), bins=[";
    for (int i = 0; i < bins; ++i) {
        if (i) oss << ',';
        oss << hist[i];
    }
    oss << "]";

    return {
        "Test 4 - RNG distribution sanity",
        mean_ok && var_ok && bins_ok,
        (mean_ok && var_ok && bins_ok)
            ? ("Uniformity checks look reasonable | " + oss.str())
            : ("Distribution looks suspicious | " + oss.str())
    };
}

static bool print_results(const std::vector<TestResult>& results) {
    bool all_ok = true;
    std::cout << "Random-number reproducibility test results\n";
    std::cout << "-----------------------------------------\n";
    for (const auto& r : results) {
        std::cout << (r.passed ? "[PASS] " : "[FAIL] ")
                  << r.name << " -> " << r.detail << "\n";
        all_ok = all_ok && r.passed;
    }
    return all_ok;
}

int main() {
    try {
        std::vector<TestResult> results;
        results.push_back(test_same_seed_same_simulation());
        results.push_back(test_different_seed_changes_simulation());
        results.push_back(test_rep_seed_logic());
        results.push_back(test_rng_distribution_sanity());

        bool ok = print_results(results);
        std::cout << "\nOverall: " << (ok ? "PASS" : "FAIL") << "\n";
        return ok ? 0 : 2;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
