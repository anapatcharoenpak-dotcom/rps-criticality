#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <cstdint>

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

static bool approx_equal(double a, double b, double eps = 1e-12) {
    return std::fabs(a - b) < eps;
}

static void set_state_and_counts(RPS_Sim& sim, const std::vector<Species>& cfg) {
    if ((int)cfg.size() != sim.g.N) {
        throw std::runtime_error("Configuration size does not match graph size.");
    }

    sim.state.resize(sim.g.N);
    sim.nR = sim.nP = sim.nS = 0;

    for (int i = 0; i < sim.g.N; ++i) {
        sim.state[i] = static_cast<uint8_t>(cfg[i]);
        if (cfg[i] == ROCK) sim.nR++;
        else if (cfg[i] == PAPER) sim.nP++;
        else if (cfg[i] == SCISSORS) sim.nS++;
    }
}

static std::string counts_string(const RPS_Sim& sim) {
    return "(nR,nP,nS)=(" + std::to_string(sim.nR) + "," +
           std::to_string(sim.nP) + "," + std::to_string(sim.nS) + ")";
}

// Test 1: extinction condition should trigger iff one species count is zero.
static TestResult test_extinction_condition(const Graph& g) {
    RPS_Sim sim(g, 12345ULL, 1.0);

    // Case A: no species extinct
    set_state_and_counts(sim, {ROCK, PAPER, SCISSORS, ROCK});
    SimResult a = sim.run_until_extinction(1); // one attempt only, enough to probe the initial condition
    bool okA = (a.extinct == -1); // should not be instantly classified as extinct

    // Case B: R extinct
    set_state_and_counts(sim, {PAPER, PAPER, SCISSORS, SCISSORS});
    SimResult b = sim.run_until_extinction(10);
    bool okB = (b.extinct == (int)ROCK);

    // Case C: P extinct
    set_state_and_counts(sim, {ROCK, ROCK, SCISSORS, SCISSORS});
    SimResult c = sim.run_until_extinction(10);
    bool okC = (c.extinct == (int)PAPER);

    // Case D: S extinct
    set_state_and_counts(sim, {ROCK, ROCK, PAPER, PAPER});
    SimResult d = sim.run_until_extinction(10);
    bool okD = (d.extinct == (int)SCISSORS);

    bool ok = okA && okB && okC && okD;
    std::string detail =
        std::string("non-extinct->") + (okA ? "OK" : "WRONG") +
        ", R-missing->" + (okB ? "OK" : "WRONG") +
        ", P-missing->" + (okC ? "OK" : "WRONG") +
        ", S-missing->" + (okD ? "OK" : "WRONG");

    return {"Test 1 - Extinction condition", ok, detail};
}

// Test 2: extinct species label should match the missing species, and already-extinct states should report zero time.
static TestResult test_correct_extinct_species_and_zero_time(const Graph& g) {
    RPS_Sim sim(g, 999ULL, 1.0);

    set_state_and_counts(sim, {PAPER, PAPER, PAPER, SCISSORS}); // ROCK missing already
    SimResult res = sim.run_until_extinction(100);

    bool ok = (res.extinct == (int)ROCK) &&
              (res.Text_attempts == 0) &&
              approx_equal(res.Text_mcs, 0.0);

    std::string detail =
        "extinct=" + std::to_string(res.extinct) +
        ", attempts=" + std::to_string(res.Text_attempts) +
        ", mcs=" + std::to_string(res.Text_mcs);

    return {"Test 2 - Correct extinct species / zero-time detection", ok, detail};
}

// Test 3: time bookkeeping for censored runs should satisfy Text_mcs = attempts / N.
static TestResult test_time_bookkeeping_censored(const Graph& g) {
    RPS_Sim sim(g, 2026ULL, 0.0); // no replacement can ever occur
    set_state_and_counts(sim, {ROCK, PAPER, SCISSORS, ROCK});

    const long long max_attempts = 7;
    SimResult res = sim.run_until_extinction(max_attempts);

    bool ok = (res.extinct == -1) &&
              (res.Text_attempts == max_attempts) &&
              approx_equal(res.Text_mcs, (double)max_attempts / (double)g.N);

    std::string detail =
        "extinct=" + std::to_string(res.extinct) +
        ", attempts=" + std::to_string(res.Text_attempts) +
        ", expected_mcs=" + std::to_string((double)max_attempts / (double)g.N) +
        ", got_mcs=" + std::to_string(res.Text_mcs);

    return {"Test 3 - Time bookkeeping (censored run)", ok, detail};
}

// Test 4: no false extinction when all species are still present.
static TestResult test_no_false_extinction(const Graph& g) {
    RPS_Sim sim(g, 314159ULL, 0.0); // freeze dynamics
    set_state_and_counts(sim, {ROCK, PAPER, SCISSORS, PAPER});

    SimResult res = sim.run_until_extinction(3);
    bool ok = (sim.nR > 0 && sim.nP > 0 && sim.nS > 0) &&
              (res.extinct == -1);

    std::string detail =
        counts_string(sim) +
        ", result.extinct=" + std::to_string(res.extinct);

    return {"Test 4 - No false extinction", ok, detail};
}

// Test 5: very small system (2x2 lattice) should still detect extinction and record sensible time.
static TestResult test_small_system_edge_case() {
    Graph g = build_lattice2D(2); // N=4, degree-2 torus degeneracy is expected
    RPS_Sim sim(g, 777ULL, 1.0);

    // Already extinct on a tiny system: PAPER missing.
    set_state_and_counts(sim, {ROCK, ROCK, SCISSORS, SCISSORS});
    SimResult res0 = sim.run_until_extinction(100);

    bool ok0 = (res0.extinct == (int)PAPER) &&
               (res0.Text_attempts == 0) &&
               approx_equal(res0.Text_mcs, 0.0);

    // Non-extinct tiny system: should eventually either go extinct or hit the cap, but time must be consistent.
    set_state_and_counts(sim, {ROCK, PAPER, SCISSORS, ROCK});
    const long long cap = 200;
    SimResult res1 = sim.run_until_extinction(cap);

    bool time_ok = (res1.Text_attempts >= 0) &&
                   ((res1.extinct == -1 && res1.Text_attempts == cap) ||
                    (res1.extinct != -1 && res1.Text_attempts <= cap)) &&
                   approx_equal(res1.Text_mcs, (double)res1.Text_attempts / (double)g.N);

    bool ok = ok0 && time_ok;

    std::string detail =
        std::string("already-extinct tiny system->") + (ok0 ? "OK" : "WRONG") +
        ", dynamic tiny system: extinct=" + std::to_string(res1.extinct) +
        ", attempts=" + std::to_string(res1.Text_attempts) +
        ", mcs=" + std::to_string(res1.Text_mcs);

    return {"Test 5 - Small-system edge case (2x2 lattice)", ok, detail};
}

static bool run_and_print(const std::vector<TestResult>& results) {
    bool all_ok = true;

    std::cout << "\nExtinction detection test results\n";
    std::cout << "---------------------------------\n";

    for (const auto& r : results) {
        std::cout << (r.passed ? "[PASS] " : "[FAIL] ")
                  << r.name << " -> " << r.detail << "\n";
        all_ok = all_ok && r.passed;
    }

    return all_ok;
}

int main() {
    try {
        // Use a very small graph for targeted logic tests.
        Graph g(4);
        g.name = "manual4";
        g.add_edge_undirected(0, 1);
        g.add_edge_undirected(1, 2);
        g.add_edge_undirected(2, 3);
        g.add_edge_undirected(3, 0);
        g.finalize_simple();

        std::vector<TestResult> results;
        results.push_back(test_extinction_condition(g));
        results.push_back(test_correct_extinct_species_and_zero_time(g));
        results.push_back(test_time_bookkeeping_censored(g));
        results.push_back(test_no_false_extinction(g));
        results.push_back(test_small_system_edge_case());

        bool ok = run_and_print(results);
        std::cout << "\nOverall: " << (ok ? "PASS" : "FAIL") << "\n";
        return ok ? 0 : 2;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
