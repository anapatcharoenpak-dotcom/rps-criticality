// sanity test 2: initialization
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdint>

#include "graph.hpp"
#include "rng.hpp"
#include "rps_sim.hpp"

// Forward declarations
Graph build_lattice2D(int L);

struct TestResult {
    std::string name;
    bool passed;
    std::string detail;
};

static TestResult test_count_conservation(RPS_Sim& sim, int N) {

    int sum = sim.nR + sim.nP + sim.nS;

    bool ok = (sum == N);

    return {
        "Test 1 - Count conservation",
        ok,
        ok ? "OK" :
        ("Counts do not sum to N: " + std::to_string(sum) +
         " vs N=" + std::to_string(N))
    };
}

static TestResult test_state_vector_consistency(RPS_Sim& sim) {

    int cR = 0, cP = 0, cS = 0;

    for (auto s : sim.state) {
        if (s == ROCK) cR++;
        else if (s == PAPER) cP++;
        else if (s == SCISSORS) cS++;
    }

    bool ok = (cR == sim.nR && cP == sim.nP && cS == sim.nS);

    std::string detail =
        "vector=(" + std::to_string(cR) + "," +
                      std::to_string(cP) + "," +
                      std::to_string(cS) + ")"
        " counters=(" + std::to_string(sim.nR) + "," +
                        std::to_string(sim.nP) + "," +
                        std::to_string(sim.nS) + ")";

    return {"Test 2 - State vector consistency", ok, detail};
}

static TestResult test_uniform_statistics(const Graph& g, uint64_t seed) {

    RPS_Sim sim(g, seed, 1.0);
    sim.init_random_uniform();

    double pR = (double)sim.nR / g.N;
    double pP = (double)sim.nP / g.N;
    double pS = (double)sim.nS / g.N;

    double target = 1.0 / 3.0;

    bool ok =
        std::abs(pR - target) < 0.15 &&
        std::abs(pP - target) < 0.15 &&
        std::abs(pS - target) < 0.15;

    std::string detail =
        "fractions=(" +
        std::to_string(pR) + "," +
        std::to_string(pP) + "," +
        std::to_string(pS) + ")";

    return {"Test 3 - Uniform probability statistics", ok, detail};
}

static TestResult test_seed_reproducibility(const Graph& g, uint64_t seed) {

    RPS_Sim sim1(g, seed, 1.0);
    RPS_Sim sim2(g, seed, 1.0);

    sim1.init_random_uniform();
    sim2.init_random_uniform();

    bool ok = (sim1.state == sim2.state);

    return {"Test 4 - Seed reproducibility", ok,
            ok ? "States identical for same seed"
               : "States differ with same seed"};
}

static TestResult test_seed_variation(const Graph& g, uint64_t seed) {

    RPS_Sim sim1(g, seed, 1.0);
    RPS_Sim sim2(g, seed + 1, 1.0);

    sim1.init_random_uniform();
    sim2.init_random_uniform();

    bool ok = (sim1.state != sim2.state);

    return {"Test 5 - Seed variation", ok,
            ok ? "Different seeds produce different states"
               : "States identical (unexpected)"};
}

static bool run_and_print(const std::vector<TestResult>& results) {

    bool all_ok = true;

    std::cout << "\nInitialization test results\n";
    std::cout << "---------------------------\n";

    for (const auto& r : results) {

        std::cout << (r.passed ? "[PASS] " : "[FAIL] ")
                  << r.name << " -> "
                  << r.detail << "\n";

        all_ok = all_ok && r.passed;
    }

    return all_ok;
}

int main() {

    const int L = 20;
    const uint64_t seed = 123;

    Graph g = build_lattice2D(L);

    RPS_Sim sim(g, seed, 1.0);
    sim.init_random_uniform();

    std::vector<TestResult> results;

    results.push_back(test_count_conservation(sim, g.N));
    results.push_back(test_state_vector_consistency(sim));
    results.push_back(test_uniform_statistics(g, seed));
    results.push_back(test_seed_reproducibility(g, seed));
    results.push_back(test_seed_variation(g, seed));

    bool ok = run_and_print(results);

    std::cout << "\nOverall: " << (ok ? "PASS" : "FAIL") << "\n";

    return ok ? 0 : 2;
}