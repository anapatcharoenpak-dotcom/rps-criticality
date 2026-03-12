// sanity test 3: update rule
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <sstream>

#include "graph.hpp"
#include "rng.hpp"

// test-only hack so we can call step_attempt() and helper methods
#define private public
#include "rps_sim.hpp"
#undef private

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

static Graph build_two_node_graph() {
    Graph g(2);
    g.name = "two_node";
    g.add_edge_undirected(0, 1);
    g.finalize_simple();
    return g;
}

static Graph build_line3_graph() {
    Graph g(3);
    g.name = "line3";
    g.add_edge_undirected(0, 1);
    g.add_edge_undirected(1, 2);
    g.finalize_simple();
    return g;
}

static void set_state(RPS_Sim& sim, const std::vector<Species>& s) {
    if ((int)s.size() != sim.g.N) {
        throw std::runtime_error("State size does not match graph size.");
    }

    sim.state.assign(sim.g.N, 0);
    sim.nR = sim.nP = sim.nS = 0;

    for (int i = 0; i < sim.g.N; ++i) {
        sim.state[i] = (uint8_t)s[i];
        if (s[i] == ROCK) sim.nR++;
        else if (s[i] == PAPER) sim.nP++;
        else if (s[i] == SCISSORS) sim.nS++;
    }
}

static std::string state_string(const RPS_Sim& sim) {
    std::ostringstream oss;
    oss << "[";
    for (int i = 0; i < sim.g.N; ++i) {
        if (i) oss << " ";
        oss << species_name(sim.state[i]);
    }
    oss << "]";
    return oss.str();
}

static int count_state_differences(const std::vector<uint8_t>& a,
                                   const std::vector<uint8_t>& b) {
    if (a.size() != b.size()) return -1;
    int diff = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) diff++;
    }
    return diff;
}

// Test 1: k = 1 gives deterministic invasion on a two-node graph.
// Initial state: [R, S]. Since R beats S, after one attempt the state must become [R, R].
static TestResult test_deterministic_invasion() {
    Graph g = build_two_node_graph();
    RPS_Sim sim(g, 12345ULL, 1.0);
    set_state(sim, {ROCK, SCISSORS});

    sim.step_attempt();

    bool ok = ((Species)sim.state[0] == ROCK) &&
              ((Species)sim.state[1] == ROCK) &&
              sim.nR == 2 && sim.nP == 0 && sim.nS == 0;

    std::string detail =
        "final_state=" + state_string(sim) +
        " counters=(" + std::to_string(sim.nR) + "," +
                         std::to_string(sim.nP) + "," +
                         std::to_string(sim.nS) + ")";

    return {"Test 1 - Deterministic invasion (k=1)", ok, detail};
}

// Test 2: k = 0 forbids invasion on a two-node graph.
// Initial state: [R, S]. Since the invasion probability is zero, one attempt must leave the state unchanged.
static TestResult test_no_invasion() {
    Graph g = build_two_node_graph();
    RPS_Sim sim(g, 12345ULL, 0.0);
    set_state(sim, {ROCK, SCISSORS});

    std::vector<uint8_t> before = sim.state;
    sim.step_attempt();

    bool ok = (sim.state == before) && sim.nR == 1 && sim.nP == 0 && sim.nS == 1;

    std::string detail =
        "initial=[R S], final=" + state_string(sim) +
        " counters=(" + std::to_string(sim.nR) + "," +
                         std::to_string(sim.nP) + "," +
                         std::to_string(sim.nS) + ")";

    return {"Test 2 - No invasion (k=0)", ok, detail};
}

// Test 3: population conservation under repeated updates.
// After each attempt, nR+nP+nS must remain equal to N.
static TestResult test_population_conservation() {
    Graph g = build_line3_graph();
    RPS_Sim sim(g, 20260312ULL, 1.0);
    set_state(sim, {ROCK, PAPER, SCISSORS});

    const int N = g.N;
    bool ok = true;
    std::string detail = "OK";

    for (int t = 1; t <= 100; ++t) {
        sim.step_attempt();
        int total = sim.nR + sim.nP + sim.nS;
        if (total != N) {
            ok = false;
            detail = "Failed at step " + std::to_string(t) +
                     ": total=" + std::to_string(total) +
                     ", expected=" + std::to_string(N) +
                     ", state=" + state_string(sim);
            break;
        }
    }

    if (ok) {
        detail = "Total count stayed at N=" + std::to_string(N) +
                 " for 100 attempts";
    }

    return {"Test 3 - Population conservation", ok, detail};
}

// Test 4: one-node update rule.
// In one attempt, at most one node can change because apply_replace() acts on a single loser index.
static TestResult test_one_node_update() {
    Graph g = build_line3_graph();
    RPS_Sim sim(g, 987654321ULL, 1.0);
    set_state(sim, {ROCK, PAPER, SCISSORS});

    bool ok = true;
    std::string detail = "OK";

    for (int t = 1; t <= 100; ++t) {
        std::vector<uint8_t> before = sim.state;
        sim.step_attempt();
        int diff = count_state_differences(before, sim.state);

        if (diff < 0 || diff > 1) {
            ok = false;
            detail = "Failed at step " + std::to_string(t) +
                     ": changed_nodes=" + std::to_string(diff) +
                     ", before_size=" + std::to_string(before.size()) +
                     ", after_size=" + std::to_string(sim.state.size());
            break;
        }
    }

    if (ok) {
        detail = "In 100 attempts, each update changed at most one node";
    }

    return {"Test 4 - One-node update", ok, detail};
}

static bool run_and_print(const std::vector<TestResult>& results) {
    bool all_ok = true;
    std::cout << "Update-rule test results\n";
    std::cout << "------------------------\n";

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
        results.push_back(test_deterministic_invasion());
        results.push_back(test_no_invasion());
        results.push_back(test_population_conservation());
        results.push_back(test_one_node_update());

        bool ok = run_and_print(results);
        std::cout << "\nOverall: " << (ok ? "PASS" : "FAIL") << "\n";
        return ok ? 0 : 2;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
