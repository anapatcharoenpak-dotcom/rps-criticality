// sanity test 1: graph topology
#include <iostream>
#include <vector>
#include <string>
#include <unordered_set>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cstdint>

#include "graph.hpp"
#include "rng.hpp"

// Forward declarations from graph_builders.cpp
Graph build_lattice2D(int L);
Graph build_watts_strogatz(int N, int K, double beta, RNG& rng);
Graph build_barabasi_albert(int N, int m0, int m, RNG& rng);

struct TestResult {
    std::string name;
    bool passed;
    std::string detail;
};

static bool contains_neighbor(const std::vector<int>& nb, int x) {
    return std::binary_search(nb.begin(), nb.end(), x);
}

static std::vector<int> degrees(const Graph& g) {
    std::vector<int> deg(g.N);
    for (int i = 0; i < g.N; ++i) deg[i] = (int)g.adj[i].size();
    return deg;
}

static int min_degree(const Graph& g) {
    int ans = (g.N > 0) ? (int)g.adj[0].size() : 0;
    for (int i = 1; i < g.N; ++i) ans = std::min(ans, (int)g.adj[i].size());
    return ans;
}

static int max_degree(const Graph& g) {
    int ans = 0;
    for (int i = 0; i < g.N; ++i) ans = std::max(ans, (int)g.adj[i].size());
    return ans;
}

static double mean_degree(const Graph& g) {
    if (g.N == 0) return 0.0;
    long long sum = 0;
    for (int i = 0; i < g.N; ++i) sum += (int)g.adj[i].size();
    return (double)sum / (double)g.N;
}

// Test 1: node count
static TestResult test_node_count(const Graph& g, int expected_N) {
    bool ok = (g.N == expected_N);
    return {
        "Test 1 - Node count",
        ok,
        ok ? "OK" : ("Expected N=" + std::to_string(expected_N) +
                     ", got N=" + std::to_string(g.N))
    };
}

// Test 2: undirected edges
static TestResult test_undirected(const Graph& g) {
    for (int u = 0; u < g.N; ++u) {
        for (int v : g.adj[u]) {
            if (v < 0 || v >= g.N) {
                return {"Test 2 - Undirected edges", false,
                        "Neighbor index out of range at u=" + std::to_string(u) +
                        ", v=" + std::to_string(v)};
            }
            if (!contains_neighbor(g.adj[v], u)) {
                return {"Test 2 - Undirected edges", false,
                        "Missing reverse edge for (" + std::to_string(u) +
                        "," + std::to_string(v) + ")"};
            }
        }
    }
    return {"Test 2 - Undirected edges", true, "OK"};
}

// Test 3: no self-loops
static TestResult test_no_self_loops(const Graph& g) {
    for (int u = 0; u < g.N; ++u) {
        if (contains_neighbor(g.adj[u], u)) {
            return {"Test 3 - No self-loops", false,
                    "Found self-loop at node " + std::to_string(u)};
        }
    }
    return {"Test 3 - No self-loops", true, "OK"};
}

// Test 4: no duplicate edges
static TestResult test_no_duplicate_neighbors(const Graph& g) {
    for (int u = 0; u < g.N; ++u) {
        for (size_t k = 1; k < g.adj[u].size(); ++k) {
            if (g.adj[u][k] == g.adj[u][k - 1]) {
                return {"Test 4 - No duplicate edges", false,
                        "Duplicate neighbor " + std::to_string(g.adj[u][k]) +
                        " at node " + std::to_string(u)};
            }
        }
    }
    return {"Test 4 - No duplicate edges", true, "OK"};
}

// Test 5a: lattice degree pattern
static TestResult test_lattice_degree_pattern(const Graph& g, int L) {
    std::vector<int> deg = degrees(g);

    if (L == 1) {
        bool ok = (deg.size() == 1 && deg[0] == 0);
        return {
            "Test 5 - Lattice degree pattern",
            ok,
            ok ? "OK (L=1 trivial case, degree 0)"
               : "For L=1 expected the only node to have degree 0"
        };
    }

    if (L == 2) {
        bool ok = std::all_of(deg.begin(), deg.end(), [](int d) { return d == 2; });
        return {
            "Test 5 - Lattice degree pattern",
            ok,
            ok ? "OK (L=2 torus degeneracy gives degree 2)"
               : "For L=2 expected all degrees = 2 due to periodic degeneracy"
        };
    }

    bool ok = std::all_of(deg.begin(), deg.end(), [](int d) { return d == 4; });
    return {
        "Test 5 - Lattice degree pattern",
        ok,
        ok ? "OK (all degrees = 4)"
           : "For L>=3 expected all degrees = 4, but min=" +
             std::to_string(*std::min_element(deg.begin(), deg.end())) +
             ", max=" +
             std::to_string(*std::max_element(deg.begin(), deg.end()))
    };
}

// Test 5b: small-world degree pattern
static TestResult test_smallworld_degree_pattern(const Graph& g, int K) {
    std::vector<int> deg = degrees(g);
    bool ok = std::all_of(deg.begin(), deg.end(), [K](int d) { return d == K; });

    return {
        "Test 5 - Small-world degree pattern",
        ok,
        ok ? "OK (all degrees = K)"
           : "Expected all degrees = " + std::to_string(K) +
             ", but min=" + std::to_string(*std::min_element(deg.begin(), deg.end())) +
             ", max=" + std::to_string(*std::max_element(deg.begin(), deg.end()))
    };
}

// Test 5c: scale-free degree pattern
static TestResult test_scalefree_degree_pattern(const Graph& g, int m) {
    int dmin = min_degree(g);
    int dmax = max_degree(g);
    double dmean = mean_degree(g);

    bool has_hub = (dmax > 2 * m);
    bool sensible_min = (dmin >= m || dmin >= 1);

    bool ok = sensible_min && has_hub;

    std::string detail =
        "min_deg=" + std::to_string(dmin) +
        ", max_deg=" + std::to_string(dmax) +
        ", mean_deg=" + std::to_string(dmean);

    if (!sensible_min) detail += " | minimum degree suspicious";
    if (!has_hub) detail += " | no clear hub yet (maybe N too small)";

    return {"Test 5 - Scale-free degree pattern", ok, detail};
}

// Test 6: no isolated nodes
static TestResult test_no_isolated_nodes(const Graph& g) {
    for (int i = 0; i < g.N; ++i) {
        if (g.adj[i].empty()) {
            return {"Test 6 - No isolated nodes", false,
                    "Node " + std::to_string(i) + " has degree 0"};
        }
    }
    return {"Test 6 - No isolated nodes", true, "OK"};
}

static void print_graph_summary(const Graph& g) {
    std::cout << "\nGraph summary\n";
    std::cout << "-------------\n";
    std::cout << "name      : " << g.name << "\n";
    std::cout << "N         : " << g.N << "\n";
    std::cout << "min degree: " << min_degree(g) << "\n";
    std::cout << "max degree: " << max_degree(g) << "\n";
    std::cout << "mean deg  : " << mean_degree(g) << "\n";
}

static bool run_and_print(const std::vector<TestResult>& results) {
    bool all_ok = true;
    std::cout << "\nTopology test results\n";
    std::cout << "---------------------\n";
    for (const auto& r : results) {
        std::cout << (r.passed ? "[PASS] " : "[FAIL] ")
                  << r.name << " -> " << r.detail << "\n";
        all_ok = all_ok && r.passed;
    }
    return all_ok;
}

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr
                << "Usage:\n"
                << "  " << argv[0] << " lattice2D L\n"
                << "  " << argv[0] << " smallworld N K beta seed\n"
                << "  " << argv[0] << " scalefree N m seed\n\n"
                << "Examples:\n"
                << "  " << argv[0] << " lattice2D 3\n"
                << "  " << argv[0] << " smallworld 20 4 0.10 123\n"
                << "  " << argv[0] << " scalefree 50 2 123\n";
            return 1;
        }

        const std::string graph_type = argv[1];
        Graph g;
        std::vector<TestResult> results;

        if (graph_type == "lattice2D") {
            if (argc != 3) throw std::runtime_error("lattice2D needs: L");
            const int L = std::stoi(argv[2]);

            g = build_lattice2D(L);

            results.push_back(test_node_count(g, L * L));
            results.push_back(test_undirected(g));
            results.push_back(test_no_self_loops(g));
            results.push_back(test_no_duplicate_neighbors(g));
            results.push_back(test_lattice_degree_pattern(g, L));
            results.push_back(test_no_isolated_nodes(g));

        } else if (graph_type == "smallworld") {
            if (argc != 6) throw std::runtime_error("smallworld needs: N K beta seed");
            const int N = std::stoi(argv[2]);
            const int K = std::stoi(argv[3]);
            const double beta = std::stod(argv[4]);
            const uint64_t seed = (uint64_t)std::stoull(argv[5]);

            RNG rng(seed);
            g = build_watts_strogatz(N, K, beta, rng);

            results.push_back(test_node_count(g, N));
            results.push_back(test_undirected(g));
            results.push_back(test_no_self_loops(g));
            results.push_back(test_no_duplicate_neighbors(g));
            results.push_back(test_smallworld_degree_pattern(g, K));
            results.push_back(test_no_isolated_nodes(g));

        } else if (graph_type == "scalefree") {
            if (argc != 5) throw std::runtime_error("scalefree needs: N m seed");
            const int N = std::stoi(argv[2]);
            const int m = std::stoi(argv[3]);
            const uint64_t seed = (uint64_t)std::stoull(argv[4]);

            const int m0 = 2 * m; // same default choice as main.cpp
            RNG rng(seed);
            g = build_barabasi_albert(N, m0, m, rng);

            results.push_back(test_node_count(g, N));
            results.push_back(test_undirected(g));
            results.push_back(test_no_self_loops(g));
            results.push_back(test_no_duplicate_neighbors(g));
            results.push_back(test_scalefree_degree_pattern(g, m));
            results.push_back(test_no_isolated_nodes(g));

        } else {
            throw std::runtime_error("Unknown graph_type. Use lattice2D, smallworld, or scalefree.");
        }

        print_graph_summary(g);
        bool ok = run_and_print(results);

        std::cout << "\nOverall: " << (ok ? "PASS" : "FAIL") << "\n";
        return ok ? 0 : 2;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}