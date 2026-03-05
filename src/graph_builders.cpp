#include "graph.hpp"
#include "rng.hpp"

#include <unordered_set>
#include <stdexcept>

// 2D periodic lattice LxL (von Neumann 4-neighbor)
Graph build_lattice2D(int L) {
    int N = L * L;
    Graph g(N);
    g.name = "lattice2D";

    auto idx = [L](int x, int y) {
        x = (x % L + L) % L;
        y = (y % L + L) % L;
        return y * L + x;
    };

    for (int y = 0; y < L; ++y) {
        for (int x = 0; x < L; ++x) {
            int u = idx(x, y);
            g.add_edge_undirected(u, idx(x + 1, y));
            g.add_edge_undirected(u, idx(x - 1, y));
            g.add_edge_undirected(u, idx(x, y + 1));
            g.add_edge_undirected(u, idx(x, y - 1));
        }
    }

    g.finalize_simple();
    return g;
}

// Watts–Strogatz small-world:
// start ring lattice (degree K even), rewire forward edges with prob beta
Graph build_watts_strogatz(int N, int K, double beta, RNG& rng) {
    if (K % 2 != 0) throw std::runtime_error("Watts–Strogatz requires even K.");
    if (K >= N) throw std::runtime_error("Watts–Strogatz requires K < N.");

    Graph g(N);
    g.name = "smallworld";

    std::vector<std::unordered_set<int>> nbset(N);

    auto add_undirected_set = [&](int u, int v) {
        if (u == v) return;
        nbset[u].insert(v);
        nbset[v].insert(u);
    };

    int half = K / 2;

    // ring lattice
    for (int u = 0; u < N; ++u) {
        for (int d = 1; d <= half; ++d) {
            int v = (u + d) % N;
            add_undirected_set(u, v);
        }
    }

    // rewire forward edges
    for (int u = 0; u < N; ++u) {
        for (int d = 1; d <= half; ++d) {
            int v = (u + d) % N;
            if (!nbset[u].count(v)) continue;

            if (rng.u01() < beta) {
                // remove (u,v)
                nbset[u].erase(v);
                nbset[v].erase(u);

                // choose new w
                int w;
                int guard = 0;
                do {
                    w = rng.randint(0, N - 1);
                    guard++;
                    if (guard > 10 * N) { w = v; break; }
                } while (w == u || nbset[u].count(w));

                add_undirected_set(u, w);
            }
        }
    }

    // convert to adjacency list
    for (int i = 0; i < N; ++i) {
        g.adj[i].reserve(nbset[i].size());
        for (int v : nbset[i]) g.adj[i].push_back(v);
    }
    g.finalize_simple();

    // safety: no isolated nodes
    for (int i = 0; i < N; ++i) {
        if (g.adj[i].empty()) throw std::runtime_error("Isolated node generated; adjust parameters.");
    }

    return g;
}

// --- Scale-free network: Barabási–Albert (preferential attachment) ---
// Parameters:
//   N  : total nodes
//   m0 : initial fully-connected core size (>=2)
//   m  : edges added per new node (1 <= m <= m0-1)
//
// Implementation notes:
// - Undirected graph.
// - Uses "repeated nodes" method to sample proportional to degree.
// - No self-loops, no multi-edges.
Graph build_barabasi_albert(int N, int m0, int m, RNG& rng) {
    if (N <= 0) throw std::runtime_error("BA: N must be > 0");
    if (m0 < 2) throw std::runtime_error("BA: m0 must be >= 2");
    if (m < 1) throw std::runtime_error("BA: m must be >= 1");
    if (m >= m0) throw std::runtime_error("BA: require m < m0");
    if (m0 > N) throw std::runtime_error("BA: require m0 <= N");

    Graph g(N);
    g.name = "scalefree";

    // Use sets during construction to avoid duplicate edges easily.
    std::vector<std::unordered_set<int>> nbset(N);

    auto add_edge_set = [&](int u, int v) {
        if (u == v) return false;
        if (nbset[u].count(v)) return false;
        nbset[u].insert(v);
        nbset[v].insert(u);
        return true;
    };

    // 1) Initial complete graph on m0 nodes
    for (int u = 0; u < m0; ++u) {
        for (int v = u + 1; v < m0; ++v) {
            add_edge_set(u, v);
        }
    }

    // 2) "Repeated nodes" list for preferential sampling
    // Each node appears in the list proportional to its current degree.
    std::vector<int> repeated;
    repeated.reserve(2 * N * m);

    // Initialize repeated list from the complete graph
    for (int u = 0; u < m0; ++u) {
        int deg = (int)nbset[u].size(); // should be m0-1
        for (int k = 0; k < deg; ++k) repeated.push_back(u);
    }

    // 3) Add new nodes one by one
    for (int new_node = m0; new_node < N; ++new_node) {
        std::unordered_set<int> targets;
        targets.reserve(m * 2);

        // Select m distinct existing nodes with probability ~ degree
        int guard = 0;
        while ((int)targets.size() < m) {
            if (repeated.empty()) {
                // Should not happen, but safeguard
                int fallback = rng.randint(0, new_node - 1);
                if (fallback != new_node) targets.insert(fallback);
            } else {
                int pick = repeated[rng.randint(0, (int)repeated.size() - 1)];
                if (pick != new_node) targets.insert(pick);
            }
            guard++;
            if (guard > 1000000) throw std::runtime_error("BA: selection guard triggered (unexpected).");
        }

        // Add edges from new_node to chosen targets
        for (int t : targets) {
            add_edge_set(new_node, t);
        }

        // Update repeated list:
        // - new node appears deg(new_node)=m times
        // - each target gains +1 degree => add target once
        for (int k = 0; k < m; ++k) repeated.push_back(new_node);
        for (int t : targets) repeated.push_back(t);
    }

    // Convert nbset -> adjacency list
    for (int i = 0; i < N; ++i) {
        g.adj[i].reserve(nbset[i].size());
        for (int v : nbset[i]) g.adj[i].push_back(v);
    }
    g.finalize_simple();

    // Basic safety: no isolated nodes (BA should not create isolated nodes if m>=1)
    for (int i = 0; i < N; ++i) {
        if (g.adj[i].empty()) {
            throw std::runtime_error("BA: isolated node found; check parameters.");
        }
    }
    return g;
}