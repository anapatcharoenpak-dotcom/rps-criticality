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