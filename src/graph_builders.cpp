#include "graph.hpp"
#include "rng.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

inline bool contains_neighbor(const std::vector<int>& nb, int x) {
    return std::find(nb.begin(), nb.end(), x) != nb.end();
}

inline void erase_neighbor(std::vector<int>& nb, int x) {
    auto it = std::find(nb.begin(), nb.end(), x);
    if (it == nb.end()) return;
    *it = nb.back();
    nb.pop_back();
}

} // namespace

// 2D periodic lattice LxL (von Neumann 4-neighbor)
Graph build_lattice2D(int L) {
    const int N = L * L;
    Graph g(N);
    g.name = "lattice2D";

    auto idx = [L](int x, int y) {
        x = (x % L + L) % L;
        y = (y % L + L) % L;
        return y * L + x;
    };

    for (int y = 0; y < L; ++y) {
        for (int x = 0; x < L; ++x) {
            const int u = idx(x, y);
            auto& nb = g.adj[u];
            nb.reserve(4);
            nb.push_back(idx(x + 1, y));
            nb.push_back(idx(x - 1, y));
            nb.push_back(idx(x, y + 1));
            nb.push_back(idx(x, y - 1));
        }
    }

    g.finalize_simple();
    return g;
}

// Degree-preserving small-world builder:
// start from a K-regular ring lattice, then randomize via edge swaps.
Graph build_watts_strogatz(int N, int K, double beta, RNG& rng) {
    if (K % 2 != 0) throw std::runtime_error("Watts-Strogatz requires even K.");
    if (K >= N) throw std::runtime_error("Watts-Strogatz requires K < N.");

    Graph g(N);
    g.name = "smallworld";
    for (int i = 0; i < N; ++i) g.adj[i].reserve(K);

    const int half = K / 2;
    std::vector<std::pair<int, int>> forward_edges;
    forward_edges.reserve(static_cast<std::size_t>(N) * static_cast<std::size_t>(half));

    for (int u = 0; u < N; ++u) {
        for (int d = 1; d <= half; ++d) {
            const int v = (u + d) % N;
            g.add_edge_undirected(u, v);
            forward_edges.push_back({u, v});
        }
    }

    for (std::size_t edge_idx = 0; edge_idx < forward_edges.size(); ++edge_idx) {
        if (rng.u01() >= beta) continue;

        const int u = forward_edges[edge_idx].first;
        const int v = forward_edges[edge_idx].second;

        int guard = 0;
        while (guard++ < 10 * N) {
            const std::size_t swap_idx =
                static_cast<std::size_t>(rng.randint(0, static_cast<int>(forward_edges.size()) - 1));
            if (swap_idx == edge_idx) continue;

            const int x = forward_edges[swap_idx].first;
            const int y = forward_edges[swap_idx].second;

            if (u == x || u == y || v == x || v == y) continue;
            if (contains_neighbor(g.adj[u], y) || contains_neighbor(g.adj[x], v)) continue;

            erase_neighbor(g.adj[u], v);
            erase_neighbor(g.adj[v], u);
            erase_neighbor(g.adj[x], y);
            erase_neighbor(g.adj[y], x);

            g.add_edge_undirected(u, y);
            g.add_edge_undirected(x, v);

            forward_edges[edge_idx] = {u, y};
            forward_edges[swap_idx] = {x, v};
            break;
        }
    }

    g.finalize_simple();

    for (int i = 0; i < N; ++i) {
        if (g.adj[i].empty()) throw std::runtime_error("Isolated node generated; adjust parameters.");
    }

    return g;
}

// --- Scale-free network: Barabasi-Albert (preferential attachment) ---
// Parameters:
//   N  : total nodes
//   m0 : initial fully-connected core size (>=2)
//   m  : edges added per new node (1 <= m <= m0-1)
//
// Implementation notes:
// - Undirected graph.
// - Uses the repeated-nodes method to sample proportional to degree.
// - No self-loops, no multi-edges.
Graph build_barabasi_albert(int N, int m0, int m, RNG& rng) {
    if (N <= 0) throw std::runtime_error("BA: N must be > 0");
    if (m0 < 2) throw std::runtime_error("BA: m0 must be >= 2");
    if (m < 1) throw std::runtime_error("BA: m must be >= 1");
    if (m >= m0) throw std::runtime_error("BA: require m < m0");
    if (m0 > N) throw std::runtime_error("BA: require m0 <= N");

    Graph g(N);
    g.name = "scalefree";
    for (int u = 0; u < m0; ++u) g.adj[u].reserve(m0 - 1);

    for (int u = 0; u < m0; ++u) {
        for (int v = u + 1; v < m0; ++v) {
            g.add_edge_undirected(u, v);
        }
    }

    std::vector<int> repeated;
    repeated.reserve(static_cast<std::size_t>(2) * static_cast<std::size_t>(N) * static_cast<std::size_t>(m));

    for (int u = 0; u < m0; ++u) {
        for (int k = 0; k < m0 - 1; ++k) repeated.push_back(u);
    }

    for (int new_node = m0; new_node < N; ++new_node) {
        std::vector<int> targets;
        g.adj[new_node].reserve(m);
        targets.reserve(m * 2);

        int guard = 0;
        while ((int)targets.size() < m) {
            int pick;
            if (repeated.empty()) {
                pick = rng.randint(0, new_node - 1);
            } else {
                pick = repeated[rng.randint(0, (int)repeated.size() - 1)];
            }

            if (pick != new_node && std::find(targets.begin(), targets.end(), pick) == targets.end()) {
                targets.push_back(pick);
            }

            guard++;
            if (guard > 1000000) throw std::runtime_error("BA: selection guard triggered (unexpected).");
        }

        for (int t : targets) {
            g.add_edge_undirected(new_node, t);
        }

        for (int k = 0; k < m; ++k) repeated.push_back(new_node);
        for (int t : targets) repeated.push_back(t);
    }

    g.finalize_simple();

    for (int i = 0; i < N; ++i) {
        if (g.adj[i].empty()) {
            throw std::runtime_error("BA: isolated node found; check parameters.");
        }
    }
    return g;
}
