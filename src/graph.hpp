#pragma once
#include <vector>
#include <string>
#include <algorithm>

struct Graph {
    std::string name;
    int N = 0;
    std::vector<std::vector<int>> adj;

    explicit Graph(int n = 0) : N(n), adj(n) {}

    const std::vector<int>& neighbors(int i) const { return adj[i]; }

    void add_edge_undirected(int u, int v) {
        if (u == v) return;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    void finalize_simple() {
        for (int i = 0; i < N; ++i) {
            auto& nb = adj[i];
            nb.erase(std::remove(nb.begin(), nb.end(), i), nb.end());
            std::sort(nb.begin(), nb.end());
            nb.erase(std::unique(nb.begin(), nb.end()), nb.end());
        }
    }
};