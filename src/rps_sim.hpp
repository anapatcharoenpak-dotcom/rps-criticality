#pragma once
#include "graph.hpp"
#include "rng.hpp"

#include <vector>
#include <cstdint>

enum Species : uint8_t { ROCK = 0, PAPER = 1, SCISSORS = 2 };

static inline bool beats(Species a, Species b) {
    return (a == ROCK && b == SCISSORS) ||
           (a == SCISSORS && b == PAPER) ||
           (a == PAPER && b == ROCK);
}

struct SimResult {
    long long Text_attempts = 0;
    double Text_mcs = 0.0;
    int extinct = -1; // 0 R, 1 P, 2 S
};

struct RPS_Sim {
    const Graph& g;
    RNG rng;
    std::vector<uint8_t> state;
    int nR = 0, nP = 0, nS = 0;

    // invasion rates (symmetric by default)
    double kR, kP, kS;

    RPS_Sim(const Graph& graph, uint64_t seed, double k);
    RPS_Sim(const Graph& graph, uint64_t seed, double kR_, double kP_, double kS_);

    void init_random_uniform();

    SimResult run_until_extinction(long long max_attempts = -1);

private:
    inline double invasion_rate(Species winner) const;
    inline void apply_replace(int loser_idx, Species loser, Species winner);
    inline bool extinct() const;
    inline int extinct_species() const;
    inline void step_attempt();
};