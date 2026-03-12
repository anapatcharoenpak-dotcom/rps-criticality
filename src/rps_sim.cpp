#include "rps_sim.hpp"
#include <algorithm>

RPS_Sim::RPS_Sim(const Graph& graph, uint64_t seed, double k)
    : g(graph), rng(seed), state(graph.N, 0), kR(k), kP(k), kS(k) {}

RPS_Sim::RPS_Sim(const Graph& graph, uint64_t seed, double kR_, double kP_, double kS_)
    : g(graph), rng(seed), state(graph.N, 0), kR(kR_), kP(kP_), kS(kS_) {}

void RPS_Sim::init_random_uniform() {
    int N = g.N;

    nR = 0;
    nP = 0;
    nS = 0;

    state.resize(N);

    for (int i = 0; i < N; ++i) {
        int x = rng.randint(0, 2); // 0,1,2 with uniform probability

        state[i] = (uint8_t)x;

        if (x == (int)ROCK) nR++;
        else if (x == (int)PAPER) nP++;
        else nS++;
    }
}

inline double RPS_Sim::invasion_rate(Species winner) const {
    if (winner == ROCK) return kR;
    if (winner == PAPER) return kP;
    return kS;
}

inline void RPS_Sim::apply_replace(int loser_idx, Species loser, Species winner) {
    if (loser == ROCK) nR--;
    else if (loser == PAPER) nP--;
    else nS--;

    if (winner == ROCK) nR++;
    else if (winner == PAPER) nP++;
    else nS++;

    state[loser_idx] = (uint8_t)winner;
}

inline bool RPS_Sim::extinct() const {
    return (nR == 0 || nP == 0 || nS == 0);
}

inline int RPS_Sim::extinct_species() const {
    if (nR == 0) return (int)ROCK;
    if (nP == 0) return (int)PAPER;
    if (nS == 0) return (int)SCISSORS;
    return -1;
}

// One update attempt: choose reference node i, choose neighbor j, apply pairwise invasion
void RPS_Sim::step_attempt() {
    int i = rng.randint(0, g.N - 1);
    const auto& nb = g.neighbors(i);
    int j = nb[rng.randint(0, (int)nb.size() - 1)];

    Species si = (Species)state[i];
    Species sj = (Species)state[j];

    if (si == sj) return;

    if (beats(si, sj)) {
        if (rng.u01() < invasion_rate(si)) {
            apply_replace(j, sj, si);
        }
    } else {
        if (rng.u01() < invasion_rate(sj)) {
            apply_replace(i, si, sj);
        }
    }
}

SimResult RPS_Sim::run_until_extinction(long long max_attempts) {
    SimResult out;

    if (extinct()) {
        out.extinct = extinct_species();
        return out;
    }

    long long attempts = 0;

    while (!extinct()) {
        step_attempt();
        attempts++;

        if (max_attempts > 0 && attempts >= max_attempts) {
            // extinction not reached within limit
            out.Text_attempts = attempts;
            out.Text_mcs = (double)attempts / (double)g.N;
            out.extinct = -1;  // special flag: not extinct
            return out;
        }
    }

    // extinction reached normally
    out.Text_attempts = attempts;
    out.Text_mcs = (double)attempts / (double)g.N;
    out.extinct = extinct_species();
    return out;
}