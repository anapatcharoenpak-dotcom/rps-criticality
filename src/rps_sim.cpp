#include "rps_sim.hpp"
#include <algorithm>

RPS_Sim::RPS_Sim(const Graph& graph, uint64_t seed, double k)
    : g(graph), rng(seed), state(graph.N, 0), kR(k), kP(k), kS(k) {}

RPS_Sim::RPS_Sim(const Graph& graph, uint64_t seed, double kR_, double kP_, double kS_)
    : g(graph), rng(seed), state(graph.N, 0), kR(kR_), kP(kP_), kS(kS_) {}

void RPS_Sim::init_random_uniform() {
    const int N = g.N;

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

inline double RPS_Sim::invasion_rate(uint8_t winner) const {
    if (winner == ROCK) return kR;
    if (winner == PAPER) return kP;
    return kS;
}

inline bool RPS_Sim::apply_replace(int loser_idx, uint8_t loser, uint8_t winner) {
    bool extinct_now = false;

    if (loser == ROCK) extinct_now = (--nR == 0);
    else if (loser == PAPER) extinct_now = (--nP == 0);
    else extinct_now = (--nS == 0);

    if (winner == ROCK) nR++;
    else if (winner == PAPER) nP++;
    else nS++;

    state[loser_idx] = winner;
    return extinct_now;
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
bool RPS_Sim::step_attempt() {
    static constexpr uint8_t prey_of[3] = { (uint8_t)SCISSORS, (uint8_t)ROCK, (uint8_t)PAPER };

    const int i = rng.randint(0, g.N - 1);
    const auto& nb = g.neighbors(i);
    const int j = nb[rng.randint(0, (int)nb.size() - 1)];

    const uint8_t si = state[i];
    const uint8_t sj = state[j];

    if (si == sj) return false;

    const bool i_wins = (prey_of[si] == sj);
    const uint8_t winner = i_wins ? si : sj;
    const uint8_t loser = i_wins ? sj : si;
    const int loser_idx = i_wins ? j : i;
    const double rate = invasion_rate(winner);

    if (rate <= 0.0) return false;
    if (rate >= 1.0 || rng.u01() < rate) {
        return apply_replace(loser_idx, loser, winner);
    }

    return false;
}

SimResult RPS_Sim::run_until_extinction(long long max_attempts) {
    SimResult out;

    if (extinct()) {
        out.extinct = extinct_species();
        return out;
    }

    long long attempts = 0;

    while (max_attempts <= 0 || attempts < max_attempts) {
        attempts++;

        if (step_attempt()) {
            out.Text_attempts = attempts;
            out.Text_mcs = (double)attempts / (double)g.N;
            out.extinct = extinct_species();
            return out;
        }
    }

    out.Text_attempts = attempts;
    out.Text_mcs = (double)attempts / (double)g.N;
    out.extinct = -1;
    return out;
}
