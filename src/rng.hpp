#pragma once
#include <random>
#include <algorithm>
#include <cstdint>
#include <vector>

struct RNG {
    std::mt19937_64 eng;
    std::uniform_real_distribution<double> uni01{0.0, 1.0};

    explicit RNG(uint64_t seed) : eng(seed) {}

    inline double u01() { return uni01(eng); }

    inline int randint(int lo, int hi_inclusive) {
        std::uniform_int_distribution<int> dist(lo, hi_inclusive);
        return dist(eng);
    }

    template <class T>
    inline void shuffle(std::vector<T>& v) {
        std::shuffle(v.begin(), v.end(), eng);
    }
};