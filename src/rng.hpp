#pragma once
#include <cstdint>
#include <vector>

struct RNG {
    uint64_t s[4]{};

    explicit RNG(uint64_t seed) {
        uint64_t sm = seed;
        for (uint64_t& x : s) x = splitmix64(sm);
    }

    inline uint64_t next_u64() {
        const uint64_t result = rotl(s[1] * 5ULL, 7) * 9ULL;
        const uint64_t t = s[1] << 17;

        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 45);

        return result;
    }

    inline double u01() {
        return (next_u64() >> 11) * (1.0 / 9007199254740992.0);
    }

    inline uint64_t uniform_u64(uint64_t bound) {
        if (bound <= 1) return 0;

        __uint128_t product = static_cast<__uint128_t>(next_u64()) * bound;
        uint64_t low = static_cast<uint64_t>(product);

        if (low < bound) {
            const uint64_t threshold = static_cast<uint64_t>(-bound) % bound;
            while (low < threshold) {
                product = static_cast<__uint128_t>(next_u64()) * bound;
                low = static_cast<uint64_t>(product);
            }
        }

        return static_cast<uint64_t>(product >> 64);
    }

    inline int randint(int lo, int hi_inclusive) {
        const uint64_t span =
            static_cast<uint64_t>(static_cast<int64_t>(hi_inclusive) - static_cast<int64_t>(lo) + 1);
        return lo + static_cast<int>(uniform_u64(span));
    }

    template <class T>
    inline void shuffle(std::vector<T>& v) {
        for (std::size_t i = v.size(); i > 1; --i) {
            const std::size_t j = static_cast<std::size_t>(uniform_u64(i));
            std::swap(v[i - 1], v[j]);
        }
    }

private:
    static inline uint64_t rotl(uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }

    static inline uint64_t splitmix64(uint64_t& x) {
        x += 0x9e3779b97f4a7c15ULL;
        uint64_t z = x;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
};
