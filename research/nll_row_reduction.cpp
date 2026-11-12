// SPDX-License-Identifier: GPL-3.0-only
// Independent full-vocabulary target-NLL reduction for private FP32 rows.
// Reads exactly one little-endian FP32 row from stdin; emits one scalar only.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

namespace {
bool read_u32(uint32_t & value) {
    unsigned char bytes[4];
    std::cin.read(reinterpret_cast<char *>(bytes), sizeof(bytes));
    if (std::cin.gcount() != static_cast<std::streamsize>(sizeof(bytes))) return false;
    value = uint32_t(bytes[0]) | uint32_t(bytes[1]) << 8 |
            uint32_t(bytes[2]) << 16 | uint32_t(bytes[3]) << 24;
    return true;
}
}

int main(int argc, char **) {
    if (argc != 1) return 2;
    uint32_t target = 0, vocabulary = 0;
    if (!read_u32(target) || !read_u32(vocabulary)
            || vocabulary == 0 || vocabulary > 262144 || target >= vocabulary) return 2;
    std::vector<float> logits(vocabulary);
    const auto bytes = static_cast<std::streamsize>(vocabulary) *
                       static_cast<std::streamsize>(sizeof(float));
    std::cin.read(reinterpret_cast<char *>(logits.data()), bytes);
    if (std::cin.gcount() != bytes) return 3;
    char extra = 0;
    if (std::cin.read(&extra, 1)) return 4;

    long double peak = -std::numeric_limits<long double>::infinity();
    for (float value : logits) {
        if (!std::isfinite(value)) return 5;
        peak = std::max(peak, static_cast<long double>(value));
    }
    long double partition = 0;
    for (float value : logits) partition += std::exp(static_cast<long double>(value) - peak);
    if (!(partition > 0) || !std::isfinite(partition)) return 6;
    const long double nll = peak - static_cast<long double>(logits[target]) + std::log(partition);
    if (!std::isfinite(nll) || nll < 0) return 7;
    std::cout << std::setprecision(std::numeric_limits<long double>::max_digits10) << nll << '\n';
    return std::cout ? 0 : 8;
}
