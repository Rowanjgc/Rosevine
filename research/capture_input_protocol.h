// SPDX-License-Identifier: GPL-3.0-only
// Bounded binary input framing shared by the isolated capture helper and its
// standalone protocol checks. This header has no llama/ggml API dependency.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace rosevine_capture {

struct parsed_inputs {
    uint32_t vocabulary = 0;
    uint32_t capacity = 0;
    uint64_t rows = 0;
    bool selective_rows = false;
    bool nll_mode = false;
    bool extra_buffers = false;
    std::vector<std::vector<uint32_t>> sequences;
    std::vector<std::vector<uint32_t>> capture_positions;
};

inline uint32_t read_u32(const unsigned char * data, size_t size, size_t & offset) {
    if (offset > size || size - offset < 4) throw std::runtime_error("truncated input");
    const auto value = uint32_t(data[offset]) | uint32_t(data[offset + 1]) << 8
                     | uint32_t(data[offset + 2]) << 16 | uint32_t(data[offset + 3]) << 24;
    offset += 4;
    return value;
}

inline parsed_inputs decode_inputs(const unsigned char * data, size_t size) {
    if (!data || size < 20 || size > 2 * 1024 * 1024) {
        throw std::runtime_error("input size outside capture envelope");
    }
    const bool v3 = std::memcmp(data, "RVLOG003", 8) == 0;
    const bool selective_v2 = std::memcmp(data, "RVLOG002", 8) == 0;
    const bool nll = std::memcmp(data, "RVNLL001", 8) == 0;
    if (!v3 && !selective_v2 && !nll && std::memcmp(data, "RVLOG001", 8) != 0) {
        throw std::runtime_error("input version mismatch");
    }

    parsed_inputs result;
    result.selective_rows = selective_v2;
    result.nll_mode = nll;
    size_t offset = 8;
    result.vocabulary = read_u32(data, size, offset);
    result.capacity = read_u32(data, size, offset);
    const auto count = read_u32(data, size, offset);
    if (v3) {
        const auto flags = read_u32(data, size, offset);
        if (flags & ~3U) throw std::runtime_error("unknown RVLOG003 policy flags");
        result.selective_rows = (flags & 1U) != 0;
        result.extra_buffers = (flags & 2U) != 0;
    }
    if (!result.vocabulary || result.vocabulary > 262144 || !result.capacity
            || result.capacity > 4096 || result.capacity % 256 || !count
            || count > (nll ? 1024U : 8U)) {
        throw std::runtime_error("input dimensions outside capture envelope");
    }
    for (uint32_t sequence_index = 0; sequence_index < count; ++sequence_index) {
        const auto length = read_u32(data, size, offset);
        if (length < 2 || length > ((result.selective_rows || nll)
                                    ? result.capacity : 513)
                || length > result.capacity) {
            throw std::runtime_error("input sequence outside capture envelope");
        }
        std::vector<uint32_t> sequence;
        sequence.reserve(length);
        for (uint32_t j = 0; j < length; ++j) {
            const auto token = read_u32(data, size, offset);
            if (token >= result.vocabulary) throw std::runtime_error("invalid input token");
            sequence.push_back(token);
        }
        result.sequences.push_back(std::move(sequence));
        if (result.selective_rows) {
            const auto selected_count = read_u32(data, size, offset);
            if (!selected_count || selected_count > 96
                    || result.rows + selected_count > 96) {
                throw std::runtime_error("selected output row count outside capture envelope");
            }
            std::vector<uint32_t> positions;
            positions.reserve(selected_count);
            for (uint32_t i = 0; i < selected_count; ++i) {
                const auto position = read_u32(data, size, offset);
                if (position >= length - 1
                        || (!positions.empty() && position <= positions.back())) {
                    throw std::runtime_error("selected output positions invalid or unordered");
                }
                positions.push_back(position);
            }
            result.rows += selected_count;
            result.capture_positions.push_back(std::move(positions));
        } else {
            result.rows += length - 1;
            if (nll && result.rows > 200000) {
                throw std::runtime_error("NLL target count exceeds capture envelope");
            }
        }
    }
    if (offset != size) throw std::runtime_error("trailing input bytes");
    return result;
}

} // namespace rosevine_capture
