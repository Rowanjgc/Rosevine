// SPDX-License-Identifier: GPL-3.0-only
#include "capture_input_protocol.h"

#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

using bytes = std::vector<unsigned char>;

static void append_u32(bytes & out, uint32_t value) {
    out.push_back(static_cast<unsigned char>(value));
    out.push_back(static_cast<unsigned char>(value >> 8));
    out.push_back(static_cast<unsigned char>(value >> 16));
    out.push_back(static_cast<unsigned char>(value >> 24));
}

static bytes selected_context(uint32_t flags = 3, uint32_t final_position = 598) {
    bytes out{'R','V','L','O','G','0','0','3'};
    append_u32(out, 64);
    append_u32(out, 1024);
    append_u32(out, 1);
    append_u32(out, flags);
    append_u32(out, 600);
    for (uint32_t i = 0; i < 600; ++i) append_u32(out, i % 64);
    append_u32(out, 3);
    append_u32(out, 2);
    append_u32(out, 512);
    append_u32(out, final_position);
    return out;
}

static bool rejects(const bytes & raw) {
    try {
        (void) rosevine_capture::decode_inputs(raw.data(), raw.size());
    } catch (const std::runtime_error &) {
        return true;
    }
    return false;
}

int main() {
    const auto raw = selected_context();
    const auto parsed = rosevine_capture::decode_inputs(raw.data(), raw.size());
    assert(parsed.vocabulary == 64 && parsed.capacity == 1024);
    assert(parsed.selective_rows && parsed.extra_buffers && !parsed.nll_mode);
    assert(parsed.rows == 3 && parsed.sequences.size() == 1);
    assert(parsed.sequences[0].size() == 600);
    assert((parsed.capture_positions[0] == std::vector<uint32_t>{2, 512, 598}));
    assert(rejects(selected_context(7)));
    assert(rejects(selected_context(3, 599)));

    auto trailing = raw;
    trailing.push_back(0);
    assert(rejects(trailing));

    bytes legacy{'R','V','L','O','G','0','0','1'};
    append_u32(legacy, 64);
    append_u32(legacy, 1024);
    append_u32(legacy, 1);
    append_u32(legacy, 514);
    for (uint32_t i = 0; i < 514; ++i) append_u32(legacy, i % 64);
    assert(rejects(legacy));

    bytes nll{'R','V','N','L','L','0','0','1'};
    append_u32(nll, 64);
    append_u32(nll, 1024);
    append_u32(nll, 1);
    append_u32(nll, 600);
    for (uint32_t i = 0; i < 600; ++i) append_u32(nll, i % 64);
    const auto parsed_nll = rosevine_capture::decode_inputs(nll.data(), nll.size());
    assert(parsed_nll.nll_mode && !parsed_nll.selective_rows && !parsed_nll.extra_buffers);
    assert(parsed_nll.rows == 599 && parsed_nll.sequences[0].size() == 600);
    return 0;
}
