// SPDX-License-Identifier: GPL-3.0-only
// Parity track P1: fixed-token replay timing adapter for the pinned llama C API.
// It is a measurement adapter, not a runtime selection. llama-bench feeds random
// token IDs, which changes MoE routing; this harness replays the same frozen IDs
// that the Rosevine `replay` command consumes, so both arms route identically
// up to numerical routing flips.
//
// Engine-only boundary (both arms): synchronized completion of each evaluation,
// last-position logits produced on the host, no sampling, no tokenization.
//
// --validate-only TOKENS N_PROMPT N_DECODE checks the input without loading a backend.
// --run MODEL TOKENS N_PROMPT N_DECODE REPS CTX BATCH UBATCH THREADS CPU_MOE KV FA REPACK
// loads weights and MUST be externally supervised in a reviewed run window.
// Output: one JSON config line, then one JSON line per repetition (stdout).
#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::vector<std::string> g_buffer_lines;

void log_callback(ggml_log_level level, const char * text, void *) {
    // Keep only allocation summaries (no paths); everything else is discarded.
    if (text && std::strstr(text, "buffer size") && g_buffer_lines.size() < 256) {
        std::string line(text);
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
        g_buffer_lines.push_back(line);
    }
    if (level == GGML_LOG_LEVEL_ERROR) fputs("native error (detail withheld)\n", stderr);
}

std::string json_escape(const std::string & s) {
    std::string out;
    for (const unsigned char c : s) {
        if (c == '"' || c == '\\') { out += '\\'; out += static_cast<char>(c); }
        else if (c < 0x20) { char b[8]; snprintf(b, sizeof(b), "\\u%04x", c); out += b; }
        else out += static_cast<char>(c);
    }
    return out;
}

uint64_t fnv1a64(const std::vector<llama_token> & toks) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (const llama_token t : toks) {
        const uint32_t v = static_cast<uint32_t>(t);
        for (int i = 0; i < 4; ++i) {
            h ^= (v >> (8 * i)) & 0xff;
            h *= 0x100000001b3ULL;
        }
    }
    return h;
}

long parse_count(const char * s, long lo, long hi) {
    char * end = nullptr;
    errno = 0;
    const long v = std::strtol(s, &end, 10);
    if (errno || !end || *end || v < lo || v > hi) throw std::runtime_error("invalid count argument");
    return v;
}

std::vector<llama_token> load_tokens(const char * path, long n_prompt, long n_decode) {
    struct closer { void operator()(FILE * f) const { fclose(f); } };
    std::unique_ptr<FILE, closer> f(fopen(path, "rb"));
    if (!f) throw std::runtime_error("token file open failed");
    std::vector<llama_token> toks;
    long v = 0;
    while (fscanf(f.get(), "%ld", &v) == 1) {
        if (v < 0 || v > INT32_MAX || toks.size() >= (1u << 20)) throw std::runtime_error("invalid token");
        toks.push_back(static_cast<llama_token>(v));
    }
    if (!feof(f.get())) throw std::runtime_error("malformed token file");
    if (n_prompt < 1 || static_cast<long>(toks.size()) < n_prompt + n_decode) {
        throw std::runtime_error("token file shorter than prompt plus decode");
    }
    toks.resize(static_cast<size_t>(n_prompt + n_decode));
    return toks;
}

int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Evaluate tokens[from, to) in n_batch chunks; logits only for the final position.
void eval_span(llama_context * ctx, const std::vector<llama_token> & toks, long from, long to, long n_batch) {
    llama_batch batch = llama_batch_init(static_cast<int32_t>(n_batch), 0, 1);
    try {
        for (long p = from; p < to; p += n_batch) {
            const long n = std::min(n_batch, to - p);
            batch.n_tokens = static_cast<int32_t>(n);
            for (long i = 0; i < n; ++i) {
                batch.token[i] = toks[static_cast<size_t>(p + i)];
                batch.pos[i] = static_cast<llama_pos>(p + i);
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = (p + i == to - 1) ? 1 : 0;
            }
            if (llama_decode(ctx, batch) != 0) throw std::runtime_error("decode failed");
        }
        llama_synchronize(ctx);
        if (!llama_get_logits_ith(ctx, -1)) throw std::runtime_error("missing output logits");
    } catch (...) {
        llama_batch_free(batch);
        throw;
    }
    llama_batch_free(batch);
}

int argmax(llama_context * ctx, const llama_vocab * vocab) {
    const float * lg = llama_get_logits_ith(ctx, -1);
    if (!lg) throw std::runtime_error("missing output logits");
    const int32_t n = llama_vocab_n_tokens(vocab);
    int32_t best = 0;
    for (int32_t i = 1; i < n; ++i) {
        if (lg[i] > lg[best]) best = i;
    }
    return best;
}

void run(char ** a) {
    const char * model_path = a[0];
    const long n_prompt = parse_count(a[2], 1, 1 << 20);
    const long n_decode = parse_count(a[3], 0, 1 << 16);
    const auto toks = load_tokens(a[1], n_prompt, n_decode);
    const long reps = parse_count(a[4], 1, 1000);
    const long n_ctx = parse_count(a[5], n_prompt + n_decode, 1 << 20);
    const long n_batch = parse_count(a[6], 1, 1 << 16);
    const long n_ubatch = parse_count(a[7], 1, n_batch);
    const long threads = parse_count(a[8], 1, 256);
    const long cpu_moe = parse_count(a[9], 0, 1024);
    const std::string kv = a[10], fa = a[11], repack = a[12];
    if ((kv != "q8_0" && kv != "f16") || (fa != "on" && fa != "off") || (repack != "on" && repack != "off")) {
        throw std::runtime_error("kv must be q8_0|f16, fa on|off, repack on|off");
    }
    llama_log_set(log_callback, nullptr);
    llama_backend_init();
    // Same per-layer pattern as common's --n-cpu-moe at this pin (LLM_FFN_EXPS_REGEX).
    std::vector<std::string> patterns;
    std::vector<llama_model_tensor_buft_override> overrides;
    for (long l = 0; l < cpu_moe; ++l) {
        patterns.push_back("blk\\." + std::to_string(l) + "\\.ffn_(up|down|gate|gate_up)_(ch|)exps");
    }
    for (const auto & p : patterns) overrides.push_back({p.c_str(), ggml_backend_cpu_buffer_type()});
    overrides.push_back({nullptr, nullptr});
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = -1;
    mp.split_mode = LLAMA_SPLIT_MODE_NONE;
    mp.main_gpu = 0;
    mp.load_mode = LLAMA_LOAD_MODE_MMAP;
    mp.load_mtp = false;
    mp.use_extra_bufts = repack == "on";
    mp.tensor_buft_overrides = overrides.data();
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(model_path, mp), llama_model_free);
    if (!model) throw std::runtime_error("model load failed");
    auto cp = llama_context_default_params();
    cp.n_ctx = static_cast<uint32_t>(n_ctx);
    cp.n_batch = static_cast<uint32_t>(n_batch);
    cp.n_ubatch = static_cast<uint32_t>(n_ubatch);
    cp.n_seq_max = 1;
    cp.n_threads = cp.n_threads_batch = static_cast<int32_t>(threads);
    cp.flash_attn_type = fa == "on" ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.type_k = cp.type_v = kv == "q8_0" ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
    cp.moe_cache_size = 0;
    cp.no_perf = true;
    std::unique_ptr<llama_context, decltype(&llama_free)> context(
        llama_init_from_model(model.get(), cp), llama_free);
    if (!context) throw std::runtime_error("context initialization failed");
    llama_context * ctx = context.get();
    const auto * vocab = llama_model_get_vocab(model.get());
    for (const auto t : toks) {
        if (t >= llama_vocab_n_tokens(vocab)) throw std::runtime_error("token outside vocabulary");
    }
    std::string buffers;
    for (size_t i = 0; i < g_buffer_lines.size(); ++i) {
        buffers += (i ? ",\"" : "\"") + json_escape(g_buffer_lines[i]) + "\"";
    }
    printf("{\"schema\":\"rosevine-parity-replay-v1\",\"kind\":\"config\",\"engine\":\"llama.cpp\","
           "\"n_prompt\":%ld,\"n_decode\":%ld,\"reps\":%ld,\"n_ctx\":%u,\"n_batch\":%u,\"n_ubatch\":%u,"
           "\"threads\":%ld,\"cpu_moe_layers\":%ld,\"kv_type\":\"%s\",\"flash_attn\":\"%s\","
           "\"cpu_repack\":\"%s\",\"cpu_expert_activation\":\"q8_0\",\"swa_full\":%s,\"op_offload\":%s,"
           "\"tokens_fnv1a64\":\"%016llx\",\"buffers\":[%s]}\n",
           n_prompt, n_decode, reps, llama_n_ctx(ctx), llama_n_batch(ctx), llama_n_ubatch(ctx), threads,
           cpu_moe, kv.c_str(), fa.c_str(), repack.c_str(), cp.swa_full ? "true" : "false",
           cp.op_offload ? "true" : "false", static_cast<unsigned long long>(fnv1a64(toks)), buffers.c_str());
    fflush(stdout);
    for (long r = 0; r < reps; ++r) {
        llama_memory_clear(llama_get_memory(ctx), true);
        const int64_t t0 = now_ns();
        eval_span(ctx, toks, 0, n_prompt, n_batch);
        const int64_t prefill_ns = now_ns() - t0;
        const int prefill_top1 = argmax(ctx, vocab);
        std::string steps, top1;
        for (long i = 0; i < n_decode; ++i) {
            const int64_t s = now_ns();
            eval_span(ctx, toks, n_prompt + i, n_prompt + i + 1, 1);
            steps += (i ? "," : "") + std::to_string(now_ns() - s);
            // Correctness record only; computed after the timed span closes.
            top1 += (i ? "," : "") + std::to_string(argmax(ctx, vocab));
        }
        printf("{\"schema\":\"rosevine-parity-replay-v1\",\"kind\":\"rep\",\"rep\":%ld,"
               "\"prefill_ns\":%lld,\"decode_ns\":[%s],\"prefill_top1\":%d,\"top1\":[%s]}\n",
               r, static_cast<long long>(prefill_ns), steps.c_str(), prefill_top1, top1.c_str());
        fflush(stdout);
    }
    context.reset();
    model.reset();
    llama_backend_free();
}
} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc == 5 && std::string(argv[1]) == "--validate-only") {
            const auto toks = load_tokens(argv[2], parse_count(argv[3], 1, 1 << 20), parse_count(argv[4], 0, 1 << 16));
            printf("{\"valid\":true,\"tokens\":%zu,\"tokens_fnv1a64\":\"%016llx\"}\n", toks.size(),
                   static_cast<unsigned long long>(fnv1a64(toks)));
            return 0;
        }
        if (argc != 15 || std::string(argv[1]) != "--run") {
            throw std::runtime_error("usage: --validate-only TOKENS N_PROMPT N_DECODE | --run MODEL TOKENS "
                                     "N_PROMPT N_DECODE REPS CTX BATCH UBATCH THREADS CPU_MOE KV FA REPACK");
        }
        run(argv + 2);
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "parity replay failed: %s\n", e.what());
        return 1;
    }
}
