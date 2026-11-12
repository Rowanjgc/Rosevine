// SPDX-License-Identifier: GPL-3.0-only
// Test double ONLY: link instead of every llama/ggml library. No weight file,
// native backend, GPU, tokenizer or inference implementation is used here.
// Compiling against the explicitly supplied pinned headers checks signatures;
// this is not a deployed ABI, numerical or performance validation.
#include "llama.h"
#include "ggml-cpu.h"
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

struct llama_model {};
struct llama_vocab {};
struct llama_context { llama_context_params params; };

namespace {
std::string scenario;
bool registered = false;
int sequence = -1, position = 0, calls = 0;
bool last_capture = false;
char device, registry, memory;
llama_vocab vocabulary;
std::vector<float> values(64);

void require(bool condition) {
    if (!condition) throw std::runtime_error("test-double API contract violation");
}

void hang() {
    signal(SIGTERM, SIG_IGN);
    puts("DOUBLE waiting; ignores TERM");
    fflush(stdout);
    for (;;) pause();
}
} // namespace

extern "C" {
size_t ggml_backend_reg_count(void) {
    const auto * disabled = getenv("GGML_DISABLE_VULKAN");
    require(disabled && std::string(disabled) == "1");
    return registered ? 1 : 0;
}
ggml_backend_reg_t ggml_backend_cpu_reg(void) {
    return reinterpret_cast<ggml_backend_reg_t>(&registry);
}
void ggml_backend_register(ggml_backend_reg_t reg) {
    require(reg == reinterpret_cast<ggml_backend_reg_t>(&registry));
    registered = true;
}
size_t ggml_backend_dev_count(void) {
    return registered && !getenv("ROSEVINE_DOUBLE_NO_DEVICE") ? 1 : 0;
}
ggml_backend_dev_t ggml_backend_dev_get(size_t index) {
    require(index == 0);
    return reinterpret_cast<ggml_backend_dev_t>(&device);
}
enum ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t dev) {
    require(dev == reinterpret_cast<ggml_backend_dev_t>(&device));
    return getenv("ROSEVINE_DOUBLE_GPU") ? GGML_BACKEND_DEVICE_TYPE_GPU : GGML_BACKEND_DEVICE_TYPE_CPU;
}
void llama_backend_init(void) { puts("DOUBLE backend-init"); }
void llama_backend_free(void) { puts("DOUBLE backend-free"); }
llama_model_params llama_model_default_params(void) { return {}; }
llama_context_params llama_context_default_params(void) { return {}; }
llama_model * llama_model_load_from_file(const char * name, llama_model_params p) {
    // name is a scenario literal, never a path; the test double never opens it.
    scenario = name;
    require(p.devices && !p.devices[0] && p.n_gpu_layers == 0
            && p.split_mode == LLAMA_SPLIT_MODE_NONE && p.load_mode == LLAMA_LOAD_MODE_MMAP
            && p.lazy_mode == LLAMA_LAZY_MODE_OFF && !p.load_mtp && !p.vocab_only
            && !p.no_alloc && p.use_extra_bufts == (scenario == "extra-buffers")
            && !p.no_host);
    puts("DOUBLE model-load");
    if (scenario == "hang-load") hang();
    if (scenario == "load-failure") return nullptr;
    return new llama_model;
}
void llama_model_free(llama_model * p) {
    puts("DOUBLE model-free");
    delete p;
}
const llama_vocab * llama_model_get_vocab(const llama_model * p) {
    require(p != nullptr);
    return &vocabulary;
}
int32_t llama_vocab_n_tokens(const llama_vocab * p) {
    require(p == &vocabulary);
    return scenario == "wrong-vocabulary" ? 63 : 64;
}
llama_context * llama_init_from_model(llama_model * p, llama_context_params c) {
    const uint32_t expected_context = scenario == "selected-context" ? 1024 : 256;
    require(p && c.n_ctx == expected_context && c.n_batch == 1 && c.n_ubatch == 1
            && c.n_seq_max == 1 && c.n_outputs_max == 1 && c.n_threads == 2
            && c.n_threads_batch == 2 && c.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED
            && c.attention_type == LLAMA_ATTENTION_TYPE_CAUSAL && c.type_k == GGML_TYPE_F16
            && c.type_v == GGML_TYPE_F16 && !c.offload_kqv && !c.op_offload
            && !c.embeddings && !c.moe_cache_size && c.no_perf);
    puts("DOUBLE context-init");
    if (scenario == "context-failure") return nullptr;
    return new llama_context{c};
}
void llama_free(llama_context * p) { puts("DOUBLE context-free"); delete p; }
llama_memory_t llama_get_memory(const llama_context * p) {
    require(p != nullptr);
    return reinterpret_cast<llama_memory_t>(&memory);
}
uint32_t llama_n_ctx(const llama_context * p) {
    return p->params.n_ctx + (scenario == "changed-capacity" ? 256 : 0);
}
uint32_t llama_n_batch(const llama_context * p) { return p->params.n_batch; }
uint32_t llama_n_ubatch(const llama_context * p) { return p->params.n_ubatch; }
void llama_memory_clear(llama_memory_t p, bool data) {
    require(p == reinterpret_cast<llama_memory_t>(&memory) && data);
    ++sequence;
    position = 0;
    printf("DOUBLE clear %d\n", sequence);
}
llama_batch llama_batch_init(int32_t count, int32_t embd, int32_t seqs) {
    require(count == 1 && embd == 0 && seqs == 1);
    llama_batch b{};
    b.token = new llama_token[1];
    b.pos = new llama_pos[1];
    b.n_seq_id = new int32_t[1];
    b.seq_id = new llama_seq_id *[1];
    b.seq_id[0] = new llama_seq_id[1];
    b.logits = new int8_t[1];
    return b;
}
void llama_batch_free(llama_batch b) {
    delete[] b.token; delete[] b.pos; delete[] b.n_seq_id;
    delete[] b.seq_id[0]; delete[] b.seq_id; delete[] b.logits;
    puts("DOUBLE batch-free");
}
int32_t llama_decode(llama_context * p, llama_batch b) {
    require(p && sequence >= 0 && b.n_tokens == 1 && !b.embd
            && b.pos[0] == position && b.n_seq_id[0] == 1
            && b.seq_id[0][0] == 0 && (b.logits[0] == 0 || b.logits[0] == 1));
    printf("DOUBLE decode %d %d %d\n", sequence, position, b.token[0]);
    last_capture = b.logits[0] != 0;
    if (last_capture) printf("DOUBLE selected %d %d\n", sequence, position);
    ++calls;
    if (scenario == "hang-decode" && calls == 2) hang();
    if (scenario == "decode-warning" && calls == 2) return 1;
    if (scenario == "decode-abort" && calls == 2) return 2;
    if (scenario == "decode-error" && calls == 2) return -1;
    if (last_capture) {
        for (size_t v = 0; v < values.size(); ++v) {
            values[v] = float(sequence * 10000 + position * 100 + b.token[0] + v);
        }
    }
    if (scenario == "nonfinite" && calls == 2) values[3] = std::numeric_limits<float>::quiet_NaN();
    ++position;
    return 0;
}
float * llama_get_logits_ith(llama_context * p, int32_t index) {
    require(p && index == 0 && last_capture);
    if (scenario == "missing-logits" && calls == 2) return nullptr;
    return values.data();
}
} // extern C
