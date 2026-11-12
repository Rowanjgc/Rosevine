// SPDX-License-Identifier: GPL-3.0-only
// Isolated research adapter for the pinned llama C API, not a runtime selection.
// --validate-only checks the input protocol without initializing any backend.
// --run loads weights and MUST be externally supervised in a reviewed run window.
#include "llama.h"
#include "ggml-cpu.h"
#include "capture_input_protocol.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {
struct fd_owner {
    int value;
    explicit fd_owner(int fd) : value(fd) {
        if (fd < 0) throw std::runtime_error("file open failed");
    }
    ~fd_owner() { if (value >= 0) close(value); }
    fd_owner(const fd_owner &) = delete;
    fd_owner & operator=(const fd_owner &) = delete;
};

void read_exact(int fd, void * buffer, size_t count) {
    auto * out = static_cast<unsigned char *>(buffer);
    while (count) {
        const auto n = read(fd, out, count);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) throw std::runtime_error("truncated input");
        out += n;
        count -= n;
    }
}

void write_exact(int fd, const void * buffer, size_t count) {
    const auto * in = static_cast<const unsigned char *>(buffer);
    while (count) {
        const auto n = write(fd, in, count);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) throw std::runtime_error("capture write failed");
        in += n;
        count -= n;
    }
}

struct inputs {
    uint32_t vocabulary, capacity;
    uint64_t rows = 0;
    bool selective_rows = false;
    bool nll_mode = false;
    bool extra_buffers = false;
    std::vector<std::vector<llama_token>> sequences;
    std::vector<std::vector<uint32_t>> capture_positions;
};

inputs load_inputs(const char * path) {
    fd_owner fd(open(path, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC));
    struct stat st{};
    if (fstat(fd.value, &st) || !S_ISREG(st.st_mode) || (st.st_mode & 0777) != 0600
            || st.st_uid != getuid() || st.st_size > 2 * 1024 * 1024) {
        throw std::runtime_error("private bounded input required");
    }
    std::vector<unsigned char> raw(static_cast<size_t>(st.st_size));
    read_exact(fd.value, raw.data(), raw.size());
    const auto parsed = rosevine_capture::decode_inputs(raw.data(), raw.size());
    inputs result;
    result.vocabulary = parsed.vocabulary;
    result.capacity = parsed.capacity;
    result.rows = parsed.rows;
    result.selective_rows = parsed.selective_rows;
    result.nll_mode = parsed.nll_mode;
    result.extra_buffers = parsed.extra_buffers;
    result.capture_positions = parsed.capture_positions;
    result.sequences.reserve(parsed.sequences.size());
    for (const auto & ids : parsed.sequences) {
        std::vector<llama_token> sequence(ids.begin(), ids.end());
        result.sequences.push_back(std::move(sequence));
    }
    return result;
}

double target_nll(const float * logits, uint32_t vocabulary, llama_token target) {
    if (target < 0 || static_cast<uint32_t>(target) >= vocabulary) {
        throw std::runtime_error("target token outside vocabulary");
    }
    long double peak = -std::numeric_limits<long double>::infinity();
    for (uint32_t i = 0; i < vocabulary; ++i) {
        if (!std::isfinite(logits[i])) throw std::runtime_error("nonfinite output logits");
        peak = std::max(peak, static_cast<long double>(logits[i]));
    }
    long double partition = 0;
    for (uint32_t i = 0; i < vocabulary; ++i) {
        partition += std::exp(static_cast<long double>(logits[i]) - peak);
    }
    if (!(partition > 0) || !std::isfinite(partition)) {
        throw std::runtime_error("invalid full-vocabulary partition");
    }
    const long double value = peak - static_cast<long double>(logits[target]) + std::log(partition);
    if (!std::isfinite(value) || value < 0
            || value > static_cast<long double>(std::numeric_limits<double>::max())) {
        throw std::runtime_error("invalid target NLL");
    }
    return static_cast<double>(value);
}

struct batch_owner {
    llama_batch value = llama_batch_init(1, 0, 1);
    batch_owner() {
        if (!value.token || !value.pos || !value.n_seq_id || !value.seq_id
                || !value.seq_id[0] || !value.logits) throw std::runtime_error("batch allocation failed");
    }
    ~batch_owner() { llama_batch_free(value); }
};

void capture(const char * model_path, const inputs & input, const char * directory) {
    // Only a little-endian IEEE754 host can emit this raw FP32 protocol.
    const uint32_t one = 1;
    if (*reinterpret_cast<const unsigned char *>(&one) != 1 || sizeof(float) != 4
            || !std::numeric_limits<float>::is_iec559) throw std::runtime_error("unsupported float encoding");
    fd_owner root(open(directory, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
    struct stat st{};
    if (fstat(root.value, &st) || (st.st_mode & 0777) != 0700 || st.st_uid != getuid()) {
        throw std::runtime_error("private owned output directory required");
    }
    // The caller creates a fresh directory. Reserve its first file before any
    // model load; a rerun cannot overwrite existing partial evidence.
    fd_owner started(openat(root.value, "started", O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600));
    constexpr char logits_marker[] = "rosevine-native-logits-v1\n";
    constexpr char nll_marker[] = "rosevine-native-nll-v1\n";
    const char * marker = input.nll_mode ? nll_marker : logits_marker;
    write_exact(started.value, marker, strlen(marker));
    if (fsync(started.value)) throw std::runtime_error("start marker flush failed");

    // The pinned registry may statically register Vulkan on its first access.
    // Its source-defined disable switch must precede that access. This changes
    // only this process's environment; it is also part of the capture policy.
    if (setenv("GGML_DISABLE_VULKAN", "1", 1)) throw std::runtime_error("backend restriction failed");
    if (!ggml_backend_reg_count()) ggml_backend_register(ggml_backend_cpu_reg());
    if (!ggml_backend_dev_count()) throw std::runtime_error("CPU backend unavailable");
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        if (ggml_backend_dev_type(ggml_backend_dev_get(i)) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            throw std::runtime_error("unexpected non-CPU registered device");
        }
    }
    llama_backend_init();
    ggml_backend_dev_t devices[] = {nullptr};
    auto mp = llama_model_default_params();
    mp.devices = devices;
    mp.n_gpu_layers = 0;
    mp.split_mode = LLAMA_SPLIT_MODE_NONE;
    mp.load_mode = LLAMA_LOAD_MODE_MMAP;
    mp.lazy_mode = LLAMA_LAZY_MODE_OFF;
    mp.load_mtp = false;
    mp.use_extra_bufts = input.extra_buffers;
    mp.vocab_only = false;
    mp.no_alloc = false;
    mp.no_host = false;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(model_path, mp), llama_model_free);
    if (!model) throw std::runtime_error("model load failed");
    if (llama_vocab_n_tokens(llama_model_get_vocab(model.get())) != static_cast<int32_t>(input.vocabulary)) {
        throw std::runtime_error("model vocabulary mismatch");
    }
    auto cp = llama_context_default_params();
    cp.n_ctx = input.capacity;
    cp.n_batch = cp.n_ubatch = cp.n_seq_max = cp.n_outputs_max = 1;
    cp.n_threads = cp.n_threads_batch = 2;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.attention_type = LLAMA_ATTENTION_TYPE_CAUSAL;
    cp.type_k = cp.type_v = GGML_TYPE_F16;
    cp.offload_kqv = cp.op_offload = false;
    cp.embeddings = false;
    cp.moe_cache_size = 0;
    cp.no_perf = true;
    std::unique_ptr<llama_context, decltype(&llama_free)> context(
        llama_init_from_model(model.get(), cp), llama_free);
    if (!context || !llama_get_memory(context.get())) throw std::runtime_error("context initialization failed");
    if (llama_n_ctx(context.get()) != input.capacity || llama_n_batch(context.get()) != 1
            || llama_n_ubatch(context.get()) != 1) throw std::runtime_error("native dimensions changed");
    batch_owner batch;
    uint64_t rows = 0;
    uint32_t shard = 0;
    int output = -1;
    if (input.nll_mode) {
        const uint64_t one = 1;
        if (sizeof(double) != 8 || !std::numeric_limits<double>::is_iec559
                || *reinterpret_cast<const unsigned char *>(&one) != 1) {
            throw std::runtime_error("unsupported float64 encoding");
        }
        output = openat(root.value, "nll-000000.f64",
                        O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
        if (output < 0) throw std::runtime_error("NLL output create failed");
    }
    try {
        for (size_t sequence_index = 0; sequence_index < input.sequences.size(); ++sequence_index) {
            const auto & sequence = input.sequences[sequence_index];
            const auto * selected_positions = input.selective_rows ? &input.capture_positions[sequence_index] : nullptr;
            llama_memory_clear(llama_get_memory(context.get()), true);
            // Each row j predicts the already-frozen target ID at j+1. The last
            // input ID is a target only; no sampled/generated ID enters the KV.
            for (size_t j = 0; j + 1 < sequence.size(); ++j) {
                batch.value.n_tokens = 1;
                batch.value.token[0] = sequence[j];
                batch.value.pos[0] = static_cast<llama_pos>(j);
                batch.value.n_seq_id[0] = 1;
                batch.value.seq_id[0][0] = 0;
                const bool capture_row = input.nll_mode || !input.selective_rows ||
                    std::binary_search(selected_positions->begin(), selected_positions->end(), static_cast<uint32_t>(j));
                batch.value.logits[0] = capture_row ? 1 : 0;
                if (llama_decode(context.get(), batch.value) != 0) throw std::runtime_error("decode failed or aborted");
                if (!capture_row) continue;
                const auto * logits = llama_get_logits_ith(context.get(), 0);
                if (!logits) throw std::runtime_error("missing output logits");
                if (input.nll_mode) {
                    const auto value = target_nll(logits, input.vocabulary, sequence[j + 1]);
                    write_exact(output, &value, sizeof(value));
                    ++rows;
                    continue;
                }
                for (uint32_t v = 0; v < input.vocabulary; ++v) {
                    if (!std::isfinite(logits[v])) throw std::runtime_error("nonfinite output logits");
                }
                if (rows % 32 == 0) {
                    char name[40];
                    snprintf(name, sizeof(name), "logits-%06u.f32", shard++);
                    output = openat(root.value, name, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
                    if (output < 0) throw std::runtime_error("shard create failed");
                }
                write_exact(output, logits, size_t(input.vocabulary) * sizeof(float));
                ++rows;
                if (rows % 32 == 0 || rows == input.rows) {
                    if (fsync(output)) throw std::runtime_error("shard flush failed");
                    const int old = output;
                    output = -1;
                    if (close(old)) throw std::runtime_error("shard close failed");
                }
            }
        }
    } catch (...) {
        if (output >= 0) close(output);
        throw;
    }
    if (input.nll_mode) {
        if (fsync(output)) throw std::runtime_error("NLL output flush failed");
        const int old = output;
        output = -1;
        if (close(old)) throw std::runtime_error("NLL output close failed");
    }
    // Written only after all rows and context/model cleanup complete. A killed
    // process or failed decode retains shards but cannot produce a complete index.
    context.reset();
    model.reset();
    llama_backend_free();
    fd_owner complete(openat(root.value, "complete", O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600));
    const std::string receipt = std::to_string(rows) + "\n";
    write_exact(complete.value, receipt.data(), receipt.size());
    if (fsync(complete.value) || fsync(root.value)) throw std::runtime_error("completion flush failed");
}
} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "--validate-only") {
            load_inputs(argv[2]);
            puts("Capture input valid; no backend initialized or model loaded.");
            return 0;
        }
        if (argc != 5 || std::string(argv[1]) != "--run") {
            throw std::runtime_error("usage: --validate-only INPUT or --run MODEL INPUT PRIVATE_DIRECTORY");
        }
        const auto input = load_inputs(argv[3]);
        capture(argv[2], input, argv[4]);
        puts("Native logit rows written; independent validation still required.");
        return 0;
    } catch (const std::exception &) {
        // Native library diagnostics go only to the private supervised stderr.
        fputs("Native logit capture failed; retain partial evidence.\n", stderr);
        return 1;
    }
}
