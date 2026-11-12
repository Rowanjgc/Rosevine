// SPDX-License-Identifier: GPL-3.0-only
// Bounded CPU-only diagnostic callback for named Gemma 4 MoE graph tensors.
// This helper must be externally supervised under an approved run profile.
#include "llama.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
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
constexpr size_t capture_limit = 100 * 1024 * 1024;
constexpr size_t tensor_limit = 16 * 1024 * 1024;
constexpr const char * selected[] = {
    "ffn_moe_logits-", "ffn_moe_probs-", "ffn_moe_topk-", "ffn_moe_weights-",
    "ffn_moe_gate_up-", "ffn_moe_gelu-", "ffn_moe_down-", "ffn_moe_weighted-",
    "ffn_moe_out-", "ffn_moe-", "ffn_moe_combined-",
};

struct fd_owner {
    int value;
    explicit fd_owner(int fd) : value(fd) {
        if (fd < 0) throw std::runtime_error("file open failed");
    }
    ~fd_owner() { if (value >= 0) close(value); }
    fd_owner(const fd_owner &) = delete;
    fd_owner & operator=(const fd_owner &) = delete;
};

void write_exact(int fd, const void * buffer, size_t count) {
    auto * p = static_cast<const uint8_t *>(buffer);
    while (count) {
        const auto n = write(fd, p, count);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) throw std::runtime_error("capture write failed");
        p += n;
        count -= size_t(n);
    }
}

template<class T> void write_value(int fd, T value) { write_exact(fd, &value, sizeof(value)); }

struct inputs {
    uint32_t vocabulary = 0;
    uint32_t capacity = 0;
    std::vector<llama_token> tokens;
};

void read_exact(int fd, void * buffer, size_t count) {
    auto * p = static_cast<uint8_t *>(buffer);
    while (count) {
        const auto n = read(fd, p, count);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) throw std::runtime_error("truncated input");
        p += n;
        count -= size_t(n);
    }
}

uint32_t read_u32(int fd) {
    uint8_t b[4];
    read_exact(fd, b, sizeof(b));
    return uint32_t(b[0]) | uint32_t(b[1]) << 8 | uint32_t(b[2]) << 16 | uint32_t(b[3]) << 24;
}

inputs load_inputs(const char * path) {
    fd_owner fd(open(path, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC));
    struct stat st{};
    if (fstat(fd.value, &st) || !S_ISREG(st.st_mode) || (st.st_mode & 0777) != 0600
            || st.st_uid != getuid() || st.st_size < 24 || st.st_size > 256 * 1024) {
        throw std::runtime_error("bounded private input required");
    }
    char magic[8];
    read_exact(fd.value, magic, sizeof(magic));
    if (memcmp(magic, "RVLOG001", 8)) throw std::runtime_error("input version mismatch");
    inputs in;
    in.vocabulary = read_u32(fd.value);
    in.capacity = read_u32(fd.value);
    const auto sequences = read_u32(fd.value);
    if (!in.vocabulary || in.vocabulary > 262144 || !in.capacity || in.capacity > 4096
            || in.capacity % 256 || sequences != 1) throw std::runtime_error("single-sequence envelope required");
    const auto length = read_u32(fd.value);
    if (length < 2 || length > 513 || length > in.capacity) throw std::runtime_error("input length outside envelope");
    in.tokens.reserve(length);
    for (uint32_t i = 0; i < length; ++i) {
        const auto token = read_u32(fd.value);
        if (token >= in.vocabulary) throw std::runtime_error("input token outside vocabulary");
        in.tokens.push_back(static_cast<llama_token>(token));
    }
    char extra;
    if (read(fd.value, &extra, 1) != 0) throw std::runtime_error("trailing input bytes");
    return in;
}

struct observer {
    int fd = -1;
    size_t bytes = 8;
    uint32_t records = 0;
    uint32_t position = UINT32_MAX;
    bool failed = false;
};

bool selected_name(const char * name) {
    for (const auto * prefix : selected) {
        const size_t n = strlen(prefix);
        if (strncmp(name, prefix, n) == 0 && name[n] >= '0' && name[n] <= '9') return true;
    }
    return false;
}

bool observe_tensor(ggml_tensor * tensor, bool ask, void * user_data) {
    auto & out = *static_cast<observer *>(user_data);
    if (!selected_name(tensor->name) || out.position != 0) return false;
    if (ask) return true;
    try {
        const char * type = ggml_type_name(tensor->type);
        if (strcmp(type, "f32") != 0 && strcmp(type, "i32") != 0) throw std::runtime_error("unsupported capture dtype");
        const size_t nbytes = ggml_nbytes(tensor);
        const size_t name_size = strlen(tensor->name);
        const size_t type_size = strlen(type);
        if (!nbytes || nbytes > tensor_limit || !ggml_is_contiguous(tensor) || name_size > 127 || type_size > 15
                || out.bytes + 4 + name_size + type_size + sizeof(uint32_t) + 4 * sizeof(uint64_t) + sizeof(uint64_t) + nbytes > capture_limit) {
            throw std::runtime_error("intermediate capture budget exceeded");
        }
        std::vector<uint8_t> data(nbytes);
        if (tensor->buffer) {
            ggml_backend_tensor_get(tensor, data.data(), 0, nbytes);
        } else if (tensor->data) {
            // CPU scheduler callbacks can expose synchronized host-backed
            // nodes without a backend buffer handle. This helper rejects all
            // non-CPU devices before graph execution, so copying the host data
            // pointer is valid for this diagnostic path.
            memcpy(data.data(), tensor->data, nbytes);
        } else {
            throw std::runtime_error("observed tensor has no readable CPU storage");
        }
        if (tensor->type == GGML_TYPE_F32) {
            const auto * values = reinterpret_cast<const float *>(data.data());
            for (size_t i = 0; i < nbytes / sizeof(float); ++i) {
                if (!std::isfinite(values[i])) throw std::runtime_error("nonfinite intermediate");
            }
        }
        const uint16_t name_len = static_cast<uint16_t>(name_size);
        const uint16_t type_len = static_cast<uint16_t>(type_size);
        write_value(out.fd, out.position);
        write_value(out.fd, name_len);
        write_exact(out.fd, tensor->name, name_size);
        write_value(out.fd, type_len);
        write_exact(out.fd, type, type_size);
        write_value(out.fd, static_cast<uint32_t>(ggml_n_dims(tensor)));
        for (int i = 0; i < GGML_MAX_DIMS; ++i) write_value(out.fd, static_cast<uint64_t>(tensor->ne[i]));
        write_value(out.fd, static_cast<uint64_t>(nbytes));
        write_exact(out.fd, data.data(), data.size());
        out.bytes += 4 + name_size + type_size + sizeof(uint32_t) + 4 * sizeof(uint64_t) + sizeof(uint64_t) + nbytes;
        ++out.records;
        return true;
    } catch (...) {
        out.failed = true;
        return false;
    }
}

struct batch_owner {
    llama_batch value = llama_batch_init(1, 0, 1);
    batch_owner() {
        if (!value.token || !value.pos || !value.n_seq_id || !value.seq_id || !value.seq_id[0] || !value.logits)
            throw std::runtime_error("batch allocation failed");
    }
    ~batch_owner() { llama_batch_free(value); }
};

void run(const char * model_path, const char * input_path, const char * directory, bool repack) {
    const auto in = load_inputs(input_path);
    const uint32_t endian = 1;
    if (*reinterpret_cast<const uint8_t *>(&endian) != 1 || sizeof(float) != 4 || !std::numeric_limits<float>::is_iec559)
        throw std::runtime_error("little-endian IEEE754 host required");
    fd_owner root(open(directory, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
    struct stat st{};
    if (fstat(root.value, &st) || !S_ISDIR(st.st_mode) || (st.st_mode & 0777) != 0700 || st.st_uid != getuid())
        throw std::runtime_error("private output directory required");
    fd_owner started(openat(root.value, "started", O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600));
    constexpr char marker[] = "rosevine-moe-intermediates-v1\n";
    write_exact(started.value, marker, sizeof(marker) - 1);
    if (fsync(started.value)) throw std::runtime_error("start marker flush failed");

    if (setenv("GGML_DISABLE_VULKAN", "1", 1)) throw std::runtime_error("CPU-only restriction failed");
    if (!ggml_backend_reg_count()) ggml_backend_register(ggml_backend_cpu_reg());
    if (!ggml_backend_dev_count()) throw std::runtime_error("CPU backend unavailable");
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        if (ggml_backend_dev_type(ggml_backend_dev_get(i)) != GGML_BACKEND_DEVICE_TYPE_CPU)
            throw std::runtime_error("non-CPU backend registered");
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
    mp.use_extra_bufts = repack;
    mp.vocab_only = false;
    mp.no_alloc = false;
    mp.no_host = false;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(model_path, mp), llama_model_free);
    if (!model || llama_vocab_n_tokens(llama_model_get_vocab(model.get())) != static_cast<int32_t>(in.vocabulary))
        throw std::runtime_error("model load or vocabulary validation failed");

    auto cp = llama_context_default_params();
    cp.n_ctx = in.capacity;
    cp.n_batch = cp.n_ubatch = cp.n_seq_max = cp.n_outputs_max = 1;
    cp.n_threads = cp.n_threads_batch = 2;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.attention_type = LLAMA_ATTENTION_TYPE_CAUSAL;
    cp.type_k = cp.type_v = GGML_TYPE_F16;
    cp.offload_kqv = cp.op_offload = false;
    cp.embeddings = false;
    cp.moe_cache_size = 0;
    cp.no_perf = true;
    observer capture;
    fd_owner data(openat(root.value, "nodes.bin", O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600));
    capture.fd = data.value;
    constexpr char data_magic[] = "RVMOE001";
    write_exact(capture.fd, data_magic, sizeof(data_magic) - 1);
    auto observer_data = capture;
    cp.cb_eval = observe_tensor;
    cp.cb_eval_user_data = &observer_data;
    std::unique_ptr<llama_context, decltype(&llama_free)> context(llama_init_from_model(model.get(), cp), llama_free);
    if (!context || !llama_get_memory(context.get()) || llama_n_ctx(context.get()) != in.capacity)
        throw std::runtime_error("context initialization or capacity validation failed");

    batch_owner batch;
    llama_memory_clear(llama_get_memory(context.get()), true);
    for (size_t j = 0; j + 1 < in.tokens.size(); ++j) {
        observer_data.position = static_cast<uint32_t>(j);
        batch.value.n_tokens = 1;
        batch.value.token[0] = in.tokens[j];
        batch.value.pos[0] = static_cast<llama_pos>(j);
        batch.value.n_seq_id[0] = 1;
        batch.value.seq_id[0][0] = 0;
        batch.value.logits[0] = 1;
        const int status = llama_decode(context.get(), batch.value);
        if (observer_data.failed || status != 0) throw std::runtime_error("decode or intermediate capture failed");
    }
    if (observer_data.records == 0) throw std::runtime_error("no named MoE tensors observed at position zero");
    if (fsync(data.value)) throw std::runtime_error("intermediate capture flush failed");
    const uint32_t records = observer_data.records;
    context.reset();
    model.reset();
    llama_backend_free();
    fd_owner complete(openat(root.value, "complete", O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600));
    write_value(complete.value, records);
    write_value(complete.value, uint32_t(0));
    if (fsync(complete.value) || fsync(root.value)) throw std::runtime_error("completion receipt flush failed");
}
} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc != 6 || std::string(argv[1]) != "--run")
            throw std::runtime_error("usage: --run MODEL INPUT PRIVATE_DIRECTORY ordinary|repack");
        const std::string mode = argv[5];
        if (mode != "ordinary" && mode != "repack") throw std::runtime_error("invalid buffer mode");
        run(argv[2], argv[3], argv[4], mode == "repack");
        puts("Bounded MoE intermediates captured; offline validation and runtime equivalence are still required.");
        return 0;
    } catch (const std::exception &) {
        fputs("MoE intermediate capture failed; retain partial evidence.\n", stderr);
        return 1;
    }
}
