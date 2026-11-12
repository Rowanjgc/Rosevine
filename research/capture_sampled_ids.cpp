// Process-local observer for the pinned llama.cpp common sampler entrypoint.
// It forwards the exact token returned by the native sampler and writes a
// checksummable, bounded event stream to stderr. It does not change sampling.
#include <cerrno>
#include <cstdint>
#include <dlfcn.h>
#include <pthread.h>
#include <unistd.h>

struct common_sampler;
struct llama_context;

namespace {
using sample_fn = int (*)(common_sampler *, llama_context *, int, bool);
constexpr char symbol_name[] = "_Z21common_sampler_sampleP14common_samplerP13llama_contextib";
sample_fn next_sample = nullptr;
pthread_mutex_t event_mutex = PTHREAD_MUTEX_INITIALIZER;
uint64_t event_count = 0;
bool write_failed = false;

char * append_unsigned(char * out, uint64_t value) {
    char digits[24];
    unsigned count = 0;
    do {
        digits[count++] = static_cast<char>('0' + value % 10);
        value /= 10;
    } while (value);
    while (count) *out++ = digits[--count];
    return out;
}

bool write_record(const char * data, size_t size) {
    while (size) {
        const ssize_t written = write(STDERR_FILENO, data, size);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) return false;
        data += written;
        size -= static_cast<size_t>(written);
    }
    return true;
}

void log_hook_error() {
    static constexpr char record[] = "ROSEVINE_SAMPLER_HOOK_V1 ERROR target_unavailable\n";
    (void) write_record(record, sizeof(record) - 1);
}

__attribute__((constructor)) void initialize_observer() {
    next_sample = reinterpret_cast<sample_fn>(dlsym(RTLD_NEXT, symbol_name));
    if (next_sample) {
        static constexpr char record[] = "ROSEVINE_SAMPLER_HOOK_V1 READY\n";
        if (!write_record(record, sizeof(record) - 1)) write_failed = true;
    } else {
        log_hook_error();
        write_failed = true;
    }
}

__attribute__((destructor)) void finish_observer() {
    char record[96];
    char * out = record;
    constexpr char prefix[] = "ROSEVINE_SAMPLER_HOOK_V1 DONE ";
    for (char ch : prefix) {
        if (ch == '\0') break;
        *out++ = ch;
    }
    out = append_unsigned(out, event_count);
    *out++ = ' ';
    *out++ = write_failed ? '1' : '0';
    *out++ = '\n';
    (void) write_record(record, static_cast<size_t>(out - record));
}
} // namespace

extern "C" __attribute__((visibility("default"))) int rosevine_sample_observer(
        common_sampler * sampler, llama_context * context, int logits_index, bool grammar_first)
        asm("_Z21common_sampler_sampleP14common_samplerP13llama_contextib");

extern "C" int rosevine_sample_observer(common_sampler * sampler, llama_context * context,
                                         int logits_index, bool grammar_first) {
    if (!next_sample) {
        log_hook_error();
        write_failed = true;
        return -1;
    }
    const int token = next_sample(sampler, context, logits_index, grammar_first);
    const int saved_errno = errno;
    if (pthread_mutex_lock(&event_mutex) == 0) {
        char record[96];
        char * out = record;
        constexpr char prefix[] = "ROSEVINE_SAMPLED_TOKEN_V1 ";
        for (char ch : prefix) {
            if (ch == '\0') break;
            *out++ = ch;
        }
        out = append_unsigned(out, event_count++);
        *out++ = ' ';
        if (token < 0) {
            *out++ = '-';
            out = append_unsigned(out, static_cast<uint64_t>(-static_cast<int64_t>(token)));
        } else {
            out = append_unsigned(out, static_cast<uint64_t>(token));
        }
        *out++ = '\n';
        if (!write_record(record, static_cast<size_t>(out - record))) write_failed = true;
        (void) pthread_mutex_unlock(&event_mutex);
    } else {
        write_failed = true;
    }
    errno = saved_errno;
    return token;
}
