// SPDX-License-Identifier: GPL-3.0-only
// Model-free smoke check for the native intermediate observer and tensor reads.
#define main rosevine_moe_capture_cli_main
#include "capture_moe_intermediates.cpp"
#undef main

int main(int argc, char ** argv) {
    try {
        if (argc != 2) throw std::runtime_error("private output path required");
        ggml_init_params params{1024 * 1024, nullptr, false};
        auto * ctx = ggml_init(params);
        if (!ctx) throw std::runtime_error("GGML context initialization failed");
        auto * logits_input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
        auto * topk_input = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2);
        auto * weights_input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
        auto * unrelated = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        ggml_set_name(unrelated, "attn_out-0");
        const float logits_data[] = {-2.0f, 0.0f, 1.0f, 4.0f};
        const int32_t topk_data[] = {1, 3};
        const float weights_data[] = {0.25f, 0.75f};
        for (int i = 0; i < 4; ++i) ggml_set_f32_1d(logits_input, i, logits_data[i]);
        for (int i = 0; i < 2; ++i) ggml_set_i32_1d(topk_input, i, topk_data[i]);
        for (int i = 0; i < 2; ++i) ggml_set_f32_1d(weights_input, i, weights_data[i]);

        auto * logits = ggml_scale(ctx, logits_input, 1.0f);
        auto * topk = ggml_dup(ctx, topk_input);
        auto * weights = ggml_scale(ctx, weights_input, 1.0f);
        ggml_set_name(logits, "ffn_moe_logits-0");
        ggml_set_name(topk, "ffn_moe_topk-0");
        ggml_set_name(weights, "ffn_moe_weights-0");
        ggml_set_output(logits);
        ggml_set_output(topk);
        ggml_set_output(weights);
        auto * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, logits);
        ggml_build_forward_expand(graph, topk);
        ggml_build_forward_expand(graph, weights);
        auto * backend = ggml_backend_cpu_init();
        if (!backend) throw std::runtime_error("CPU backend initialization failed");
        ggml_backend_t backends[] = {backend};
        ggml_backend_buffer_type_t bufts[] = {ggml_backend_get_default_buffer_type(backend)};
        auto * sched = ggml_backend_sched_new(backends, bufts, 1, 128, false, true);
        if (!sched) throw std::runtime_error("backend scheduler initialization failed");

        fd_owner file(open(argv[1], O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600));
        observer state;
        state.fd = file.value;
        write_exact(state.fd, "RVMOE001", 8);
        state.position = 0;
        ggml_backend_sched_set_eval_callback(sched, observe_tensor, &state);
        if (ggml_backend_sched_graph_compute(sched, graph) != GGML_STATUS_SUCCESS || state.failed)
            throw std::runtime_error("scheduler callback graph execution failed");
        if (observe_tensor(unrelated, true, &state) || observe_tensor(unrelated, false, &state))
            throw std::runtime_error("unselected tensor was captured");
        state.position = 1;
        if (observe_tensor(logits, true, &state) || observe_tensor(logits, false, &state))
            throw std::runtime_error("nonzero position was captured");
        state.position = 0;
        if (state.records != 3 || fsync(file.value)) throw std::runtime_error("callback record count/flush mismatch");
        ggml_backend_sched_free(sched);
        ggml_backend_free(backend);
        ggml_free(ctx);
        ggml_init_params bad_params{1024, nullptr, true};
        auto * bad_ctx = ggml_init(bad_params);
        auto * bad = ggml_new_tensor_1d(bad_ctx, GGML_TYPE_F32, 1);
        ggml_set_name(bad, "ffn_moe_probs-0");
        auto * bad_buffer = ggml_backend_alloc_ctx_tensors_from_buft(bad_ctx, ggml_backend_cpu_buffer_type());
        if (!bad_buffer) throw std::runtime_error("nonfinite tensor allocation failed");
        const float inf = INFINITY;
        ggml_backend_tensor_set(bad, &inf, 0, sizeof(inf));
        observer invalid;
        invalid.fd = file.value;
        invalid.position = 0;
        if (!observe_tensor(bad, true, &invalid) || observe_tensor(bad, false, &invalid) || !invalid.failed)
            throw std::runtime_error("nonfinite tensor was not rejected");
        ggml_backend_buffer_free(bad_buffer);
        ggml_free(bad_ctx);
        puts("Native observer smoke check passed; no model loaded and no inference run.");
        return 0;
    } catch (const std::exception &) {
        fputs("Native observer smoke check failed.\n", stderr);
        return 1;
    }
}
