#pragma once
#include "llama.h"
#include <string>
#include <vector>
#include <functional>
#include <print>

// RAII Wrapper for Llama Objects
struct LlamaContextDeleter {
    void operator()(llama_context* ctx) { llama_free(ctx); }
};
struct LlamaModelDeleter {
    void operator()(llama_model* model) { llama_free_model(model); }
};
struct LlamaBatchDeleter {
    void operator()(llama_batch* batch) { llama_batch_free(*batch); delete batch; }
};

using LlamaCtxPtr = std::unique_ptr<llama_context, LlamaContextDeleter>;
using LlamaModelPtr = std::unique_ptr<llama_model, LlamaModelDeleter>;
using LlamaBatchPtr = std::unique_ptr<llama_batch, LlamaBatchDeleter>;

class LlamaEngine {
    LlamaModelPtr embed_model;
    LlamaCtxPtr embed_ctx;

    LlamaModelPtr chat_model;
    LlamaCtxPtr chat_ctx;

    // Batch for chat generation
    llama_batch batch;

public:
    LlamaEngine(const std::string& embed_path, const std::string& chat_path) {
        llama_backend_init();

        // 1. Load Embedding Model
        auto mparams = llama_model_default_params();
        mparams.n_gpu_layers = 99; // Offload all
        embed_model.reset(llama_load_model_from_file(embed_path.c_str(), mparams));

        auto cparams = llama_context_default_params();
        cparams.embeddings = true;
        cparams.n_ctx = 2048;
        embed_ctx.reset(llama_new_context_with_model(embed_model.get(), cparams));

        // 2. Load Chat Model
        mparams.n_gpu_layers = 99;
        chat_model.reset(llama_load_model_from_file(chat_path.c_str(), mparams));

        cparams = llama_context_default_params();
        cparams.n_ctx = 8192; // Larger context for RAG
        chat_ctx.reset(llama_new_context_with_model(chat_model.get(), cparams));

        // Initialize batch
        batch = llama_batch_init(8192, 0, 1);
    }

    ~LlamaEngine() {
        llama_batch_free(batch);
        llama_backend_free();
    }

    std::vector<float> embed(const std::string& text) {
        std::string query = "search_query: " + text; // Nomic specific prefix

        // Tokenize
        std::vector<llama_token> tokens(query.length() + 2);
        int n = llama_tokenize(embed_model.get(), query.c_str(), query.length(), tokens.data(), tokens.size(), true, false);
        tokens.resize(n);

        llama_batch ebatch = llama_batch_get_one(tokens.data(), n, 0, 0);

        if (llama_decode(embed_ctx.get(), ebatch) != 0) {
            std::println(stderr, "Embedding decode failed");
            return {};
        }

        // Get last token embedding
        const float* emb_ptr = llama_get_embeddings_seq(embed_ctx.get(), 0);
        int dim = llama_n_embd(embed_model.get());

        // Normalize
        std::vector<float> vec(emb_ptr, emb_ptr + dim);
        float norm = 0;
        for (float f : vec) norm += f * f;
        norm = std::sqrt(norm);
        for (float& f : vec) f /= norm;

        return vec;
    }

    void chat(const std::string& prompt, std::function<void(const std::string&)> callback) {
        // Tokenize Prompt
        std::vector<llama_token> tokens(prompt.length() + 100);
        int n = llama_tokenize(chat_model.get(), prompt.c_str(), prompt.length(), tokens.data(), tokens.size(), true, false);
        tokens.resize(n);

        // Prepare Batch
        // Clear KV cache for new turn (simplified for single-turn RAG)
        llama_kv_cache_clear(chat_ctx.get());

        batch.n_tokens = 0;
        for (int i = 0; i < n; i++) {
            llama_batch_add(batch, tokens[i], i, { 0 }, false);
        }
        // Force logits on last token
        batch.logits[batch.n_tokens - 1] = true;

        if (llama_decode(chat_ctx.get(), batch) != 0) return;

        // Generation Loop
        int n_cur = batch.n_tokens;
        int n_vocab = llama_n_vocab(chat_model.get());

        while (n_cur < 8192) {
            // Sample
            auto* logits = llama_get_logits_ith(chat_ctx.get(), batch.n_tokens - 1);
            llama_token_data_array candidates_p = { new llama_token_data[n_vocab], (size_t)n_vocab, false };
            for (int i = 0; i < n_vocab; i++) candidates_p.data[i] = { (llama_token)i, logits[i], 0.0f };

            llama_token new_token_id = llama_sample_token_greedy(chat_ctx.get(), &candidates_p);
            delete[] candidates_p.data;

            // Is EOG?
            if (llama_token_is_eog(chat_model.get(), new_token_id)) break;

            // Print
            char buf[256];
            int n_chars = llama_token_to_piece(chat_model.get(), new_token_id, buf, sizeof(buf), 0, true);
            if (n_chars < 0) n_chars = 0;
            std::string piece(buf, n_chars);
            callback(piece);

            // Prepare next token
            llama_batch_clear(batch);
            llama_batch_add(batch, new_token_id, n_cur, { 0 }, true);
            n_cur++;

            if (llama_decode(chat_ctx.get(), batch) != 0) break;
        }
    }
};