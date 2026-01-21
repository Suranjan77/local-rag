#include "VectorStore.hpp"
#include "LlamaEngine.hpp"
#include <iostream>
#include <print>
#include <string>

int main() {
    // Paths
    const std::string model_embed = "models/nomic-embed.gguf";
    const std::string model_chat = "models/llama-3.gguf";
    const std::string vec_bin = "data/vectors.bin";
    const std::string chunk_json = "data/chunks.json";

    try {
        // 1. Initialize Components
        LlamaEngine engine(model_embed, model_chat);
        VectorStore store(vec_bin, chunk_json);

        std::println("\n=== High-Performance RAG Agent Ready ===");
        std::println("Type 'quit' to exit.\n");

        // 2. Chat Loop
        std::string query;
        while (true) {
            std::print("> ");
            std::fflush(stdout); // Ensure prompt shows up

            if (!std::getline(std::cin, query) || query == "quit") break;
            if (query.empty()) continue;

            // A. Retrieval
            std::println("[Retrieving context...]");
            auto q_vec = engine.embed(query);
            std::string context = store.search(q_vec);

            // B. Prompt Construction (Llama 3 Format)
            std::string prompt =
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                "Use the following context to answer the user question.\n" + context +
                "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" +
                query + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";

            // C. Generation
            engine.chat(prompt, [](const std::string& token) {
                std::print("{}", token);
                std::fflush(stdout);
                });
            std::println(""); // Newline after generation
        }

    }
    catch (const std::exception& e) {
        std::println(stderr, "Error: {}", e.what());
        return 1;
    }

    return 0;
}
