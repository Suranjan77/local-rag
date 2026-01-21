#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <span>
#include <print>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#include "hnswlib/hnswlib.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class VectorStore {
    struct MappedFile {
        void* data = nullptr;
        size_t size = 0;
        int fd = -1;

        MappedFile(const std::string& path) {
            fd = open(path.c_str(), O_RDONLY);
            if (fd == -1) throw std::runtime_error("Failed to open vector file");
            size = lseek(fd, 0, SEEK_END);
            data = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        }

        ~MappedFile() {
            if (data) munmap(data, size);
            if (fd != -1) close(fd);
        }
    };

    std::unique_ptr<MappedFile> raw_vectors;
    std::unique_ptr<hnswlib::L2Space> space;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> alg_hnsw;
    json chunks_metadata;
    int dim = 0;
    int max_elements = 0;

public:
    VectorStore(const std::string& vec_path, const std::string& json_path) {
        std::println("Loading Vector Store...");

        // 1. Load JSON Metadata
        std::ifstream f(json_path);
        chunks_metadata = json::parse(f);

        // 2. Memory Map Binary Vectors (Zero-Copy Load)
        raw_vectors = std::make_unique<MappedFile>(vec_path);

        // Read Header: [rows (int32), dim (int32)]
        int32_t* header = static_cast<int32_t*>(raw_vectors->data);
        max_elements = header[0];
        dim = header[1];

        float* vector_data = reinterpret_cast<float*>(header + 2);

        // 3. Initialize HNSW Index
        space = std::make_unique<hnswlib::L2Space>(dim);
        alg_hnsw = std::make_unique<hnswlib::HierarchicalNSW<float>>(space.get(), max_elements);

        // Bulk add points (This is fast in memory)
        std::println("Indexing {} vectors...", max_elements);
        // Parallel addition
#pragma omp parallel for
        for (int i = 0; i < max_elements; i++) {
            alg_hnsw->addPoint(vector_data + (i * dim), i);
        }
    }

    // Returns formatted context string
    std::string search(std::span<const float> query_vec, int k = 3) {
        auto pq = alg_hnsw->searchKnn(query_vec.data(), k);
        std::string context;

        while (!pq.empty()) {
            auto item = pq.top();
            // JSON structure: [{"text": "...", "source": "..."}]
            std::string text = chunks_metadata[item.second]["text"];
            context = "Document Fragment:\n" + text + "\n---\n" + context;
            pq.pop();
        }
        return context;
    }
};
