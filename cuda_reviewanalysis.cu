#include <stdio.h>
#include <stdlib.h> 
#include <cuda.h> 
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include "simdjson.h"

using namespace simdjson;

// --- CPU GLOBALS ---
std::vector<char> reviews;
std::vector<int> offsets;

__host__ __device__ uint32_t hash_word(const char* word) {
    uint32_t hash = 2166136261u; // FNV offset basis
    int i = 0;
    while (word[i] != '\0') {
        hash ^= (uint8_t)word[i];
        hash *= 16777619u; // FNV prime
        i++;
    }
    return hash;
}

const int EMPTY_KEY = 0;
const int TABLE_SIZE = 16384; // Must be larger than lexicon, ideally power of 2

std::vector<uint32_t> h_lexicon_keys(TABLE_SIZE, EMPTY_KEY);
std::vector<float> h_lexicon_scores(TABLE_SIZE, 0.0f);

void build_flat_lexicon() {
    FILE* fp = fopen("lexicon_scores.txt", "r");
    if (!fp) {
        printf("Error opening lexicon.\n");
        return;
    }

    char line[256];
    char word[64];
    float score;
    int words_loaded = 0;

    while (fgets(line, sizeof(line), fp)) {
        if (sscanf(line, "%63s %f", word, &score) == 2) {
            uint32_t hash = hash_word(word);
            uint32_t slot = hash % TABLE_SIZE;

            while (h_lexicon_keys[slot] != EMPTY_KEY) {
                slot = (slot + 1) % TABLE_SIZE;
            }

            h_lexicon_keys[slot] = hash;
            h_lexicon_scores[slot] = score;
            words_loaded++;
        }
    }
    fclose(fp);
    std::cout << "Successfully loaded " << words_loaded << " words into Lexicon.\n";
}

__device__ float device_get_score(const char* word, uint32_t* keys, float* scores, int table_size) {
    uint32_t hash = hash_word(word);

    uint32_t slot = hash % table_size;

    while (true) {
        if (keys[slot] == hash) {
            return scores[slot];
        }

        if (keys[slot] == EMPTY_KEY) {
            return 0.0f;
        }

        slot = (slot + 1) % table_size;
    }
}

void parse_data(const std::string& filename) {
    ondemand::parser p;
    padded_string json = padded_string::load(filename);
    ondemand::document_stream stream = p.iterate_many(json);

    for (auto doc : stream) {
        std::string_view review = doc["reviewText"].get_string();
        offsets.push_back(reviews.size());
        reviews.insert(reviews.end(), review.begin(), review.end());
    }
    offsets.push_back(reviews.size());
}

// --- KERNEL ---
__global__ void calculate_scores(char* d_reviews, int* d_offsets, int* d_review_scores,
    int num_reviews, uint32_t* keys, float* scores, int table_size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_reviews) {
        int start = d_offsets[idx];
        int end = d_offsets[idx + 1];

        char word[64];
        float mean_score = 0.0f;
        int word_idx = 0;

        for (int i = start; i < end; i++) {
            char c = d_reviews[i];

            bool is_letter = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c == '\'');

            if (is_letter) {
                if (word_idx < 63) {
                    if (c >= 'A' && c <= 'Z') {
                        c += 32; // Convert to lowercase
                    }
                    word[word_idx++] = c;
                }
            }
            else {
                if (word_idx > 0) {
                    word[word_idx] = '\0';
                    mean_score += device_get_score(word, keys, scores, table_size);
                    word_idx = 0; 
                }
            }
        }

        if (word_idx > 0) {
            word[word_idx] = '\0';
            mean_score += device_get_score(word, keys, scores, table_size);
        }

        d_review_scores[idx] = (mean_score > 0.0f) ? 1 : 0;
    }
}

int main() {
    std::string filename = "reviews_Movies_and_TV_5.json/Movies_and_TV_5.json";

    auto start_time = std::chrono::high_resolution_clock::now();
    std::cout << "Parsing JSON..." << std::endl;
    parse_data(filename);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;
    std::cout << "CPU Parsing Time: " << duration.count() << " ms\n";

    std::cout << "Building lexicon map...\n";
    start_time = std::chrono::high_resolution_clock::now();
    build_flat_lexicon();
    end_time = std::chrono::high_resolution_clock::now();
    duration = end_time - start_time;
    std::cout << "Lexicon Build Time: " << duration.count() << " ms\n";

    int num_characters = reviews.size();
    int num_reviews = offsets.size() - 1;

    std::cout << "Characters to process: " << num_characters << "\n";
    std::cout << "Reviews to process: " << num_reviews << "\n";

    // --- GPU Pointers ---
    char* d_reviews;
    int* d_offsets;
    int* d_review_scores;
    std::vector<int> h_review_scores(num_reviews);
    uint32_t* d_lexicon_keys;
    float* d_lexicon_scores;

    cudaMalloc(&d_lexicon_keys, TABLE_SIZE * sizeof(uint32_t));
    cudaMalloc(&d_lexicon_scores, TABLE_SIZE * sizeof(float));

    cudaMemcpy(d_lexicon_keys, h_lexicon_keys.data(), TABLE_SIZE * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lexicon_scores, h_lexicon_scores.data(), TABLE_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_reviews, num_characters * sizeof(char));
    cudaMalloc(&d_offsets, offsets.size() * sizeof(int));
    cudaMalloc(&d_review_scores, num_reviews * sizeof(int));

    // --- Transfer Data ---
    cudaMemcpy(d_reviews, reviews.data(), num_characters * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (num_reviews + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::cout << "Running GPU Sentiment Analysis..." << std::endl;
    cudaEventRecord(start);

    calculate_scores << <blocks, threadsPerBlock >> > (d_reviews, d_offsets, d_review_scores, num_reviews, d_lexicon_keys, d_lexicon_scores, TABLE_SIZE);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time = 0.0;
    cudaEventElapsedTime(&time, start, stop);
    std::cout << "GPU Kernel Execution Time: " << time << " ms\n";

    cudaMemcpy(h_review_scores.data(), d_review_scores, num_reviews * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "\nSENTIMENT SCORES (First 10): -----\n";
    for (int i = 0; i < std::min(10, num_reviews); i++) {
        std::cout << "Review #" << i + 1 << " is " << h_review_scores[i] << "\n";
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_review_scores); cudaFree(d_offsets); cudaFree(d_reviews);

    return 0;
}