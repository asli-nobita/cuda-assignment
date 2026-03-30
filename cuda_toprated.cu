#include <stdio.h>
#include <stdlib.h> 
#include <cuda.h> 
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <chrono>
#include "simdjson.h"

using namespace simdjson; 

std::unordered_map<std::string, int> asin_to_id; 
std::vector<std::string> id_to_asin;
std::vector<int> h_movie_ids; 
std::vector<float> h_ratings; 

int unique_id_count = 0; 

void parse_data(const std::string& filename) { 
    ondemand::parser p; 
    padded_string json = padded_string::load(filename); 
    
    ondemand::document_stream stream = p.iterate_many(json);  
    
    for (auto doc : stream) { 
        std::string_view m_id_view = doc["asin"].get_string(); 
        float m_rating = static_cast<float>(doc["overall"].get_double()); 
        
        std::string m_id(m_id_view); 

        if(asin_to_id.find(m_id) == asin_to_id.end()) { 
            asin_to_id[m_id] = unique_id_count; 
            id_to_asin.push_back(m_id); 
            unique_id_count++; 
        }
        h_movie_ids.push_back(asin_to_id[m_id]); 
        h_ratings.push_back(m_rating); 
    }
}

__global__ void aggregate_ratings(int* movie_ids, float* ratings, float* sums, int* counts, int num_records) { 
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_records) {
        int m_id = movie_ids[idx];
        float m_rating = ratings[idx];
        
        atomicAdd(&sums[m_id], m_rating);
        atomicAdd(&counts[m_id], 1);
    }
}

__global__ void calculate_averages(float* sums, int* counts, float* avgs, int num_movies) {
    int m_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (m_id < num_movies && counts[m_id] > 0) {
        avgs[m_id] = sums[m_id] / (float)counts[m_id];
    }
}

int main() { 
    std::string filename = "reviews_Movies_and_TV_5.json/Movies_and_TV_5.json"; 
    
    std::cout << "Parsing JSON..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now(); 
    parse_data(filename);  
    auto end_time = std::chrono::high_resolution_clock::now(); 

    std::chrono::duration<double, std::milli> duration = end_time - start_time;
    std::cout << "CPU Execution Time: " << duration.count() << " milliseconds\n";
    
    const int NUM_RECORDS = h_movie_ids.size(); 
    const int NUM_MOVIES = unique_id_count;
    std::cout << "Number of reviews: " << NUM_RECORDS << "\n"; 
    std::cout << "Unique movies: " << NUM_MOVIES << "\n"; 

    int *d_movie_ids, *d_counts, *d_unique_ids;
    float *d_ratings, *d_sums, *d_avgs;

    cudaMalloc(&d_movie_ids, NUM_RECORDS * sizeof(int)); 
    cudaMalloc(&d_ratings, NUM_RECORDS * sizeof(float)); 
    cudaMalloc(&d_sums, NUM_MOVIES * sizeof(float)); 
    cudaMalloc(&d_avgs, NUM_MOVIES * sizeof(float)); 
    cudaMalloc(&d_counts, NUM_MOVIES * sizeof(int)); 
    cudaMalloc(&d_unique_ids, NUM_MOVIES * sizeof(int));

    cudaMemcpy(d_movie_ids, h_movie_ids.data(), NUM_RECORDS * sizeof(int), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_ratings, h_ratings.data(), NUM_RECORDS * sizeof(float), cudaMemcpyHostToDevice); 
    
    cudaMemset(d_sums, 0, NUM_MOVIES * sizeof(float)); 
    cudaMemset(d_avgs, 0, NUM_MOVIES * sizeof(float)); 
    cudaMemset(d_counts, 0, NUM_MOVIES * sizeof(int)); 

    // this creates a array of unique movie ids [0,1,2...NUM_MOVIES-1], we need this for sorting 
    thrust::sequence(thrust::device, d_unique_ids, d_unique_ids + NUM_MOVIES);

    int threadsPerBlock = 256;
    int blocksRecords = (NUM_RECORDS + threadsPerBlock - 1) / threadsPerBlock;
    int blocksMovies = (NUM_MOVIES + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start); 

    std::cout << "Running Aggregation..." << std::endl;
    aggregate_ratings<<<blocksRecords, threadsPerBlock>>>(d_movie_ids, d_ratings, d_sums, d_counts, NUM_RECORDS); 
    cudaDeviceSynchronize(); 

    std::cout << "Calculating Averages..." << std::endl;
    calculate_averages<<<blocksMovies, threadsPerBlock>>>(d_sums, d_counts, d_avgs, NUM_MOVIES); 
    cudaDeviceSynchronize(); 

    std::cout << "Sorting Results..." << std::endl;
    // this performs parallel radix sort on the GPU. pretty fast!
    thrust::sort_by_key(thrust::device, d_avgs, d_avgs + NUM_MOVIES, d_unique_ids, thrust::greater<float>()); 

    cudaEventRecord(stop); 
    cudaEventSynchronize(stop); 
    float time = 0.0; 
    cudaEventElapsedTime(&time, start, stop); 
    std::cout << "GPU Kernel Execution Time: " << time << " milliseconds" << "\n"; 

    int top_n = 10;
    std::vector<int> top_ids(top_n);
    std::vector<float> top_avgs(top_n);

    cudaMemcpy(top_ids.data(), d_unique_ids, top_n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(top_avgs.data(), d_avgs, top_n * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "\n--- TOP " << top_n << " MOVIES ---\n";
    for(int i = 0; i < top_n; i++) {
        std::string asin = id_to_asin[top_ids[i]];
        std::cout << i + 1 << ". ASIN: " << asin << " | Avg Rating: " << top_avgs[i] << "\n";
    }

    cudaEventDestroy(start); 
    cudaEventDestroy(stop); 
    cudaFree(d_movie_ids); cudaFree(d_ratings);
    cudaFree(d_sums); cudaFree(d_avgs); cudaFree(d_counts); cudaFree(d_unique_ids);

    // the sequential json parsing is the bottleneck here as expected, 
    // the difference is quite astronomical tho, GPU execution runs almost 17k times faster 

    return 0;
}