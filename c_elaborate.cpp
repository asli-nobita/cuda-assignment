// (e) Implement a sequential C/C++ program c_elaborate.c to identify reviewers who have
// written elaborate reviews. Elaborate reviewers are defined as len(review text) >= 50
// words and at least 5 such reviews. The program should output the reviewer IDs and
// their review count and average review length.

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

struct Reviewer { 
    std::string reviewerID; 
    int total_count; 
    int total_review_len; 
    int elaborate_count; 

    Reviewer() = default; 

    Reviewer(std::string& rID, int tc, int trl, int ec) : reviewerID(rID), total_count(tc), total_review_len(trl), elaborate_count(ec) {}
};

std::unordered_map<std::string, Reviewer> reviewers;

void parse_data(const std::string& filename) {
    ondemand::parser p;
    padded_string json = padded_string::load(filename);
    ondemand::document_stream stream = p.iterate_many(json);

    for (auto doc : stream) {
        std::string_view rID_view = doc["reviewerID"].get_string(); 
        std::string_view rText_view = doc["reviewText"].get_string(); 
        std::string reviewerID(rID_view.begin(), rID_view.end()); 
        std::string reviewText(rText_view.begin(), rText_view.end()); 
        int len = reviewText.length();
        int tc = 0, trl = 0, ec = 0; 
        if (reviewers.count(reviewerID)) { 
            auto rv = reviewers[reviewerID]; 
            tc = rv.total_count, trl = rv.total_review_len, ec = rv.elaborate_count; 
            if (len >= 50) ec++; 
        } 
        reviewers[reviewerID] = Reviewer(reviewerID, tc + 1, trl + len, ec); 
    }
}

int main() { 
    const std::string filename = "reviews_Movies_and_TV_5.json/Movies_and_TV_5.json";

    auto start_time = std::chrono::high_resolution_clock::now();
    std::cout << "Parsing JSON..." << std::endl;
    parse_data(filename);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;
    std::cout << "CPU Parsing Time: " << duration.count() << " ms\n";

    std::vector<Reviewer> vec; 
    for (auto& [rID, rv] : reviewers) { 
        if (rv.elaborate_count >= 5) vec.push_back(rv); 
    } 
    sort(vec.begin(), vec.end(), [](auto& a, auto& b) { 
        return a.elaborate_count > b.elaborate_count; 
    });
    std::cout << "Listing top 20 elaborate reviewers (at least 5 reviews of length >= 50):" << "\n"; 
    for (int i = 0; i < 20; i++) { 
        auto& rv = vec[i]; 
        std::cout << "Reviewer ID: " << rv.reviewerID << " | Total reviews: " << rv.total_count << " | Average review length: " << (double)rv.total_review_len / rv.total_count << " | Elaborate reviews: " << rv.elaborate_count << "\n"; 
    }
}