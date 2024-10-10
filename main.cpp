#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>
#include <torch/torch.h>
#include "ctc_decoder.h"
#include "ctc_decode_C.h"
#include <torch/torch.h>
#include <iostream>
#include <chrono>

using namespace std::chrono;

// Function to benchmark [] operator access
void benchmark_indexing(torch::Tensor tensor) {
    auto start = high_resolution_clock::now();
    auto sum = torch::tensor(0, torch::kFloat32);
    for (int i = 0; i < tensor.size(0); ++i) {
        for (int j = 0; j < tensor.size(1); ++j) {
            sum += tensor[i][j]; // Using [] operator
        }
    }
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start).count();
    std::cout << "[] operator access: " << duration << "us, sum: " << sum << std::endl;
}

// Function to benchmark .item<T>() access
void benchmark_item(torch::Tensor tensor) {
    auto start = high_resolution_clock::now();
    float sum = 0.0;
    for (int i = 0; i < tensor.size(0); ++i) {
        for (int j = 0; j < tensor.size(1); ++j) {
            sum += tensor.index({i, j}).item<float>(); // Using item()
        }
    }
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start).count();
    std::cout << ".item<T>() access: " << duration << "us, sum: " << sum << std::endl;
}

// Function to benchmark .data_ptr<T>() access
void benchmark_data_ptr(torch::Tensor tensor) {
    int n = tensor.size(0), m = tensor.size(1);
    auto start = high_resolution_clock::now();
    float sum = 0.0;
    float* data_ptr = tensor.data_ptr<float>(); // Get pointer to data
    int64_t size = tensor.numel(); // Total number of elements
//    for (int64_t i = 0; i < size; ++i) {
//        sum += data_ptr[i]; // Accessing using pointer arithmetic
//    }
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < m; j++) {
            sum += data_ptr[i * m + j];
        }
    }
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start).count();
    std::cout << ".data_ptr<T>() access: " << duration << "us, sum: " << sum << std::endl;
}


int main() {
    int64_t N = 1, T = 1200, C = 5;
//    torch::manual_seed(0);
    torch::Tensor probs = torch::rand({T, N, C},torch::kFloat32);
    at::Tensor sum = torch::sum(probs, 2, true);
    probs = probs / sum;
    probs = probs.log_softmax(2);
    torch::save(probs, "./test.pt");
//    std::cout << probs << std::endl; // normalized probs
    auto seqs = torch::zeros({N, T}, torch::kUInt8);
    auto moves = torch::zeros({N, T}, torch::kUInt8);
    auto quals = torch::zeros({N, T}, torch::kUInt8);

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    ctc_veterbi_decode(
            probs.data_ptr<float>(),
            seqs.data_ptr<uint8_t>(),
            moves.data_ptr<uint8_t>(),
            quals.data_ptr<uint8_t>(),
            T, N, C
            );

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::cout << "ctc_decode_C: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << "us" << std::endl;
    for (int i = 0; i < T; i++) {
        int num = seqs[0][i].item<uint8_t>();
        if (num)
            std::cout << num << " ";
    }

    float score = 0;
    for (int i = 0; i < T; i++) {
        int idx = seqs[0][i].item<uint8_t>() ;
        score += probs[i][0][idx].item<float>();
    }

    std::cout << "score: " << score << std::endl;

    std::vector<std::vector<float>> log_probs;
    for(size_t i = 0; i < T; i++) {
        log_probs.push_back({});
        for (size_t j = 0; j < C; j++) {
            log_probs[i].push_back(probs[i][0][j].item<float>());
        }
    }

    std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();
    auto decoder = Yao::CTC_Prefix_BeamSearch();
    decoder.search(log_probs, 1);
    std::chrono::high_resolution_clock::time_point t4 = std::chrono::high_resolution_clock::now();
    std::cout << "ctc_decode_C: " << std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count() << "us" << std::endl;




    decoder.display_hypo();

    return 0;
}