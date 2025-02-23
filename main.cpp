#include <vector>
#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/jit.h>
#include <torch/csrc/jit/serialization/import.h>
#include "ctc_decoder.h"
#include "ctc_decode_C.h"
#include "3rdParty/cnpy.h"
#include <chrono>

int main() {
    torch::manual_seed(0);
    torch::Tensor probs;
    cnpy::NpyArray arr = cnpy::npy_load("/tmp/tmp.kFZTrjbubY/test.npy");
    probs = torch::from_blob(arr.data_holder->data(), {(int64_t)arr.shape[0], (int64_t)arr.shape[1], (int64_t)arr.shape[2]}, torch::kFloat32);
    int T = probs.size(0), N = probs.size(1), C = probs.size(2);
    auto seqs = torch::zeros({N, T}, torch::kUInt8);
    auto moves = torch::zeros({N, T}, torch::kUInt8);
    auto quals = torch::zeros({N, T}, torch::kUInt8);

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    ctc_greedy_decode(
            probs.data_ptr<float>(),
            seqs.data_ptr<uint8_t>(),
            moves.data_ptr<uint8_t>(),
            quals.data_ptr<uint8_t>(),
            T, N, C
            );

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::cout << "ctc_decode_C: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms" << std::endl;
    for (int i = 0; i < T; i++) {
        int num = seqs[N - 1][i].item<uint8_t>();
        if (num)
        std::cout << num << " ";
    }std::cout << std::endl;
    for (int i = 0; i < T; i++) {
        int num = moves[N - 1][i].item<uint8_t>();
        std::cout << num << "\t";
    }std::cout << std::endl;
    for (int i = 0; i < T; i++) {
        int num = quals[N - 1][i].item<uint8_t>();
        std::cout << num << "\t";
    }std::cout << std::endl;

    float score = 0;
    for (int i = 0; i < T; i++) {
        int idx = seqs[N - 1][i].item<uint8_t>() ;
        score += probs[i][N - 1][idx].item<float>();
    }

    std::cout << "score: " << score << std::endl;

    size_t beam_size = 5;
    auto decoder = Yao::CTC_Prefix_BeamSearch();

    std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();
    auto [seqs_2, moves_2, quals_2] = Yao::ctc_prefix_beam_Search(probs, beam_size);
    std::chrono::high_resolution_clock::time_point t4 = std::chrono::high_resolution_clock::now();
    std::cout << "ctc_decode_C: " << std::chrono::duration_cast<std::chrono::milliseconds >(t4 - t3).count() << "ms" << std::endl;

    std::cout << static_cast<const void *>(seqs.data_ptr<uint8_t>()) << std::endl;


    for (int i = 0; i < T; i++) {
        int num = seqs_2[N - 1][i].item<uint8_t>();
        if (num)
            std::cout << num << " ";
    }std::cout << std::endl;
    for (int i = 0; i < T; i++) {
        int num = moves_2[N - 1][i].item<uint8_t>();
        std::cout << num << "\t";
    }std::cout << std::endl;
    for (int i = 0; i < T; i++) {
        int num = quals_2[N - 1][i].item<uint8_t>();
        std::cout << num << "\t";
    }std::cout << std::endl;

    return 0;
}