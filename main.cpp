#include <vector>
#include <iostream>
#include <torch/torch.h>
#include "ctc_decoder.h"
#include "ctc_decode_C.h"
#include <chrono>

int main() {
    int64_t N = 10, T = 3000, C = 5;
    torch::manual_seed(0);
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

    size_t beam_size = 10;
    auto decoder = Yao::CTC_Prefix_BeamSearch();

    std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();
    decoder.search(probs, beam_size);
    std::chrono::high_resolution_clock::time_point t4 = std::chrono::high_resolution_clock::now();
    std::cout << "ctc_decode_C: " << std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count() << "us" << std::endl;
    decoder.display_hypo();
//    decoder.clear();

    return 0;
}