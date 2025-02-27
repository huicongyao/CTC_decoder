//
// Created by yaohc on 2024/9/26.
//

#ifndef CTC_DECODER_CTC_DECODER_H
#define CTC_DECODER_CTC_DECODER_H

#include <iostream>
#include <torch/torch.h>
#include <cstring>
#include <cmath>
#include "ctc_utils.h"
#include "ctc_decode_C.h"

namespace Yao {
    struct PrefixScore {
        float s = -std::numeric_limits<float>::max();       // blank ending score
        float ns = -std::numeric_limits<float>::max();      // none blank ending score
        float v_s = -std::numeric_limits<float>::max();
        float v_ns = -std::numeric_limits<float>::max();
        float curr_token_prob = -std::numeric_limits<float>::max();
        std::vector<int> times_s;
        std::vector<int> times_ns;
        float score() const {
            return Yao::utils::log_add(s, ns);
        }
        float viterbi_score() const {
            return v_s > v_ns ? v_s : v_ns;
        }
        const std::vector<int>& times() const {
            return v_s > v_ns ? times_s : times_ns;
        }

    };

    std::unordered_map<std::vector<int>, PrefixScore, Yao::utils::PrefixHash> get_empty_hypo();
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
            ctc_prefix_beam_Search(const torch::Tensor & logp, size_t beam_size = 32, size_t num_threads = 8);

    static bool PrefixScoreCompare(
            const std::pair<std::vector<int>, PrefixScore> & a,
            const std::pair<std::vector<int>, PrefixScore> & b) {
        return a.second.score() > b.second.score();
    }

    class CTC_Prefix_BeamSearch {
    public:
        CTC_Prefix_BeamSearch();
        void search(const torch::Tensor & logp, size_t beam_size = 32);
        void display_hypo() const;
        void clear();
    private:
        std::vector<std::pair<std::vector<int>, PrefixScore>> batch_hypo;
        std::vector<std::pair<std::vector<int>, PrefixScore>> hypotheses_;
        std::unordered_map<std::vector<int>, Yao::PrefixScore, Yao::utils::PrefixHash> curr_hypo_;
    };

}

#endif //CTC_DECODER_CTC_DECODER_H
