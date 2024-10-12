//
// Created by yaohc on 9/28/24.
//

#include "ctc_decoder.h"

// search for a batch of CTC inputs
void Yao::CTC_Prefix_BeamSearch::search(const torch::Tensor &logp, size_t beam_size) {

    if (logp.size(0) == 0) return;
    size_t T = logp.size(0), N = logp.size(1), C = logp.size(2);
    torch::Tensor logp_ = logp.permute({1, 0, 2}).contiguous();
    for (size_t n = 0; n < N; ++n){
        const float *logp_t = logp_[n][0].data_ptr<float>(); // use raw pointer to accelerate element access speed
        for (size_t t = 0; t < T; ++t) {
            std::unordered_map<std::vector<int>, Yao::PrefixScore, Yao::utils::PrefixHash> next_hyps;
            // 1. first beam prune, only select topk candidates, pass this part and implement it in the furture
            // 2. Toke passing
            for (size_t id = 0; id < C; id++) {
                auto prob = logp_t[t * C + id];
                for (const auto &it: curr_hypo_) {
                    const std::vector<int> &prefix = it.first;
                    const Yao::PrefixScore &prefix_score = it.second;
                    if (id == 0) {
                        // Case 0: *a + ε => *a
                        PrefixScore &next_score = next_hyps[prefix];
                        next_score.s = Yao::utils::log_add(next_score.s, prefix_score.score() + prob);
                    } else if (!prefix.empty() && id == prefix.back()) {
                        // Case 1: *a + a => *a
                        PrefixScore &next_score = next_hyps[prefix];
                        next_score.ns = Yao::utils::log_add(next_score.ns, prefix_score.ns + prob);
                        // Case 2: *aε + a => *aa
                        std::vector<int> new_prefix(prefix);
                        new_prefix.emplace_back(id);
                        PrefixScore &next_score2 = next_hyps[new_prefix];
                        next_score2.ns = Yao::utils::log_add(next_score2.ns, prefix_score.s + prob);
                    } else {
                        // Case 3: *a + b => *ab, *aε + b => *ab
                        std::vector<int> new_prefix(prefix);
                        new_prefix.emplace_back(id);
                        PrefixScore &next_score = next_hyps[new_prefix];
                        next_score.ns = Yao::utils::log_add(next_score.ns, prefix_score.score() + prob);
                    }
                }
            }
            // 3. beam prune, only keep top n best paths
            std::vector<std::pair<std::vector<int>, Yao::PrefixScore>> arr(next_hyps.begin(), next_hyps.end());
            size_t real_beam_size = std::min(arr.size(), beam_size);
            std::nth_element(
                    arr.begin(), arr.begin() + real_beam_size, arr.end(),
                    Yao::PrefixScoreCompare);
            arr.resize(real_beam_size);
            std::sort(arr.begin(), arr.end(), PrefixScoreCompare);

            // 4. update new hypotheses
            curr_hypo_.clear();
            hypotheses_.clear();
            for (auto &item: arr) {
                curr_hypo_[item.first] = item.second;
                hypotheses_.emplace_back(item.first);
            }
        }
        batch_hypo.push_back(hypotheses_.front());
        clear();
    }
}

void Yao::CTC_Prefix_BeamSearch::display_hypo() const {
    for (const std::vector<int> & vec : batch_hypo) {
        std::cout << "hypo: ";
        for (auto &id : vec) {
            std::cout << id << " ";
        }
        std::cout << std::endl;
    }
}

void Yao::CTC_Prefix_BeamSearch::clear() {
    hypotheses_.clear();
    curr_hypo_.clear();
    PrefixScore empty_score;
    empty_score.s = 0;
    empty_score.ns = -std::numeric_limits<float>::max();
    curr_hypo_[std::vector<int>()] = empty_score;
}

