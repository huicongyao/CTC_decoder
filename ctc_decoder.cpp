//
// Created by yaohc on 9/28/24.
//

#include "ctc_decoder.h"


Yao::CTC_Prefix_BeamSearch::CTC_Prefix_BeamSearch() {
    std::vector<int> empty;
    PrefixScore & empty_score = curr_hypo_[empty] ;
    empty_score.s = 0;
    empty_score.ns = -std::numeric_limits<float>::max();
    empty_score.v_s = 0;
    empty_score.v_ns = 0;
}



// search for a batch of CTC inputs
void Yao::CTC_Prefix_BeamSearch::search(const torch::Tensor &logp, size_t beam_size) {

    if (logp.size(0) == 0) return;
    size_t T = logp.size(0), N = logp.size(1), C = logp.size(2);
    torch::Tensor logp_ = logp.permute({1, 0, 2}).contiguous();
    for (size_t n = 0; n < N; ++n) {
        const float *logp_t = logp_[n][0].data_ptr<float>(); // use raw pointer to accelerate element access speed
        for (size_t t = 0; t < T; ++t) {
            std::unordered_map<std::vector<int>, Yao::PrefixScore, Yao::utils::PrefixHash> next_hyps;
            // 1. Toke passing
            for (size_t id = 0; id < C; id++) {
                auto prob = logp_t[t * C + id];
                for (const auto &it: curr_hypo_) {
                    const std::vector<int> &prefix = it.first;
                    const Yao::PrefixScore &prefix_score = it.second;
                    if (id == 0) {
                        // Case 0: *a + ε => *a
                        PrefixScore &next_score = next_hyps[prefix];
                        next_score.s = Yao::utils::log_add(next_score.s, prefix_score.score() + prob);
                        next_score.v_s = prefix_score.viterbi_score() + prob;
                        next_score.times_s = prefix_score.times();
                    } else if (!prefix.empty() && id == prefix.back()) {
                        // Case 1: *a + a => *a
                        PrefixScore &next_score1 = next_hyps[prefix];
                        next_score1.ns = Yao::utils::log_add(next_score1.ns, prefix_score.ns + prob);
                        if (next_score1.v_ns < prefix_score.v_ns + prob) {
                            next_score1.v_ns = prefix_score.v_ns + prob;
                            if (next_score1.curr_token_prob < prob) {
                                next_score1.curr_token_prob = prob;
                                next_score1.times_ns = prefix_score.times_ns;
                                next_score1.times_ns.back() = t;
                            }
                        }
                        // Case 2: *aε + a => *aa
                        std::vector<int> new_prefix(prefix);
                        new_prefix.emplace_back(id);
                        PrefixScore &next_score2 = next_hyps[new_prefix];
                        next_score2.ns = Yao::utils::log_add(next_score2.ns, prefix_score.s + prob);
                        if (next_score2.v_ns < prefix_score.v_s + prob) {
                            next_score2.v_ns = prefix_score.v_s + prob;
                            next_score2.curr_token_prob = prob;
                            next_score2.times_ns = prefix_score.times_s;
                            next_score2.times_ns.push_back(t);
                        }
                    } else {
                        // Case 3: *a + b => *ab, *aε + b => *ab
                        std::vector<int> new_prefix(prefix);
                        new_prefix.emplace_back(id);
                        PrefixScore &next_score = next_hyps[new_prefix];
                        next_score.ns = Yao::utils::log_add(next_score.ns, prefix_score.score() + prob);
                        if (next_score.v_ns < prefix_score.viterbi_score() + prob) {
                            next_score.v_ns = prefix_score.viterbi_score() + prob;
                            next_score.curr_token_prob = prob;
                            next_score.times_ns = prefix_score.times();
                            next_score.times_ns.push_back(t);
                        }
                    }
                }
            }
            // 2. beam prune, only keep top n best paths
            std::vector<std::pair<std::vector<int>, Yao::PrefixScore>> arr(std::make_move_iterator(next_hyps.begin()), std::make_move_iterator(next_hyps.end()));
//            std::vector<std::pair<std::vector<int>, Yao::PrefixScore>> arr(next_hyps.begin(), next_hyps.end());
            size_t real_beam_size = std::min(arr.size(), beam_size);
            std::nth_element(
                    arr.begin(), arr.begin() + real_beam_size, arr.end(),
                    Yao::PrefixScoreCompare);
            arr.resize(real_beam_size);
            std::sort(arr.begin(), arr.end(), PrefixScoreCompare);

            // 3. update new hypotheses
            curr_hypo_.clear();
            for (auto &item: arr) {
                curr_hypo_[item.first] = item.second;
                if (t == T - 1)
                    hypotheses_.emplace_back(std::move(item));
            }
        }
        batch_hypo.push_back(std::move(hypotheses_.front()));
        clear();
    }
}

void Yao::CTC_Prefix_BeamSearch::display_hypo() const {
    for (const auto & item : batch_hypo) {
        for (auto &id : item.first) {
            std::cout << id << " ";
        }
        std::cout << item.second.score()  << std::endl;
    }
}

void Yao::CTC_Prefix_BeamSearch::clear() {
    hypotheses_.clear();
    curr_hypo_.clear();
    PrefixScore &empty_score = curr_hypo_[std::vector<int>()];
    empty_score.s = 0;
    empty_score.ns = -std::numeric_limits<float>::max();
    empty_score.v_s = 0;
    empty_score.v_ns = 0;
}

std::unordered_map<std::vector<int>, Yao::PrefixScore, Yao::utils::PrefixHash> Yao::get_empty_hypo() {
    std::unordered_map<std::vector<int>, Yao::PrefixScore, Yao::utils::PrefixHash> empty_hypo;
    PrefixScore &empty_score = empty_hypo[std::vector<int>()];
    empty_score.s = 0;
    empty_score.ns = -std::numeric_limits<float>::max();
    empty_score.v_s = 0;
    empty_score.v_ns = 0;
    return empty_hypo;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
Yao::ctc_prefix_beam_Search(const torch::Tensor &logp, size_t beam_size, size_t num_threads) {

    if (logp.size(0) == 0) return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>();

    int64_t T = logp.size(0), N = logp.size(1), C = logp.size(2);
    auto seqs = torch::zeros({N, T}, torch::kUInt8);
    auto moves = torch::zeros({N, T}, torch::kUInt8);
    auto quals = torch::zeros({N, T}, torch::kUInt8);
    auto seqs_ptr = seqs.accessor<uint8_t, 2>();
    auto moves_ptr = moves.accessor<uint8_t, 2>();
    auto quals_ptr = quals.accessor<uint8_t, 2>();


    const torch::Tensor logp_ = logp.permute({1, 0, 2}).contiguous();
    const auto logp_ptr = logp_.accessor<float, 3>();

    std::vector<std::pair<std::vector<int>, PrefixScore>> batch_hypo;
    batch_hypo.reserve(N);
#pragma omp parallel for num_threads(num_threads)
    for (int64_t n = 0; n < N; ++n) {
        auto curr_hyps = Yao::get_empty_hypo();
        std::vector<std::pair<std::vector<int>, PrefixScore>> hypotheses;
        for (int64_t t = 0; t < T; ++t) {
            // 1. Toke passing
            std::unordered_map<std::vector<int>, Yao::PrefixScore, Yao::utils::PrefixHash> next_hyps;
            for (int64_t id = 0; id < C; id++) {
                auto prob = logp_ptr[n][t ][id];
                for (const auto & it : curr_hyps) {
                    const std::vector<int> &prefix = it.first;
                    const Yao::PrefixScore &prefix_score = it.second;
                    if (id == 0) {
                        // Case 0: *a + ε => *a
                        PrefixScore &next_score = next_hyps[prefix];
                        next_score.s = Yao::utils::log_add(next_score.s, prefix_score.score() + prob);
                        next_score.v_s = prefix_score.viterbi_score() + prob;
                        next_score.times_s = prefix_score.times();
                    } else if (!prefix.empty() && id == prefix.back()) {
                        // Case 1: *a + a => *a
                        PrefixScore &next_score1 = next_hyps[prefix];
                        next_score1.ns = Yao::utils::log_add(next_score1.ns, prefix_score.ns + prob);
                        if (next_score1.v_ns < prefix_score.v_ns + prob) {
                            next_score1.v_ns = prefix_score.v_ns + prob;
                            if (next_score1.curr_token_prob < prob) {
                                next_score1.curr_token_prob = prob;
                                next_score1.times_ns = prefix_score.times_ns;
                                next_score1.times_ns.back() = t;
                            }
                        }
                        // Case 2: *aε + a => *aa
                        std::vector<int> new_prefix(prefix);
                        new_prefix.emplace_back(id);
                        PrefixScore &next_score2 = next_hyps[new_prefix];
                        next_score2.ns = Yao::utils::log_add(next_score2.ns, prefix_score.s + prob);
                        if (next_score2.v_ns < prefix_score.v_s + prob) {
                            next_score2.v_ns = prefix_score.v_s + prob;
                            next_score2.curr_token_prob = prob;
                            next_score2.times_ns = prefix_score.times_s;
                            next_score2.times_ns.push_back(t);
                        }
                    } else {
                        // Case 3: *a + b => *ab, *aε + b => *ab
                        std::vector<int> new_prefix(prefix);
                        new_prefix.emplace_back(id);
                        PrefixScore &next_score = next_hyps[new_prefix];
                        next_score.ns = Yao::utils::log_add(next_score.ns, prefix_score.score() + prob);
                        if (next_score.v_ns < prefix_score.viterbi_score() + prob) {
                            next_score.v_ns = prefix_score.viterbi_score() + prob;
                            next_score.curr_token_prob = prob;
                            next_score.times_ns = prefix_score.times();
                            next_score.times_ns.push_back(t);
                        }
                    }
                }
            }
            // 2. beam prune, only keep top n best paths
            std::vector<std::pair<std::vector<int>, Yao::PrefixScore>> arr(std::make_move_iterator(next_hyps.begin()), std::make_move_iterator(next_hyps.end()));
//            std::vector<std::pair<std::vector<int>, Yao::PrefixScore>> arr(next_hyps.begin(), next_hyps.end());
            size_t real_beam_size = std::min(arr.size(), beam_size);
            std::nth_element(
                    arr.begin(), arr.begin() + real_beam_size, arr.end(),
                    Yao::PrefixScoreCompare);
            arr.resize(real_beam_size);
            std::sort(arr.begin(), arr.end(), PrefixScoreCompare);

            // 3. update new hypotheses
            curr_hyps.clear();
            for (auto &item: arr) {
                curr_hyps[item.first] = item.second;
            }
            if (t == T - 1) {
                const auto & prefix = arr[0].first;
                const auto & times = arr[0].second.times();
                assert(times.size() == prefix.size());
                for (size_t i = 0; i < times.size(); i++) {
                    seqs_ptr[n][times[i]] = prefix[i];
                    moves_ptr[n][times[i]] = 1;
                    quals_ptr[n][times[i]] = get_qual(logp_ptr[n][times[i]][prefix[i]]);
                }
            }
        }
    }
    std::cout << static_cast<const void*>(seqs.data_ptr<uint8_t>()) << std::endl;
    return std::make_tuple(std::move(seqs), std::move(moves), std::move(quals));
}
