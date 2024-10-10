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

namespace Yao {
    struct PrefixScore {
        float s = -std::numeric_limits<float>::max();       // blank ending score
        float ns = -std::numeric_limits<float>::max();      // none blank ending score
        float time_step = 0;
        float score() const {
            return Yao::utils::log_add(s, ns);
        }
    };

    static bool PrefixScoreCompare(
            const std::pair<std::vector<int>, PrefixScore> & a,
            const std::pair<std::vector<int>, PrefixScore> & b) {
        return a.second.score() > b.second.score();
    }

    class CTC_Prefix_BeamSearch {
    public:
        CTC_Prefix_BeamSearch() {
            std::vector<int> empty;
            PrefixScore empty_score;
            empty_score.s = 0;
            empty_score.ns = -std::numeric_limits<float>::max();
            curr_hypo_[empty] = empty_score;
        }
        void search(const std::vector<std::vector<float>> & logp, size_t beam_size = 32) ;
        void display_hypo() const;
        void clear();
    private:
        std::vector<std::vector<int>> hypotheses_;
        std::unordered_map<std::vector<int>, Yao::PrefixScore, Yao::utils::PrefixHash> curr_hypo_;
        std::vector<std::vector<int>> times_;
    };

}

#endif //CTC_DECODER_CTC_DECODER_H
