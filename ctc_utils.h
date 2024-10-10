//
// Created by yaohc on 2024/9/27.
//

#ifndef CTC_DECODER_CTC_UTILS_H
#define CTC_DECODER_CTC_UTILS_H
#include <cmath>
#include <limits>
#include <vector>
#include <string>
#include <type_traits>

namespace Yao {
namespace utils {
// Template function to calculate safe log-sum-exp for any floating point type
    template<typename T>
    T log_add(T log_a, T log_b) {
        // Ensure that log_a and log_b are finite
        static_assert(std::is_floating_point<T>::value, "Template argument must be a floating-point type");

        // Handle special cases like negative infinity (log(0))
        if (log_a == -std::numeric_limits<T>::infinity()) return log_b;
        if (log_b == -std::numeric_limits<T>::infinity()) return log_a;

        // Find the maximum value between log_a and log_b to avoid overflow
        T max_log = std::max(log_a, log_b);

        // Return max_log + log(1 + exp(-|log_a - log_b|))
        return max_log + std::log1p(std::exp(-std::fabs(log_a - log_b)));
    }

    struct PrefixHash {
        /*
         * implement a hash for std::vector<int>
         * */
        size_t operator() (const std::vector<int>& prefix) const {
            size_t hash_code = 0;
            // here we use KB&DR hash code
            for (int id : prefix) {
                hash_code = id + 31 * hash_code;
            }
            return hash_code;
        }
    };
}
}

#endif //CTC_DECODER_CTC_UTILS_H
