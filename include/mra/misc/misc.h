#ifndef MRA_MISC_H
#define MRA_MISC_H

#if !defined(MRA_JIT_COMPILE)
#include <iostream>
#include <utility>
#include <vector>
#include <array>
#endif // !MRA_JIT_COMPILE

#include "mra/misc/platform.h"

namespace mra {

    /// Implements simple Kahan or compensated summation
    template <typename T>
    class KahanAccumulator {
        T sum;
        T c;
    public:
        KahanAccumulator() {} // Must be empty for use in shared memory

        KahanAccumulator(T s) : sum(s), c(0) {}

        KahanAccumulator& operator=(const T s) {
            sum = s;
            c = 0;
            return *this;
        }

        KahanAccumulator& operator+=(const T input) {
            T y = input - c;
            T t = sum + y;
            c = (t - sum) - y;
            sum = t;
            return *this;
        }

        KahanAccumulator& operator+=(const KahanAccumulator& input) {
            (*this) += input.sum;
            (*this) += -input.c;
            return *this;
        }

        operator T() const {
            return sum;
        }
    };

#if !defined(MRA_JIT_COMPILE)
    /// Easy printing of pairs
    template <typename T, typename R>
    std::ostream& operator<<(std::ostream& s, const std::pair<T,R>& a) {
        s << "(" << a.first << "," << a.second << ")";
        return s;
    }

    /// Easy printing of arrays
    template <typename T, size_t N>
    std::ostream& operator<<(std::ostream& s, const std::array<T,N>& a) {
        s << "[";
        for (std::size_t i = 0; i < a.size(); ++i) {
            s << a[i];
            if (i != a.size()-1) s << ", ";
        }
        s << "]";
        return s;
    }

    /// Easy printing of vectors
    template <typename T>
    std::ostream& operator<<(std::ostream& s, const std::vector<T>& a) {
        s << "[";
        for (std::size_t i = 0; i < a.size(); ++i) {
            s << a[i];
            if (i != a.size()-1) s << ", ";
        }
        s << "]";
        return s;
    }
#endif // !MRA_JIT_COMPILE

} // namespace mra

#endif // MRA_MISC_H
