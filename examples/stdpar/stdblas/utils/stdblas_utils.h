/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <iostream>
#include <chrono>

template <typename data_type>
data_type random_value() { return static_cast<data_type>( rand() ) / static_cast<data_type>( RAND_MAX ); }

template <typename T> void random_vector( int m, T * data ) {
    for ( int i = 0; i < m; ++i ) {
        data[i] = random_value<T>();
    }
}

template <typename T> void print_matrix(const int &m, const int &n, const T *A, const int &lda);

template <> void print_matrix(const int &m, const int &n, const std::int8_t *A, const int &lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%d ", A[j * lda + i]);
        }
        std::printf("\n");
    }
}

template <> void print_matrix(const int &m, const int &n, const int *A, const int &lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%d ", A[j * lda + i]);
        }
        std::printf("\n");
    }
}

template <> void print_matrix(const int &m, const int &n, const float *A, const int &lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%0.2f ", A[j * lda + i]);
        }
        std::printf("\n");
    }
}

template <> void print_matrix(const int &m, const int &n, const double *A, const int &lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%0.2f ", A[j * lda + i]);
        }
        std::printf("\n");
    }
}

class Timer
{
  private:

    std::chrono::time_point<std::chrono::steady_clock> mStart, mEnd;

  public:

    void start()
    {
        mStart = std::chrono::steady_clock::now();
    }

    double end()
    {
        mEnd = std::chrono::steady_clock::now();
        std::chrono::duration< double > diff = mEnd - mStart;
        return diff.count();
    }
};

