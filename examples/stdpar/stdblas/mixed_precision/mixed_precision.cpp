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

// Example of mixed-precision matrix_product

#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../utils/stdblas_utils.h"
#include <experimental/linalg>

using input_type   = std::int8_t;
using output_type  = float;
using extents_type = stdex::extents<int, stdex::dynamic_extent, stdex::dynamic_extent>;

int main(int argc, char *argv[]) {

    const int m = 2;
    const int n = 2;
    const int k = 2;
    const int lda = 2;
    const int ldb = 2;
    const int ldc = 2;
    /*
     *   A = | 1.0 | 2.0 |
     *       | 3.0 | 4.0 |
     *
     *   B = | 5.0 | 6.0 |
     *       | 7.0 | 8.0 |
     */

    const std::vector<input_type> A = {1, 2, 3, 4};
    const std::vector<input_type> B = {5, 6, 7, 8};
    std::vector<output_type> C(m * n);

    printf("A\n");
    print_matrix(m, k, A.data(), lda);
    printf("=====\n");

    printf("B\n");
    print_matrix(k, n, B.data(), ldb);
    printf("=====\n");

    stdex::mdspan<const  input_type, extents_type, stdex::layout_left > mds_A( A.data(), m, k );
    stdex::mdspan<const  input_type, extents_type, stdex::layout_left > mds_B( B.data(), k, n );
    stdex::mdspan<      output_type, extents_type, stdex::layout_left > mds_C( C.data(), m, n );

    stdex::linalg::matrix_product( std::execution::par, mds_A, mds_B, mds_C );

    /*
     *   C = | 23.0 | 31.0 |
     *       | 34.0 | 46.0 |
     */

    printf("C\n");
    print_matrix(m, n, C.data(), ldc);
    printf("=====\n");

    return EXIT_SUCCESS;
}
