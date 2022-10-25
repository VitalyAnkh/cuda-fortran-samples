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

#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../utils/stdblas_utils.h"
#include <experimental/linalg>

using data_type    = double;
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
     *       | 2.0 | 1.0 |
     *
     *   B = | 5.0 | 6.0 |
     *       | 6.0 | 5.0 |
     */

    const std::vector<data_type> A = {1.0, 2.0, 2.0, 1.0};
    const std::vector<data_type> B = {5.0, 6.0, 6.0, 5.0};
    std::vector<data_type> C(m * n);

    printf("A\n");
    print_matrix(m, k, A.data(), lda);
    printf("=====\n");

    printf("B\n");
    print_matrix(k, n, B.data(), ldb);
    printf("=====\n");

    stdex::mdspan<const data_type, extents_type, stdex::layout_left> mds_A( A.data(), m, k );
    stdex::mdspan<const data_type, extents_type, stdex::layout_left> mds_B( B.data(), k, n );
    stdex::mdspan<      data_type, extents_type, stdex::layout_left> mds_C( C.data(), m, n );
    stdex::linalg::lower_triangle_t triang;

    auto mds_trans_B = stdex::linalg::transposed( mds_B );

    try
    {
        stdex::linalg::symmetric_matrix_rank_2k_update( std::execution::par, mds_A, mds_trans_B, mds_C, triang );

        /*
         *   C = | 34.0 |  0.0 |
         *       | 32.0 | 34.0 |
         */

        printf("C\n");
        print_matrix(m, n, C.data(), ldc);
        printf("=====\n");
    }
    catch ( std::system_error const & e )
    {
        printf( "Caught a system error with message '%s'\n", e.code().message().c_str() );
    }

    return EXIT_SUCCESS;
}
