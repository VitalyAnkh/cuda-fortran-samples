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

// This example demonstrates performance differences between the synchronous and no_sync execution
// policies with the cuBLAS backend.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../utils/stdblas_utils.h"
#include <experimental/linalg>

using data_type    = double;
using extents_type = stdex::extents<std::int64_t, stdex::dynamic_extent, stdex::dynamic_extent>;

int main(int argc, char *argv[]) {

    const std::int64_t m = 128;
    const std::int64_t n = m;
    const std::int64_t k = m;

    std::vector<data_type> A(m * n);
    std::vector<data_type> B(m * n);
    std::vector<data_type> C(m * n);

    random_vector( m*n, A.data() );
    random_vector( m*n, B.data() );

    stdex::mdspan< data_type, extents_type, stdex::layout_left > mds_A( A.data(), m, k );
    stdex::mdspan< data_type, extents_type, stdex::layout_left > mds_B( B.data(), k, n );
    stdex::mdspan< data_type, extents_type, stdex::layout_left > mds_C( C.data(), m, n );

    stdex::linalg::matrix_product( std::execution::par, mds_A, mds_B, mds_C );

    Timer timer;
    timer.start();

#ifdef NO_SYNC
    for ( int i = 0; i < 50; ++i )
    {
        stdex::linalg::matrix_product( no_sync( std::execution::par ), mds_A, mds_B, mds_C );
    }

    cudaDeviceSynchronize();
#else
    for ( int i = 0; i < 50; ++i )
    {
        stdex::linalg::matrix_product(          std::execution::par  , mds_A, mds_B, mds_C );
    }
#endif

    double t = timer.end();
    std::cout << "timer = " << t << std::endl;

    print_matrix( 2, 2, C.data(), m );

    return EXIT_SUCCESS;
}
