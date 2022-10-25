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

// Based on examples in the stdBLAS proposal, https://wg21.link/p1673, section 17

#include <vector>
#include <utility>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <experimental/linalg>

namespace la = stdex::linalg;

using size_type = int;
using data_type = float;
using vector_type = stdex::mdspan<data_type, stdex::extents<size_type, stdex::dynamic_extent>>;
using matrix_type = stdex::mdspan<data_type, stdex::extents<size_type, stdex::dynamic_extent, stdex::dynamic_extent>, stdex::layout_left>;

data_type random_value() { return static_cast<data_type>( rand() ) / static_cast<data_type>( RAND_MAX ); }

template<class inout_matrix_t,
         class Triangle>
int cholesky_factor(inout_matrix_t A, Triangle t)
{
  using element_type = typename inout_matrix_t::element_type;
  constexpr element_type ZERO {};
  constexpr element_type ONE (1.0);
  const auto n = A.extent(0);

  if (n == 0) {
    return 0;
  }
  else if (n == 1) {
    if (A(0,0) <= ZERO || std::isnan(A(0,0))) {
      return 1;
    }
    A(0,0) = sqrt(A(0,0));
    //A[0,0] = sqrt(A[0,0]);
  }
  else {
    // Partition A into [A11, A12,
    //                   A21, A22],
    // where A21 is the transpose of A12.
    const size_type n1 = n / 2;
    const size_type n2 = n - n1;
    auto A11 = submdspan(A, std::pair{0, n1}, std::pair{0, n1});
    auto A22 = submdspan(A, std::pair{n1, n}, std::pair{n1, n});

    // Factor A11
    const int info1 = cholesky_factor(A11, t);
    if (info1 != 0) {
      return info1;
    }

    if constexpr (std::is_same_v<Triangle, la::upper_triangle_t>) {
      // Update and scale A12
      auto A12 = submdspan(A, std::pair{0, n1}, std::pair{n1, n});
      // Unlike a BLAS/cuBLAS call, here the `triangle` parameter tells the
      // algorithm which part of `transposed(A11)` (rather than A11) to use
      la::triangular_matrix_matrix_left_solve( std::execution::par
                                             , la::transposed(A11)
                                             , la::lower_triangle
                                             , la::explicit_diagonal
                                             , A12
                                             , A12
                                             );
      // A22 = A22 - A12^T * A12
      la::symmetric_matrix_rank_k_update(std::execution::par, -ONE, la::transposed(A12), A22, t);
    }
    else {
      //
      // Compute the Cholesky factorization A = L * L^T
      //
      // Update and scale A21
      auto A21 = submdspan(A, std::pair{n1, n}, std::pair{0, n1});
      la::triangular_matrix_matrix_right_solve( std::execution::par
                                              , la::transposed(A11)
                                              , la::upper_triangle
                                              , la::explicit_diagonal
                                              , A21
                                              , A21
                                              );
      // A22 = A22 - A21 * A21^T
      la::symmetric_matrix_rank_k_update(std::execution::par, -ONE, A21, A22, t);
    }

    // Factor A22
    const int info2 = cholesky_factor(A22, t);
    if (info2 != 0) {
      return info2 + n1;
    }
  }

  return 0;
}

template<class in_matrix_t,
         class Triangle,
         class in_vector_t,
         class out_vector_t>
void cholesky_solve(
  in_matrix_t A,
  Triangle t,
  in_vector_t b,
  out_vector_t x)
{
  if constexpr (std::is_same_v<Triangle, la::upper_triangle_t>) {
    // Solve Ax=b where A = U^T U
    //
    // Solve U^T c = b, using x to store c.
    la::triangular_matrix_vector_solve(std::execution::par, transposed(A), la::lower_triangle, la::explicit_diagonal, b, x);

    // Solve U x = c, overwriting x with result.
    la::triangular_matrix_vector_solve(std::execution::par, A, la::upper_triangle, la::explicit_diagonal, x, x);
  }
  else {
    // Solve Ax=b where A = L L^T
    //
    // Solve L c = b, using x to store c.
    la::triangular_matrix_vector_solve(std::execution::par, A, t, la::explicit_diagonal, b, x);

    // Solve L^T x = c, overwriting x with result.
    la::triangular_matrix_vector_solve(std::execution::par, la::transposed(A), la::upper_triangle, la::explicit_diagonal, x, x);
  }
}

int main(void)
{
  int const N = 4;
  std::vector<data_type> A_vec( N * N ), A_save_vec( N * N ), b_vec( N ), x_vec( N ), b_inverted_vec( N );

  for ( int i = 0; i < N; ++i )
  {
    A_vec[i*N+i] = N;
    for ( int j = i + 1; j < N; ++j )
    {
      A_vec[i*N + j] = random_value();
      A_vec[j*N + i] = A_vec[i*N + j];
    }
  }
  for ( int i = 0; i < N; ++i )
  {
    b_vec[i] = random_value();
  }

  matrix_type A     (      A_vec.data(), N, N );
  matrix_type A_save( A_save_vec.data(), N, N );
  vector_type b         (          b_vec.data(), N );
  vector_type x         (          x_vec.data(), N );
  vector_type b_inverted( b_inverted_vec.data(), N );

  la::copy( std::execution::par, A, A_save );

  auto uplo = la::lower_triangle_t{};

  cholesky_factor( A, uplo );

  cholesky_solve ( A, uplo, b, x );

  la::matrix_vector_product( std::execution::par, A_save, x, b_inverted );

  data_type maxDiff = 0.f;
  for ( int i = 0; i < N; ++i )
  {
    maxDiff = std::max( maxDiff, std::abs( b_vec[i] - b_inverted_vec[i] ) );
  }

  std::cout << "Max Diff = " << maxDiff << std::endl;
}
