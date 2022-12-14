/*
 *     Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef _MATRIXMUL_H_
#define _MATRIXMUL_H_

// Thread block size
#define BLOCK_SIZE 32

// Basic Matrix dimensions (can be amplified by command line switch)
// (chosen as multiples of the thread block size for simplicity)
#define WA (10  * BLOCK_SIZE) // Matrix A width
#define HA (10 * BLOCK_SIZE) // Matrix A height
#define WB (20  * BLOCK_SIZE) // Matrix B width
#define HB (10 * BLOCK_SIZE)// Matrix B height
#define WC WB  // Matrix C width 
#define HC HA  // Matrix C height

#endif // _MATRIXMUL_H_

