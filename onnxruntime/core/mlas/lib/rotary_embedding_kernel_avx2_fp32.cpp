/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    rotary_embedding_kernel_neon_fp16.cpp

Abstract:

    This module implements the fp16 rotary embedding kernels for ARM NEON.

--*/

#include <cassert>

#include "fp16_common.h"
#include "rotary_embedding.h"
#include "rotary_embedding_kernel_avx2.h"

namespace rope_avx2 {

namespace {

typedef __m256 float32x8_t;

template <bool interleaved>
void
RopeKernel_Avx2_Impl(
    const float* input,
    const float* sin,
    const float* cos,
    size_t dim,
    float* output
);

template <>
void
RopeKernel_Avx2_Impl<false>(
    const float* input,
    const float* sin,
    const float* cos,
    size_t dim,
    float* output
) {
    const size_t half_dim = dim >> 1;
    size_t i = 0, j = half_dim;
    for (; i + 7 < half_dim; i += 8, j += 8) {
        float32x8_t real = _mm256_loadu_ps(input + i); // __m256 _mm256_loadu_ps
        float32x8_t imag = _mm256_loadu_ps(input + j);
        float32x8_t sin_val = _mm256_loadu_ps(sin + i);
        float32x8_t cos_val = _mm256_loadu_ps(cos + i);
        float32x8_t real_out = _mm256_fmsub_ps(real, cos_val, _mm256_mul_ps(imag, sin_val)); //__m256 _mm256_mul_ps , __m256 _mm256_fmsub_ps
        float32x8_t imag_out = _mm256_fmadd_ps(real, sin_val, _mm256_mul_ps(imag, cos_val));
        _mm256_store_ps(output + i, real_out); // _mm256_store_ps
        _mm256_store_ps(output + j, imag_out);
    }
    if (half_dim - i != 0) {
        size_t rem = half_dim - i;
        static const int32_t mask_buffer[16] = {-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0};
        const __m256i mask = _mm256_loadu_si256((const __m256i*)(mask_buffer + 8 - rem));
        float32x8_t real = _mm256_maskload_ps(input, mask); //__m256 _mm256_maskload_ps
        float32x8_t imag = _mm256_maskload_ps(input + j, mask);
        float32x8_t sin_val = _mm256_maskload_ps(sin + i, mask);
        float32x8_t cos_val = _mm256_maskload_ps(cos + i, mask);
        float32x8_t real_out = _mm256_fmsub_ps(real, cos_val, _mm256_mul_ps(imag, sin_val)); //__m256 _mm256_mul_ps , __m256 _mm256_fmsub_ps
        float32x8_t imag_out = _mm256_fmadd_ps(real, sin_val, _mm256_mul_ps(imag, cos_val));
        _mm256_maskstore_ps(output + i, mask, real_out); //_mm256_store_ps
        _mm256_maskstore_ps(output + j, mask, imag_out);
    }
}

template <>
void
RopeKernel_Avx2_Impl<true>(
    const float* input,
    const float* sin,
    const float* cos,
    size_t dim,
    float* output
) {
    size_t i = 0;
    for (; i + 15 < dim; i += 16) {
        float32x8_t x0 = _mm256_loadu_ps(input + i);
        float32x8_t x1 = _mm256_loadu_ps(input + i + 8);
        const int real_mask[8] = {0, 2, 4, 6, 8, 10, 12, 14};
        const int imag_mask[8] = {1, 3, 5, 7, 9, 11, 13, 15};
        __m256i real_mask_vec = _mm256_loadu_si256((__m256i*)real_mask);
        __m256i imag_mask_vec = _mm256_loadu_si256((__m256i*)imag_mask);
        float32x8_t real = _mm256_permutex2var_ps(x0, real_mask_vec, x1);
        float32x8_t imag = _mm256_permutex2var_ps(x0, imag_mask_vec, x1);
        float32x8_t sin_val = _mm256_loadu_ps(sin + i);
        float32x8_t cos_val = _mm256_loadu_ps(cos + i);
        float32x8_t real_out = _mm256_fmsub_ps(real, cos_val, _mm256_mul_ps(imag, sin_val));
        float32x8_t imag_out = _mm256_fmadd_ps(real, sin_val, _mm256_mul_ps(imag, cos_val));
        float32x8_t y0 = _mm256_permutex2var_ps(real_out, _mm256_set_epi32(11, 3, 10, 2, 9, 1, 8, 0), imag_out);
        float32x8_t y1 = _mm256_permutex2var_ps(real_out, _mm256_set_epi32(15, 7, 14, 6, 13, 5, 12, 4), imag_out);
        _mm256_maskstore_ps(output + i, mask0, y0);
        _mm256_maskstore_ps(output + i + 8, mask1, y1);
    }
    if (dim - i != 0) {
        size_t rem = dim - i;
        static const int32_t mask_buffer[16] = {-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0};
        const __m256i mask0 = _mm256_loadu_si256((const __m256i*)(mask_buffer + 8 - (rem>8?8:rem)));
        const __m256i mask1 = _mm256_loadu_si256((const __m256i*)(mask_buffer + 8 - (rem>8?(rem-8):0)));
        float32x8_t x0 = _mm256_maskload_ps(input + i, mask0);
        float32x8_t x1 = _mm256_maskload_ps(input + i + 8, mask1);
        __m256i real_mask_vec = _mm256_set_epi32(14, 12, 10, 8, 6, 4, 2, 0);  //_mm256_loadu_si256((__m256i*)real_mask);
        __m256i imag_mask_vec = _mm256_set_epi32(15, 13, 11, 9, 7, 5, 3, 1);  //_mm256_loadu_si256((__m256i*)imag_mask);
        float32x8_t real = _mm256_permutex2var_ps(x0, real_mask_vec, x1);
        float32x8_t imag = _mm256_permutex2var_ps(x0, imag_mask_vec, x1);
        float32x8_t sin_val = _mm256_loadu_ps(sin + i);
        float32x8_t cos_val = _mm256_loadu_ps(cos + i);
        float32x8_t real_out = _mm256_fmsub_ps(real, cos_val, _mm256_mul_ps(imag, sin_val));
        float32x8_t imag_out = _mm256_fmadd_ps(real, sin_val, _mm256_mul_ps(imag, cos_val));
        float32x8_t y0 = _mm256_permutex2var_ps(real_out, _mm256_set_epi32(11, 3, 10, 2, 9, 1, 8, 0), imag_out);
        float32x8_t y1 = _mm256_permutex2var_ps(real_out, _mm256_set_epi32(15, 7, 14, 6, 13, 5, 12, 4), imag_out);
        _mm256_maskstore_ps(output + i, mask0, y0);
        _mm256_maskstore_ps(output + i + 8, mask1, y1);
    }
}

}  // namespace

void
RopeKernel_Avx2(
    const float* input,
    const float* sin,
    const float* cos,
    size_t dim,
    bool interleaved,
    float* output
) {
    // real part and imaginary part must be paired
    assert(dim % 2 == 0);
    const auto* input_impl = reinterpret_cast<const float*>(input);
    const auto* sin_impl = reinterpret_cast<const float*>(sin);
    const auto* cos_impl = reinterpret_cast<const float*>(cos);
    auto* output_impl = reinterpret_cast<float*>(output);

    if (interleaved) {
        RopeKernel_Avx2_Impl<true>(input_impl, sin_impl, cos_impl, dim, output_impl);
    } else {
        RopeKernel_Avx2_Impl<false>(input_impl, sin_impl, cos_impl, dim, output_impl);
    }
}

}  // namespace rope_neon
