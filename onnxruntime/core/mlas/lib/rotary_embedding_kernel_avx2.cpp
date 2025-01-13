/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    rotary_embedding_kernel_neon.cpp

Abstract:

    This module implements the rotary embedding kernels for ARM NEON.

--*/

#include "rotary_embedding.h"
#include "rotary_embedding_kernel_avx2.h"

//
// Kernel dispatch structure definition.
//
const MLAS_ROPE_DISPATCH MlasRopeDispatchAvx2 = []() {
    MLAS_ROPE_DISPATCH d;

#if defined(MLAS_TARGET_AMD64)
        d.SRope = rope_avx2::RopeKernel_Avx2;
#endif
    return d;
}();
