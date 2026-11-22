/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Shared AVX2 pieces of the profile 1 mixer, included by p1_avx2.c and
 * p1_avx512.c (each compiled with its own -m flags). */
#ifndef TAW_P1_SIMD_OUT_H
#define TAW_P1_SIMD_OUT_H

#include <immintrin.h>

#include "p1.h"

/* One 8-wide chunk of a dot4: products w * v, lanes 0..3 then 4..7 added
 * into the four partial sums. Chunks must come in increasing index order;
 * that is exactly dot4's order (lane r of every half belongs to s_r). */
static inline __m128 dot4_chunk(__m128 s, __m256 w, __m256 v)
{
	__m256 q = _mm256_mul_ps(w, v);
	s = _mm_add_ps(s, _mm256_castps256_ps128(q));
	return _mm_add_ps(s, _mm256_extractf128_ps(q, 1));
}

/* (s0 + s1) + (s2 + s3) */
static inline float dot4_finish(__m128 s)
{
	__m128 t = _mm_add_ps(s, _mm_shuffle_ps(s, s, _MM_SHUFFLE(2, 3, 0, 1)));
	return _mm_cvtss_f32(_mm_add_ss(t, _mm_movehl_ps(t, t)));
}

/* Layer 1 for 8 hidden units starting at column j: four partial sums over
 * the inputs (i mod 4), combined as in dot4, then clamped to [-8, 8]. */
static inline __m256 layer1_chunk8(const float *w1, const float *x,
    unsigned hpad, unsigned j)
{
	__m256 s[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(),
		_mm256_setzero_ps(), _mm256_setzero_ps()};
	for (unsigned i = 0; i < P1_F; i++)
		s[i & 3] = _mm256_add_ps(s[i & 3], _mm256_mul_ps(
		    _mm256_loadu_ps(w1 + i * hpad + j), _mm256_set1_ps(x[i])));
	__m256 a = _mm256_add_ps(_mm256_add_ps(s[0], s[1]),
	    _mm256_add_ps(s[2], s[3]));
	return _mm256_max_ps(_mm256_min_ps(a, _mm256_set1_ps(8.0f)),
	    _mm256_set1_ps(-8.0f));
}

/* Single-layer case: dot4 over x[0..P1_XPAD) (padding weights are 0). */
static inline float mix_linear8(const float *x, const float *w2)
{
	__m128 s = _mm_setzero_ps();
	for (unsigned i = 0; i < P1_XPAD; i += 8)
		s = dot4_chunk(s, _mm256_load_ps(w2 + i), _mm256_loadu_ps(x + i));
	return dot4_finish(s);
}

#endif
