/* Profile 1 mixer for AVX-512F. Built with -mavx512f; only called after a
 * CPUID check. Same per-lane operation order as the scalar reference. */
#include <immintrin.h>

#include "p1.h"
#include "p1_simd_out.h"

/* Layer 1 for 16 hidden units at column j (dot4 over the inputs). */
static inline __m512 layer1_chunk16(const float *w1, const float *x,
    unsigned hpad, unsigned j)
{
	__m512 s[4] = {_mm512_setzero_ps(), _mm512_setzero_ps(),
		_mm512_setzero_ps(), _mm512_setzero_ps()};
	for (unsigned i = 0; i < P1_F; i++)
		s[i & 3] = _mm512_add_ps(s[i & 3], _mm512_mul_ps(
		    _mm512_loadu_ps(w1 + i * hpad + j), _mm512_set1_ps(x[i])));
	__m512 a = _mm512_add_ps(_mm512_add_ps(s[0], s[1]),
	    _mm512_add_ps(s[2], s[3]));
	return _mm512_max_ps(_mm512_min_ps(a, _mm512_set1_ps(8.0f)),
	    _mm512_set1_ps(-8.0f));
}

float p1_mix_avx512(const p1_net *n, const float *x, const float *w1,
    const float *w2)
{
	const unsigned hpad = n->hpad;
	if (!hpad)
		return mix_linear8(x, w2);
	__m128 s = _mm_setzero_ps();
	unsigned j = 0;
	for (; j + 16 <= hpad; j += 16) {
		__m512 h = layer1_chunk16(w1, x, hpad, j);
		s = dot4_chunk(s, _mm256_load_ps(w2 + j),
		    _mm512_castps512_ps256(h));
		s = dot4_chunk(s, _mm256_load_ps(w2 + j + 8),
		    _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(h), 1)));
	}
	if (j < hpad) /* hpad is a multiple of 8 */
		s = dot4_chunk(s, _mm256_load_ps(w2 + j),
		    layer1_chunk8(w1, x, hpad, j));
	return dot4_finish(s);
}
