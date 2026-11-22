/* SPDX-License-Identifier: GPL-3.0-or-later */
/* IEEE binary16 <-> binary32 without libm. */
#ifndef TAW_HALF_H
#define TAW_HALF_H

#include <stdint.h>
#include <string.h>

static inline float taw_bits_f32(uint32_t u)
{
	float f;
	memcpy(&f, &u, sizeof f);
	return f;
}

static inline uint32_t taw_f32_bits(float f)
{
	uint32_t u;
	memcpy(&u, &f, sizeof u);
	return u;
}

/* Exact; matches F16C vcvtph2ps for every input. */
static inline float taw_h2f(uint16_t h)
{
	uint32_t s = (uint32_t)(h & 0x8000u) << 16;
	uint32_t e = (h >> 10) & 31u, m = h & 0x3FFu, u;
	if (e == 0) {
		if (m == 0) {
			u = s;
		} else {
			e = 113;
			while (!(m & 0x400u)) {
				m <<= 1;
				e--;
			}
			u = s | e << 23 | (m & 0x3FFu) << 13;
		}
	} else if (e == 31) {
		u = s | 0x7F800000u | m << 13;
	} else {
		u = s | (e + 112) << 23 | m << 13;
	}
	return taw_bits_f32(u);
}

/* Round to nearest even; finite inputs above the half range become inf.
 * NaN is not expected (encoder only). */
static inline uint16_t taw_f2h(float f)
{
	uint32_t u = taw_f32_bits(f);
	uint16_t s = (uint16_t)((u >> 16) & 0x8000u);
	u &= 0x7FFFFFFFu;
	if (u >= 0x47800000u) /* >= 65536: inf */
		return s | 0x7C00u;
	if (u < 0x38800000u) { /* below 2^-14: subnormal or zero */
		float y = taw_bits_f32(u) + 0.5f;
		return s | (uint16_t)(taw_f32_bits(y) - 0x3F000000u);
	}
	uint32_t odd = (u >> 13) & 1u;
	u += 0xC8000FFFu + odd;
	return s | (uint16_t)(u >> 13);
}

#endif
