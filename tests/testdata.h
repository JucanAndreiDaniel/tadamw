/* Deterministic test inputs. */
#ifndef TAW_TESTS_TESTDATA_H
#define TAW_TESTS_TESTDATA_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

static inline uint64_t td_next(uint64_t *s)
{
	/* splitmix64 */
	uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
	return z ^ (z >> 31);
}

enum { TD_ZERO, TD_RANDOM, TD_TEXT, TD_SKEWED, TD_KINDS };

static inline void td_fill(uint8_t *p, size_t n, int kind, uint64_t seed)
{
	static const char words[] =
	    "the quick brown fox jumps over a lazy dog while archives "
	    "predict bits with fitted models and seek tables ";
	uint64_t s = seed;
	for (size_t i = 0; i < n; i++) {
		switch (kind) {
		case TD_ZERO: p[i] = 0; break;
		case TD_RANDOM: p[i] = (uint8_t)td_next(&s); break;
		case TD_TEXT:
			p[i] = (uint8_t)words[(i * 7 + (td_next(&s) & 3)) %
			    (sizeof words - 1)];
			break;
		default: /* geometric-ish byte distribution */
			p[i] = (uint8_t)__builtin_ctzll(td_next(&s) | 1u << 20);
			break;
		}
	}
}

#endif
