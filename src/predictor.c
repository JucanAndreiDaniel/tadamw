/* Arithmetic profiles. Only profile 0 (INT_ORDER0) exists so far. */
#include "taw_internal.h"
#include "format.h"

bool taw_pred_supported(uint8_t profile, uint8_t quant)
{
	return profile == TAW_PROFILE_INT_ORDER0 && quant == TAW_QUANT_NONE;
}

taw_status taw_pred_init(taw_pred *pr, uint8_t profile, uint8_t quant,
    const uint8_t *model, size_t model_size)
{
	(void)model;
	if (!taw_pred_supported(profile, quant))
		return TAW_E_UNSUPPORTED;
	if (model_size != 0)
		return TAW_E_FORMAT;
	pr->profile = profile;
	taw_pred_reset(pr);
	return TAW_OK;
}

void taw_pred_reset(taw_pred *pr)
{
	pr->c0 = 1;
	for (int i = 0; i < 256; i++)
		pr->p[i] = 32768;
}

static inline void o0_update(taw_pred *pr, int bit)
{
	uint16_t *p = &pr->p[pr->c0];
	if (bit)
		*p = (uint16_t)(*p + ((65536u - *p) >> 4));
	else
		*p = (uint16_t)(*p - (*p >> 4));
	pr->c0 = 2 * pr->c0 + (uint32_t)bit;
}

void taw_pred_encode(taw_pred *pr, taw_rc_enc *e, const uint8_t *src,
    size_t n)
{
	for (size_t i = 0; i < n; i++) {
		unsigned c = src[i];
		for (int k = 7; k >= 0; k--) {
			int bit = (c >> k) & 1;
			taw_rc_enc_bit(e, bit, pr->p[pr->c0]);
			o0_update(pr, bit);
		}
		pr->c0 = 1;
	}
}

void taw_pred_decode(taw_pred *pr, taw_rc_dec *d, uint8_t *dst, size_t n)
{
	for (size_t i = 0; i < n; i++) {
		for (int k = 0; k < 8; k++)
			o0_update(pr, taw_rc_dec_bit(d, pr->p[pr->c0]));
		dst[i] = (uint8_t)(pr->c0 - 256);
		pr->c0 = 1;
	}
}
