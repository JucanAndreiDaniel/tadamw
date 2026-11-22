/* .aw stream variant. */
#ifndef TAW_AW_H
#define TAW_AW_H

#include "p1.h"
#include "taw_internal.h"

/* p must already be resolved (taw_params_resolve). */
taw_status taw_aw_encode(const uint8_t *src, size_t n, const taw_params *p,
    const taw_model_choice *m, taw_buf *out);

/* out must be empty on entry. */
taw_status taw_aw_decode(const uint8_t *src, size_t n, taw_buf *out);

#endif
