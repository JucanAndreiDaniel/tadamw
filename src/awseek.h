/* .awseek seekable variant. */
#ifndef TAW_AWSEEK_H
#define TAW_AWSEEK_H

#include "p1.h"
#include "taw_internal.h"

/* p must already be resolved (taw_params_resolve). */
taw_status taw_awseek_encode(const uint8_t *src, size_t n,
    const taw_params *p, const taw_model_choice *m, taw_buf *out);

#endif
