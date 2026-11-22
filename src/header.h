/* File header (docs/format.md §1). */
#ifndef TAW_HEADER_H
#define TAW_HEADER_H

#include "format.h"
#include "taw_internal.h"

typedef struct {
	uint8_t kind, profile, quant, level;
	uint32_t block_size;
	uint64_t model_size;
	uint32_t model_crc;
} taw_header;

void taw_header_write(taw_buf *b, const taw_header *h);

/* Parse and fully validate a header, including that this build supports
 * its (profile, quant) pair. Needs n >= TAW_HEADER_SIZE. */
taw_status taw_header_parse(const uint8_t *p, size_t n, taw_header *h);

/* Validate public params and fill in defaults. */
taw_status taw_params_resolve(const taw_params *in, taw_params *out);

/* Shared front matter for both variants: parse the header and check the model
 * section against [0, n). On success *model points at the model bytes. */
taw_status taw_open_common(const uint8_t *src, size_t n, uint8_t want_kind,
    taw_header *h, const uint8_t **model);

#endif
