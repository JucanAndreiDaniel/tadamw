/* tar_aw public API. */
#ifndef TAW_TAW_H
#define TAW_TAW_H

#include <stddef.h>
#include <stdint.h>

#define TAW_VERSION_MAJOR 0
#define TAW_VERSION_MINOR 1
#define TAW_VERSION_PATCH 0

typedef enum {
	TAW_OK = 0,
	TAW_E_ARG,         /* invalid argument */
	TAW_E_NOMEM,       /* allocation failed */
	TAW_E_FORMAT,      /* not a tar_aw archive, or structurally corrupt */
	TAW_E_UNSUPPORTED, /* valid but uses a version/profile/quant we refuse */
	TAW_E_CHECKSUM,    /* CRC mismatch */
	TAW_E_RANGE,       /* requested range outside the archive's data */
	TAW_E_IO,          /* read/write callback failed */
} taw_status;

/* Library version string, e.g. "0.1.0". */
const char *taw_version(void);

/* Human-readable text for a status code. */
const char *taw_strerror(taw_status s);

#endif
