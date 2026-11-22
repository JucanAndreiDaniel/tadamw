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

/* ---- compression -------------------------------------------------------- */

constexpr int TAW_LEVEL_MIN = 1;
constexpr int TAW_LEVEL_MAX = 9;
constexpr int TAW_LEVEL_DEFAULT = 5;
constexpr uint32_t TAW_BLOCK_DEFAULT = 1u << 20;

typedef struct {
	int level;           /* fitting effort 1..9; 0 = TAW_LEVEL_DEFAULT */
	bool seekable;       /* produce .awseek instead of .aw */
	uint32_t block_size; /* .awseek block size (power of two, 4 KiB..1 GiB);
	                        0 = TAW_BLOCK_DEFAULT */
} taw_params;

/* Compress src[0..n) into a new malloc'd buffer *out of *out_n bytes.
 * params may be nullptr for defaults. */
[[nodiscard]] taw_status taw_compress(const void *src, size_t n,
    const taw_params *params, void **out, size_t *out_n);

/* Decompress a whole .aw or .awseek archive into a new malloc'd buffer. */
[[nodiscard]] taw_status taw_decompress(const void *src, size_t n,
    void **out, size_t *out_n);

/* ---- inspection --------------------------------------------------------- */

typedef enum { TAW_KIND_STREAM = 0, TAW_KIND_SEEKABLE = 1 } taw_kind;

typedef struct {
	taw_kind kind;
	uint8_t arith_profile;
	uint8_t quant;
	uint8_t level;
	uint32_t block_size; /* 0 for streams */
	uint64_t model_size;
} taw_info;

/* Parse and validate the 32-byte header at the start of an archive. */
[[nodiscard]] taw_status taw_read_info(const void *src, size_t n,
    taw_info *info);

#endif
