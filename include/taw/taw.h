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

/* Predictor choice. AUTO fits the mixer and keeps it only if the archive
 * comes out smaller than with the order-0 placeholder. */
typedef enum {
	TAW_PREDICTOR_AUTO = 0,
	TAW_PREDICTOR_ORDER0, /* arith profile 0 */
	TAW_PREDICTOR_MIX,    /* arith profile 1; TAW_E_ARG if the input is
	                         too small for any model */
} taw_predictor;

/* Weight format for the mixer. AUTO lets the encoder choose. */
typedef enum {
	TAW_WEIGHTS_AUTO = 0,
	TAW_WEIGHTS_F16,
	TAW_WEIGHTS_Q8,
} taw_weights;

typedef struct {
	int level;           /* fitting effort 1..9; 0 = TAW_LEVEL_DEFAULT */
	bool seekable;       /* produce .awseek instead of .aw */
	uint32_t block_size; /* .awseek block size (power of two, 4 KiB..1 GiB);
	                        0 = TAW_BLOCK_DEFAULT */
	taw_predictor predictor;
	taw_weights weights;
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

/* ---- random access (.awseek) -------------------------------------------- */

/* A random-access byte source. read_at must fill exactly n bytes at off;
 * the library never asks for bytes past size. */
typedef struct {
	taw_status (*read_at)(void *ctx, uint64_t off, void *dst, size_t n);
	void *ctx;
	uint64_t size;
} taw_source;

/* Source over a memory buffer (which must outlive the source). */
void taw_source_memory(taw_source *s, const void *p, size_t n);

typedef struct taw_seek taw_seek;

/* Open a .awseek archive: reads and validates header, model and index. The
 * source is copied; its ctx must stay valid until taw_seek_close. */
[[nodiscard]] taw_status taw_seek_open(const taw_source *src, taw_seek **out);

/* Uncompressed size of the archive's data. */
uint64_t taw_seek_size(const taw_seek *s);

/* Read uncompressed bytes [off, off+n), decoding only overlapping blocks. */
[[nodiscard]] taw_status taw_seek_read(taw_seek *s, uint64_t off, void *dst,
    size_t n);

void taw_seek_close(taw_seek *s);

#endif
