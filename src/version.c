/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "tadamw/tadamw.h"

#define TAW_STR2(x) #x
#define TAW_STR(x) TAW_STR2(x)

const char *taw_version(void)
{
	return TAW_STR(TAW_VERSION_MAJOR) "." TAW_STR(TAW_VERSION_MINOR) "."
	    TAW_STR(TAW_VERSION_PATCH);
}

const char *taw_strerror(taw_status s)
{
	switch (s) {
	case TAW_OK: return "success";
	case TAW_E_ARG: return "invalid argument";
	case TAW_E_NOMEM: return "out of memory";
	case TAW_E_FORMAT: return "not a tar_aw archive or corrupt";
	case TAW_E_UNSUPPORTED: return "unsupported archive feature";
	case TAW_E_CHECKSUM: return "checksum mismatch";
	case TAW_E_RANGE: return "range out of bounds";
	case TAW_E_IO: return "I/O error";
	}
	return "unknown error";
}
