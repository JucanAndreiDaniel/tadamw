#include "taw/taw.h"

#define TAW_STR2(x) #x
#define TAW_STR(x) TAW_STR2(x)

const char *taw_version(void)
{
	return TAW_STR(TAW_VERSION_MAJOR) "." TAW_STR(TAW_VERSION_MINOR) "."
	    TAW_STR(TAW_VERSION_PATCH);
}
