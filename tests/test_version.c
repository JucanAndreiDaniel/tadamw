#include <string.h>

#include "check.h"
#include "taw/taw.h"

int main(void)
{
	const char *v = taw_version();
	CHECK(v != nullptr);
	CHECK(strcmp(v, "0.1.0") == 0);
	return CHECK_DONE();
}
