/* SPDX-License-Identifier: GPL-3.0-or-later */
#include <string.h>

#include "check.h"
#include "tadamw/tadamw.h"

int main(void)
{
	const char *v = taw_version();
	CHECK(v != nullptr);
	CHECK(strcmp(v, "0.1.0") == 0);
	return CHECK_DONE();
}
