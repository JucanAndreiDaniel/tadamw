/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Minimal test assertions. */
#ifndef TAW_TESTS_CHECK_H
#define TAW_TESTS_CHECK_H

#include <stdio.h>
#include <stdlib.h>

static int check_failures;

#define CHECK(cond)                                                          \
	do {                                                                 \
		if (!(cond)) {                                               \
			fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, \
			    __LINE__, #cond);                                \
			check_failures++;                                    \
		}                                                            \
	} while (0)

#define REQUIRE(cond)                                                        \
	do {                                                                 \
		if (!(cond)) {                                               \
			fprintf(stderr, "%s:%d: REQUIRE failed: %s\n",       \
			    __FILE__, __LINE__, #cond);                      \
			exit(1);                                             \
		}                                                            \
	} while (0)

#define CHECK_DONE() (check_failures ? (fprintf(stderr, "%d failure(s)\n", \
				check_failures), 1) : 0)

#endif
