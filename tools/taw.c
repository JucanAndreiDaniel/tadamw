/* taw: command-line front end for the tar_aw archive format. */
#include <stdio.h>
#include <string.h>

#include "taw/taw.h"

static void usage(FILE *f)
{
	fprintf(f, "usage: taw --version\n");
}

int main(int argc, char **argv)
{
	if (argc == 2 && strcmp(argv[1], "--version") == 0) {
		printf("taw %s\n", taw_version());
		return 0;
	}
	usage(stderr);
	return 2;
}
