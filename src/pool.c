/* SPDX-License-Identifier: GPL-3.0-or-later */
/* taw_parallel: a work queue over n jobs (taw_internal.h). */
#include <pthread.h>
#include <stdlib.h>

#include "taw_internal.h"

typedef struct {
	pthread_mutex_t mu;
	pthread_cond_t cv;
	size_t n, next, window_end;
	bool *done;
	bool stop;
	taw_work_fn work;
	void *ctx;
} pool;

typedef struct {
	pool *p;
	unsigned worker;
} worker_arg;

static void *pool_main(void *arg)
{
	worker_arg *w = arg;
	pool *p = w->p;
	for (;;) {
		pthread_mutex_lock(&p->mu);
		while (!p->stop && p->next < p->n && p->next >= p->window_end)
			pthread_cond_wait(&p->cv, &p->mu);
		if (p->stop || p->next >= p->n) {
			pthread_mutex_unlock(&p->mu);
			return nullptr;
		}
		size_t i = p->next++;
		pthread_mutex_unlock(&p->mu);
		p->work(p->ctx, i, w->worker);
		pthread_mutex_lock(&p->mu);
		p->done[i] = true;
		pthread_cond_broadcast(&p->cv);
		pthread_mutex_unlock(&p->mu);
	}
}

taw_status taw_parallel(unsigned threads, size_t n, size_t window,
    taw_work_fn work, taw_done_fn done, void *ctx)
{
	if (n == 0)
		return TAW_OK;
	if (threads == 0)
		threads = 1;
	if ((size_t)threads > n)
		threads = (unsigned)n;
	if (window == 0 || window > n)
		window = n;
	taw_status st = TAW_OK;
	if (threads == 1) { /* inline, in order */
		for (size_t i = 0; i < n && st == TAW_OK; i++) {
			work(ctx, i, 0);
			if (done)
				st = done(ctx, i);
		}
		return st;
	}
	pool p = {.n = n, .window_end = window, .work = work, .ctx = ctx};
	p.done = calloc(n, sizeof *p.done);
	pthread_t *th = calloc(threads, sizeof *th);
	worker_arg *args = calloc(threads, sizeof *args);
	if (!p.done || !th || !args) {
		free(p.done);
		free(th);
		free(args);
		return TAW_E_NOMEM;
	}
	pthread_mutex_init(&p.mu, nullptr);
	pthread_cond_init(&p.cv, nullptr);
	unsigned started = 0;
	for (; started < threads; started++) {
		args[started] = (worker_arg){&p, started};
		if (pthread_create(&th[started], nullptr, pool_main, &args[started]))
			break;
	}
	if (started == 0) {
		st = TAW_E_NOMEM;
		goto out;
	}
	for (size_t i = 0; i < n; i++) {
		pthread_mutex_lock(&p.mu);
		while (!p.done[i])
			pthread_cond_wait(&p.cv, &p.mu);
		pthread_mutex_unlock(&p.mu);
		if (done && (st = done(ctx, i)) != TAW_OK)
			break;
		pthread_mutex_lock(&p.mu);
		p.window_end = i + 1 + window < n ? i + 1 + window : n;
		pthread_cond_broadcast(&p.cv);
		pthread_mutex_unlock(&p.mu);
	}
out:
	pthread_mutex_lock(&p.mu);
	p.stop = true;
	pthread_cond_broadcast(&p.cv);
	pthread_mutex_unlock(&p.mu);
	for (unsigned k = 0; k < started; k++)
		pthread_join(th[k], nullptr);
	pthread_mutex_destroy(&p.mu);
	pthread_cond_destroy(&p.cv);
	free(p.done);
	free(th);
	free(args);
	return st;
}
