#include <igraph.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*
 * Benchmark driver for igraph's Leiden implementation.
 *
 * Input protocol (whitespace-separated tokens):
 *   n m directed weighted
 *   from to weight    (m lines; weight ignored when weighted = 0)
 *   resolution beta objective n_iterations seed
 *   K                 (number of runs)
 *
 * objective: 1 = MODULARITY, 2 = CPM, 3 = ER
 *
 * Output:
 *   K run times in microseconds, space separated, on one line
 *   "nb_clusters quality" of the last run
 */

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

int main(void) {
    igraph_int_t n, m;
    int directed_i, weighted_i;
    if (scanf("%" IGRAPH_PRId " %" IGRAPH_PRId " %d %d", &n, &m, &directed_i, &weighted_i) != 4) {
        return 2;
    }
    igraph_bool_t directed = directed_i != 0;
    igraph_bool_t weighted = weighted_i != 0;

    igraph_vector_int_t edgelist;
    igraph_vector_t weights;
    igraph_vector_int_init(&edgelist, 0);
    igraph_vector_init(&weights, 0);
    for (igraph_int_t i = 0; i < m; i++) {
        igraph_int_t a, b;
        double w;
        if (scanf("%" IGRAPH_PRId " %" IGRAPH_PRId " %lf", &a, &b, &w) != 3) {
            return 2;
        }
        igraph_vector_int_push_back(&edgelist, a);
        igraph_vector_int_push_back(&edgelist, b);
        igraph_vector_push_back(&weights, w);
    }

    double resolution, beta;
    int objective;
    igraph_int_t n_iterations;
    unsigned long long seed_ull;
    int K;
    if (scanf("%lf %lf %d %" IGRAPH_PRId " %llu %d",
              &resolution, &beta, &objective, &n_iterations, &seed_ull, &K) != 6) {
        return 2;
    }
    igraph_uint_t seed = (igraph_uint_t)seed_ull;

    igraph_t g;
    igraph_create(&g, &edgelist, n, directed);

    igraph_leiden_objective_t obj =
        objective == 1 ? IGRAPH_LEIDEN_OBJECTIVE_MODULARITY :
        objective == 2 ? IGRAPH_LEIDEN_OBJECTIVE_CPM :
                         IGRAPH_LEIDEN_OBJECTIVE_ER;

    igraph_vector_int_t membership;
    igraph_vector_int_init(&membership, 0);

    for (int run = 0; run < K; run++) {
        /* Re-seed per run so every run does identical work. */
        igraph_rng_seed(igraph_rng_default(), seed);
        igraph_vector_int_resize(&membership, 0);
        igraph_int_t nb_clusters = 0;
        igraph_real_t quality = 0.0;

        double t0 = now_sec();
        igraph_error_t err = igraph_community_leiden_simple(
            &g, weighted ? &weights : NULL, obj,
            resolution, beta, /* start */ 0, n_iterations,
            &membership, &nb_clusters, &quality);
        double t1 = now_sec();

        if (err != IGRAPH_SUCCESS) {
            printf("err\n");
            return 1;
        }
        printf("%.3f", (t1 - t0) * 1e6);
        if (run == K - 1) {
            printf("\n%" IGRAPH_PRId " %.17g\n", nb_clusters, quality);
        } else {
            printf(" ");
        }
    }

    igraph_vector_int_destroy(&membership);
    igraph_destroy(&g);
    igraph_vector_destroy(&weights);
    igraph_vector_int_destroy(&edgelist);
    return 0;
}
