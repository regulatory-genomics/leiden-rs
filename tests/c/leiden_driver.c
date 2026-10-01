#include <igraph.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Differential driver for igraph's Leiden implementation.
 *
 * Input protocol (whitespace-separated tokens):
 *   iface            "generic" or "simple"
 *   n directed m
 *   from to weight   (m lines)
 *   resolution beta start n_iterations seed
 *   [objective]      (simple only: 1=MODULARITY 2=CPM 3=ER)
 *   membership       (n integers; used when start=1, must be range when
 *                     start=0)
 *
 * Output:
 *   nb_clusters quality
 *   membership   (n integers)
 *   err          (0 on success, 1 on igraph error)
 */

int main(void) {
    char iface[16];
    if (scanf("%15s", iface) != 1) return 2;

    igraph_int_t n, directed_i, m;
    if (scanf("%" IGRAPH_PRId " %" IGRAPH_PRId " %" IGRAPH_PRId, &n, &directed_i, &m) != 3) return 2;
    igraph_bool_t directed = directed_i != 0;

    igraph_vector_int_t edgelist;
    igraph_vector_t weights;
    igraph_vector_int_init(&edgelist, 0);
    igraph_vector_init(&weights, 0);
    for (igraph_int_t i = 0; i < m; i++) {
        igraph_int_t a, b;
        double w;
        if (scanf("%" IGRAPH_PRId " %" IGRAPH_PRId " %lf", &a, &b, &w) != 3) return 2;
        igraph_vector_int_push_back(&edgelist, a);
        igraph_vector_int_push_back(&edgelist, b);
        igraph_vector_push_back(&weights, w);
    }

    double resolution, beta;
    int start_i;
    igraph_int_t n_iterations;
    unsigned long long seed_ull;
    if (scanf("%lf %lf %d %" IGRAPH_PRId " %llu",
              &resolution, &beta, &start_i, &n_iterations, &seed_ull) != 5) return 2;
    igraph_uint_t seed = (igraph_uint_t) seed_ull;

    int objective = 0;
    if (strcmp(iface, "simple") == 0) {
        if (scanf("%d", &objective) != 1) return 2;
    }

    igraph_t g;
    igraph_create(&g, &edgelist, n, directed);

    igraph_vector_int_t membership;
    igraph_vector_int_init(&membership, n);
    for (igraph_int_t i = 0; i < n; i++) {
        if (scanf("%" IGRAPH_PRId, &VECTOR(membership)[i]) != 1) return 2;
    }

    igraph_rng_seed(igraph_rng_default(), seed);

    igraph_int_t nb_clusters = n;
    igraph_real_t quality = 0.0;
    igraph_error_t err;

    if (strcmp(iface, "generic") == 0) {
        igraph_vector_t vout, vin;
        igraph_vector_init(&vout, n);
        igraph_vector_fill(&vout, 1.0);
        igraph_vector_init(&vin, n);
        igraph_vector_fill(&vin, 1.0);
        err = igraph_community_leiden(&g, &weights, &vout,
                                      directed ? &vin : NULL,
                                      resolution, beta, start_i != 0,
                                      n_iterations, &membership,
                                      &nb_clusters, &quality);
        igraph_vector_destroy(&vout);
        igraph_vector_destroy(&vin);
    } else {
        igraph_leiden_objective_t obj =
            objective == 1 ? IGRAPH_LEIDEN_OBJECTIVE_MODULARITY :
            objective == 2 ? IGRAPH_LEIDEN_OBJECTIVE_CPM :
                             IGRAPH_LEIDEN_OBJECTIVE_ER;
        err = igraph_community_leiden_simple(&g, &weights, obj,
                                             resolution, beta, start_i != 0,
                                             n_iterations, &membership,
                                             &nb_clusters, &quality);
    }

    if (err != IGRAPH_SUCCESS) {
        printf("err 1\n");
    } else {
        printf("err 0\n");
        printf("%" IGRAPH_PRId " %.17g\n", nb_clusters, quality);
        for (igraph_int_t i = 0; i < n; i++) {
            printf("%" IGRAPH_PRId " ", VECTOR(membership)[i]);
        }
        printf("\n");
    }

    igraph_vector_int_destroy(&membership);
    igraph_destroy(&g);
    igraph_vector_destroy(&weights);
    igraph_vector_int_destroy(&edgelist);
    return 0;
}
