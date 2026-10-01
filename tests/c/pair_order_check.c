#include <igraph.h>
#include <stdio.h>

/* Reads: nodes, n_edges, then pairs (v, v2) per edge; prints the pair order. */
int main(void) {
    igraph_int_t nodes, edges;
    if (scanf("%" IGRAPH_PRId " %" IGRAPH_PRId, &nodes, &edges) != 2) return 1;
    igraph_vector_int_t v, v2, res;
    igraph_vector_int_init(&v, edges);
    igraph_vector_int_init(&v2, edges);
    for (igraph_int_t i = 0; i < edges; i++) {
        igraph_int_t a, b;
        if (scanf("%" IGRAPH_PRId " %" IGRAPH_PRId, &a, &b) != 2) return 1;
        VECTOR(v)[i] = a;
        VECTOR(v2)[i] = b;
    }
    igraph_vector_int_init(&res, edges);
    igraph_vector_int_pair_order(&v, &v2, &res, nodes);
    for (igraph_int_t i = 0; i < edges; i++) {
        printf("%" IGRAPH_PRId " ", VECTOR(res)[i]);
    }
    printf("\n");
    igraph_vector_int_destroy(&v);
    igraph_vector_int_destroy(&v2);
    igraph_vector_int_destroy(&res);
    return 0;
}
