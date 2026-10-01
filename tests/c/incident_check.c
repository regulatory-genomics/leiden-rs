#include <igraph.h>
#include <stdio.h>

/* Cross-check: print incident lists (IGRAPH_ALL, LOOPS_TWICE) for a graph */
static void print_incident(const igraph_t *g) {
    igraph_inclist_t il;
    igraph_inclist_init(g, &il, IGRAPH_ALL, IGRAPH_LOOPS_TWICE);
    for (igraph_int_t v = 0; v < igraph_vcount(g); v++) {
        igraph_vector_int_t *edges = igraph_inclist_get(&il, v);
        printf("v%" IGRAPH_PRId ":", v);
        for (igraph_int_t i = 0; i < igraph_vector_int_size(edges); i++) {
            printf(" %" IGRAPH_PRId, VECTOR(*edges)[i]);
        }
        printf("\n");
    }
    igraph_inclist_destroy(&il);
}

int main(int argc, char **argv) {
    igraph_t g;
    igraph_vector_int_t edges;
    igraph_int_t n = atoi(argv[1]);
    int directed = atoi(argv[2]);

    igraph_vector_int_init(&edges, 0);
    for (int i = 3; i + 1 < argc; i += 2) {
        igraph_vector_int_push_back(&edges, atoi(argv[i]));
        igraph_vector_int_push_back(&edges, atoi(argv[i + 1]));
    }
    igraph_create(&g, &edges, n, directed ? IGRAPH_DIRECTED : IGRAPH_UNDIRECTED);
    print_incident(&g);
    igraph_destroy(&g);
    igraph_vector_int_destroy(&edges);
    return 0;
}
