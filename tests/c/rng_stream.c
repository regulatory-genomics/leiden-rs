#include <igraph.h>
#include <stdio.h>
int main(void) {
    igraph_rng_seed(igraph_rng_default(), 579819);
    printf("gets:");
    for (int i = 0; i < 10; i++) printf(" %u", (unsigned)igraph_rng_get_integer(igraph_rng_default(), 0, 0xFFFFFFFFLL));
    printf("\nints:");
    for (int i = 0; i < 20; i++) printf(" %" IGRAPH_PRId, igraph_rng_get_integer(igraph_rng_default(), 0, 13));
    printf("\nunif01:");
    for (int i = 0; i < 10; i++) printf(" %.17g", igraph_rng_get_unif01(igraph_rng_default()));
    printf("\n");
    return 0;
}
