#include "network.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

Network networkAlloc(Matrix* activations, size_t* arch, size_t arch_len) {
    assert(arch_len > 1);
    assert(activations != NULL);

    Network n;
    n.layers = arch_len;
    n.arch = arch;
    n.activations = activations;

    n.weights = calloc(arch_len, sizeof(Matrix));
    n.biases = calloc(arch_len, sizeof(Matrix));

    for (size_t i = 1; i < arch_len; i++) {
        n.weights[i-1] = matrixAlloc(n.arch[i-1], n.arch[i]);
        n.biases[i-1] = matrixAlloc(n.arch[i-1], n.arch[i]);
    }

    return n;
}

void networkRandomize(Network nn, size_t idx) {
    assert(idx < nn.layers);

    for (size_t i = 0; i < nn.arch[idx]; i++) {
        for (size_t j = 0; j < nn.arch[idx + 1]; j++) {
            MATRIX_AT(*nn.weights, i, j) = (float)rand() / (float)RAND_MAX;
            MATRIX_AT(*nn.biases, i, j) = (float)rand() / (float)RAND_MAX;
        }
    }
}

void networkPrint(Network* nn) {
    printf("NN: {layers: %ld, arch: [", nn->layers);

    for (size_t i = 0; i < nn->layers; i++) {
        printf(" %zu", nn->arch[i]);
    }

    printf("]}\n");
}
