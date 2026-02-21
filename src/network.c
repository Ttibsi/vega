#include "network.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

Network networkAlloc(Matrix a0, size_t* arch, size_t arch_len) {
    assert(arch_len > 1);

    Network n;
    n.layers = arch_len;
    n.arch = arch;

    n.weights = calloc(arch_len, sizeof(Matrix));
    n.biases = calloc(arch_len, sizeof(Matrix));
    n.activations = calloc(arch_len, sizeof(Matrix));
    n.activations[0] = a0;

    for (size_t i = 1; i < arch_len; i++) {
        n.weights[i-1] = matrixAlloc(n.arch[i-1], n.arch[i]);
        n.biases[i-1] = matrixAlloc(1, n.arch[i]);
        n.activations[i] = matrixAlloc(1, n.arch[i]);
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

    printf(" ]}\n");
}

float sigmoidf(float x) {
    return 1.f / (1.f + expf(x));
}

void networkFeedForward(Network nn) {
    // Start at second layer and look backward.
    // Don't run for final layer as there is no weights to calculate
    for (size_t layer = 1; layer < nn.layers - 1; layer++) {
        for (size_t neuron = 0; neuron < nn.arch[layer]; neuron++) {
            // multiply each weight by the activation on the LHS neuron
            // sum those up
            // add neuron's bias
            // sigmoidf()
            // total is the activation of the neuron on RHS

            float weight_sum = 0.f;

            for (size_t i = 0; i < nn.arch[layer-1]; i++) {
                weight_sum += nn.weights[layer-1].data[i] * nn.activations[layer-1].data[i];
            }
            float bias = nn.biases[layer-1].data[0];
            float activation = sigmoidf(weight_sum + bias);
            MATRIX_AT(nn.activations[layer], layer, neuron) = activation;
        }
    }
}

// expected = array of expected values
float networkCost(Network nn, float* expected) {
    float cost = 0.f;
    for (size_t i = 0; i < nn.arch[nn.layers - 1]; i++) {
        cost += powf(
            MATRIX_AT(nn.activations[nn.layers - 1], 0, i) - expected[i],
            2.f
        );
    }

    return cost;
}

void networkBackPropagate(Network nn) {
}

void networkPrintOutLayer(Network nn) {
    printf("Out: [");
    for (size_t i = 0; i < nn.arch[nn.layers - 1]; i++) {
        printf(" %f", MATRIX_AT(nn.activations[nn.layers - 1], i, 0));
    }
    printf(" ]\n");
}
