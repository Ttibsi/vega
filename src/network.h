#ifndef NETWORK_H
#define NETWORK_H

#include <stddef.h>
#include "matrix.h"

typedef struct {
    Matrix* weights;
    Matrix* biases;
    Matrix* activations;
    size_t layers;
    size_t* arch;
} Network;

Network networkAlloc(Matrix* activations, size_t* arch, size_t arch_len);
void networkRandomize(Network nn, size_t idx);
void networkPrint(Network* nn);

#endif // NETWORK_H
