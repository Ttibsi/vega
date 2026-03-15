#define MLP_IMPLEMENTATION
#include "mlp.h"

#include <stdio.h>
#include <stddef.h>

int main() {
    Arena a = {0};
    arenaInit(&a, sizeof(Value) * (1024 * 2) );

    float xor[4][2] = {
        {0.0, 0.0},
        {1.0, 0.0},
        {0.0, 1.0},
        {1.0, 1.0}
    };
    float xor_outs[4] = {0.0, 1.0, 1.0, 0.0};
    size_t xor_sizes[4] = {2, 2, 2, 2};

    size_t arch[] = {2,2,1};
    size_t arch_sz = sizeof(arch) / sizeof(arch[0]);
    MLP mlp = newPerceptron(arch, arch_sz, &a);

    size_t iterations = 100;
    float learnRate = 0.05;
    for (size_t i = 0; i < iterations; i++) {
        for (size_t j = 0; j < 4; j++) {
            trainPerceptron(&mlp, &a, xor[j], xor_sizes, xor_outs, learnRate);
        }
    }

    displayPerceptron(&mlp);
    for (size_t i = 0; i < 4; i++) {
        Value* vs = activatePerceptron(&mlp, xor[i], xor_sizes, &a);
        printf("%.3f %.3f\n", vs[0].x, vs[1].x);
    }

    arenaDestroy(&a);
    return 0;
}
