#define MLP_IMPLEMENTATION
#include "mlp.h"

#include <stdio.h>
#include <stddef.h>

int main() {
    Arena a = {0};
    arenaInit(&a, sizeof(Value) * 1024);

    float xor[4][2] = {
        {0.0, 0.0},
        {1.0, 0.0},
        {0.0, 1.0},
        {1.0, 1.0}
    };
    float xor_outs[4] = {0.0, 1.0, 1.0, 0.0};

    size_t arch[] = {2,2,1};
    size_t arch_sz = sizeof(arch) / sizeof(arch[0]);
    MLP mlp = newPerceptron(arch, arch_sz, &a);

    const size_t iterations = 100;
    const float learnRate = 0.05;
    for (size_t i = 0; i < iterations; i++) {
        for (size_t j = 0; j < 4; j++) {
            if (!trainPerceptron(&mlp, &a, xor[j], 2, &xor_outs[j], learnRate)) {
                goto finish;
            }
        }
    }

finish:
    displayPerceptron(&mlp);
    arenaDestroy(&a);
    return 0;
}
