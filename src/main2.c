#include <stdio.h>

#define MLP_IMPLEMENTATION
#include "mlp.h"

int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;

    /*
    // Value test
    Value a = newValue(4.0, NULLPREV, OP_NONE);
    Value b = newValue(3.0, NULLPREV, OP_NONE);
    Value c = plusValue(&a, &b);
    printf("%f + %f = %f\n", a.x, b.x, c.x);
    */

    Arena a = {0};
    arenaInit(&a, sizeof(Value) * 1024);

    // Neuron test
    arenaReset(&a);
    Neuron n = newNeuron(4, NULLPREV, &a);
    float inputs[] = {1.0, 2.0, 3.0, 4.0};
    Value v = activateNeuron(&n, inputs, 4);
    printf("Neuron value: %f\n", v.x);

    // Layer test
    arenaReset(&a);
    Layer l = newLayer(NULL, 2, &a);
    Value* vs = activateLayer(&l, inputs, 4, &a);
    for (size_t i = 0; i < 4; i++) {
        printf("Neuron value(%zu): %f\n", i, vs[i].x);
    }

    // MLP test
    arenaReset(&a);
    size_t arch[] = {2, 2, 1};
    size_t arch_sz = sizeof(arch) / sizeof(arch[0]);
    MLP mlp = newPerceptron(arch, arch_sz, &a);

    arenaDestroy(&a);
    return 0;
}
