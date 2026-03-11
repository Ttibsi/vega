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

    // Neuron test
    Arena a = {0};
    arenaInit(&a, sizeof(Value) * 16);
    Neuron n = newNeuron(4, &a);
    float inputs[] = {1.0, 2.0, 3.0, 4.0};
    Value v = activateNeuron(&n, inputs, 4);
    printf("Neurun value: %f\n", v.x);
    arenaDestroy(&a);

    return 0;
}
