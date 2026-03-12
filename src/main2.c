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
    //

    arenaDestroy(&a);
    return 0;
}
